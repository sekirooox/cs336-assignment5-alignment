from torch.utils.data import Dataset,DataLoader
from typing import Tuple, List, Dict, Any, Optional, Callable
from transformers import AutoModelForCausalLM, AutoTokenizer
import itertools
from utils import *
from dataclasses import dataclass
import torch
from vllm import LLM,SamplingParams
from vllm_utils import *
import random
from sft import *
import wandb
import os
from drgrpo_grader import r1_zero_reward_fn
from config import *
from grpo import *

def sft_collate_fn(batch: List[Tuple[str, str]], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """
    用于SFT数据集的批处理函数，将一批样本整理成模型输入格式。
    Example:
        >>> batch = [("What is 2+2?", "4"), ("What is capital of France?", "Paris")]
        >>> model_inputs = sft_collate_fn(batch, tokenizer)
        >>> print(model_inputs.keys())
        dict_keys(['input_ids', 'attention_mask', 'labels'])
    """
    # 解压批次数据，将prompts和ground_truths分开
    prompts, ground_truths = zip(*batch)
    
    # 将文本列表转换为模型输入格式
    return tokenize_prompt_and_output(
        prompt_strs=list(prompts),
        output_strs=list(ground_truths),
        tokenizer=tokenizer
    )

class SFTDataset(Dataset):
    """
    监督微调（Supervised Fine-Tuning）数据集类。
    """
    def __init__(
        self,
        json_path: str,
        prompt_template_path: str,
    ) -> None:
        self.json_path = json_path
        self.template = load_template(prompt_template_path)
        
        # 从JSON文件加载并格式化prompts和ground_truths
        self.prompts = get_r1_prompts(self.json_path, self.template)

        # 用于训练的ground_truths
        self.ground_truths = get_r1_ground_truths_with_template(self.json_path)

        # 用于验证的ground_truths（仅包含答案）
        self.answers = get_r1_ground_truths(self.json_path)
        
        # 确保数据长度一致
        assert len(self.prompts) == len(self.ground_truths), \
            f"Prompts length ({len(self.prompts)}) does not match ground_truths length ({len(self.ground_truths)})"
    
    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, idx: int) -> Tuple[str, str]:
        """
        Example:
            >>> prompt, ground_truth = dataset[0]
            Prompt: Question: What is the capital of France?\nAnswer:
            Ground truth: Paris
        """
        # BUG: 后面才增加的self.answers 前面需要修改
        return self.prompts[idx], self.ground_truths[idx], self.answers[idx]
    
    @classmethod
    def from_prompts_and_ground_truths(cls,
        prompts:List[str],
        ground_truths:List[str],
        answers: List[str]
    ):
        assert len(prompts) == len(ground_truths) == len(answers), (
            f"Length mismatch: prompts={len(prompts)}, "
            f"ground_truths={len(ground_truths)}, answers={len(answers)}"
        )

        dummy = cls.__new__(cls)  # 空实例,跳过构造函数
        dummy.json_path = "<from_lists>"
        dummy.template = "<from_lists>"
        dummy.prompts = prompts
        dummy.ground_truths = ground_truths
        dummy.answers = answers
        return dummy
        


import math
# cosine 动态学习率
def get_lr_cosine_schedule_with_warmup(
    it: int,
    max_lr: float,
    min_lr: float,
    warmup_iters: int,
    cosine_schedule_iters: int,
) -> float:
    """
    带有预热（warmup）的余弦退火学习率调度函数。
    该函数实现了一个三段式的学习率调度策略：
    1. Warmup阶段：学习率从0线性增加到max_lr
    2. Cosine decay阶段：学习率按照余弦函数从max_lr下降到min_lr
    3. 退火结束后：保持min_lr不变
    """
    # 1. warmup 阶段
    if it < warmup_iters:
        print(f"Warmup phase: it={it}, warmup_iters={warmup_iters}, cur_lr={max_lr * it / warmup_iters:.8f}")
        return max_lr * it / warmup_iters

    # 2. 退火结束后
    if it > cosine_schedule_iters:
        return min_lr
    

    # 3. cosine decay 阶段
    decay_ratio = (it - warmup_iters) / (cosine_schedule_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    lr = min_lr + coeff * (max_lr - min_lr)
    return lr

class SFTTrainer:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer : AutoTokenizer,
        optimizer : torch.optim.Optimizer,
        config: SFTConfig, 
        vllm: LLM,
    ):
        # AutoModelFromPretrained:训练模型
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.config =  config
        
        # vllm：离线推理模型
        self.vllm = vllm

        # Dataset
        self.train_dataset =  SFTDataset(
            json_path = config.train_dataset_path,
            prompt_template_path=config.prompt_template_path,
        )

        self.test_dataset = SFTDataset(
            json_path = config.test_dataset_path,
            prompt_template_path=config.prompt_template_path,
        )
        # 将train_dataset保存为内存中,使用sft_collate_fn加载到GPU中
        # 非循环迭代器
        self.dataloader = self._create_dataloader(self.train_dataset, config.batch_size)
        # 变成循环迭代器
        self.data_iter = itertools.cycle(self.dataloader)

    def _create_dataloader(self, dataset, batch_size):
        """创建 DataLoader 并返回循环迭代器"""
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=lambda batch: sft_collate_fn(batch, self.tokenizer)
        )
        return itertools.cycle(dataloader)

    @classmethod
    def from_ei_trainer(
        cls,
        ei_trainer,# EI Trainer
        train_dataset: SFTDataset,# from prompts and ground_truths
    ):
        dummy = cls.__new__(cls)  # 空实例,跳过构造函数
        dummy.model = ei_trainer.model
        dummy.tokenizer = ei_trainer.tokenizer
        dummy.optimizer = ei_trainer.optimizer
        dummy.config = ei_trainer.config
        dummy.vllm = ei_trainer.vllm

        # 初始化数据集
        dummy.train_dataset = train_dataset
        dummy.test_dataset = SFTDataset(
            json_path = dummy.config.test_dataset_path,
            prompt_template_path=dummy.config.prompt_template_path,
        )

        # 使用辅助方法创建 DataLoader
        dummy.dataloader = dummy._create_dataloader(dummy.train_dataset, dummy.config.batch_size)
        return dummy

    @torch.no_grad()
    def sample_responses(self)->tuple[List[str],List[str]]:
        assert len(self.test_dataset.prompts) == len(self.test_dataset.answers), \
    f"数据集长度不匹配: prompts {len(self.test_dataset.prompts)} vs answers {len(self.test_dataset.answers)}" 
        indices = range(len(self.test_dataset.prompts))
        sampled_indices = random.sample(indices,self.config.sample_size)
        sampled_prompts = [self.test_dataset.prompts[idx] for idx in sampled_indices]
        sampled_answers = [self.test_dataset.answers[idx] for idx in sampled_indices]
        return sampled_prompts,sampled_answers

    @torch.no_grad()
    def update_lr(self,it:int,optimizer:torch.optim.Optimizer):
        lr = get_lr_cosine_schedule_with_warmup(
            it,
            self.config.max_lr,
            self.config.min_lr,
            self.config.warmup_iters,
            self.config.cosine_schedule_iters# 这个值作用于全局
        )
        for param_group in optimizer.param_groups:
            param_group['lr']=lr


# + run_tokenize_prompt_and_output获取prompt,mask和labels
# + run_get_response_log_probs：给定inputs和labels,输出outputs的log_prob
# + run_sft_microbatch_train_step：给定log_prob和mask,进行mask求和与梯度累积
    def train_step(self)->dict[str:float,str:float]:
    # 每一个梯度累积算一次step
        # eval metrics
        total_batch_loss = 0.0
        total_batch_entropy = 0.0
        total_valid_tokens = 0.0

        for step in range(self.config.gradient_accumulation_steps):
            inputs = next(self.data_iter)
            input_ids = inputs['input_ids'].to(self.model.device)
            labels = inputs['labels'].to(self.model.device)
            response_mask = inputs['response_mask'].to(self.model.device)

            # 有效的tokens数量: 用于计算平均token熵

            
            log_probs_entropy = get_response_log_probs(
                model = self.model,
                input_ids = input_ids,
                labels = labels,
                return_token_entropy= True,
            )

            policy_log_probs = log_probs_entropy['log_probs']

            with torch.no_grad():
                valid_tokens = response_mask.sum().item()
                entropy = log_probs_entropy['token_entropy'] # b l
                masked_entropy = entropy * response_mask

            scaled_loss,metadata = sft_microbatch_train_step(
                policy_log_probs= policy_log_probs,
                response_mask = response_mask,
                gradient_accumulation_steps= self.config.gradient_accumulation_steps,
                normalize_constant=self.config.normalize_constant,
            )# 已经反向传播
            # TODO
            # # eval metrics
            total_batch_loss += scaled_loss.item()
            total_batch_entropy += masked_entropy.sum().item()# b l -> 1
            total_valid_tokens += valid_tokens
            del input_ids,labels,response_mask,log_probs_entropy,policy_log_probs,masked_entropy

        avg_batch_loss = total_batch_loss / (self.config.gradient_accumulation_steps)
        avg_batch_entropy = total_batch_entropy / (total_valid_tokens+1e-6)
        
        torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm = self.config.max_grad_norm,
                norm_type=2
            )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        # 缓存清理
        clear_gpu_memory()

        return {
            'avg_batch_entropy':avg_batch_entropy,
            'avg_batch_loss':avg_batch_loss,
        }
    
    def train(self,
        global_start_step:int =0, # 全局起始步数
        ):
        print(f"🚀 Starting training from iteration {self.config.start_iters} to {self.config.max_iters}")
        print(f"📊 Training configuration: batch_size={self.config.batch_size}, gradient_accumulation_steps={self.config.gradient_accumulation_steps}")
        print(f"💾 Checkpoints will be saved every {self.config.save_interval} steps to {self.config.save_dir}")
        print(f"📈 Evaluation will be performed every {self.config.eval_interval} steps\n")
        log_dict = {}
        for it in range(self.config.start_iters,self.config.max_iters):
            self.model.train()
            
            # EI：全局步数作为学习率调度的输入
            global_it = it + global_start_step # 0 + 60 = 60
            self.update_lr(global_it,self.optimizer)
            train_step_log = self.train_step()

            if it % 10 == 0:  # 每10步打印一次训练状态
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"⏱️  Iteration {it}/{self.config.max_iters} | Loss: {train_step_log['avg_batch_loss']:.4f} | Entropy: {train_step_log['avg_batch_entropy']:.4f} | LR: {current_lr:.2e}")

            log_dict ['train_avg_loss'] = train_step_log['avg_batch_loss']
            log_dict ['train_avg_token_entropy'] = train_step_log['avg_batch_entropy']


            if it % self.config.eval_interval == 0 or it == self.config.max_iters-1:
                # 必须部署vllm
                load_policy_into_vllm_instance(self.model,self.vllm)
                sampled_prompts, sampled_answers = self.sample_responses()

                # 对于sampled的结果进行输出
                sampled_overview = log_generation(
                    sampled_prompts,
                    sampled_answers,
                    self.config.reward_fn,
                    self.model,
                    self.tokenizer,
                    self.vllm,
                    self.config.sampling_params,
                )
                sampled_summary = sampled_overview['summary']

                log_dict['sampled_avg_reward'] = sampled_summary['avg_reward']
                log_dict['sampled_avg_token_entropy'] = sampled_summary['avg_token_entropy']
                log_dict['sampled_avg_resp_len'] = sampled_summary['avg_resp_len']
                log_dict['sampled_avg_len_correct'] = sampled_summary['avg_len_correct']
                log_dict['sampled_avg_len_wrong'] = sampled_summary['avg_len_wrong']
                print(f"✨ Sampled responses summary: Avg Reward={sampled_summary['avg_reward']:.4f}, Avg Resp Len={sampled_summary['avg_resp_len']:.2f}")
                print(sampled_overview)

                print(f"🧪 Running full test evaluation...")
                test_overview = self.evaluate()
                log_dict['eval_answer_correct'] = test_overview['answer_correct']
                log_dict['eval_format_correct'] = test_overview['format_correct']
                log_dict['eval_total_correct'] = test_overview['total_correct']
                log_dict['eval_format_correct_but_answer_wrong'] = test_overview['format_correct_but_answer_wrong']
                log_dict['eval_answer_correct_but_format_wrong'] = test_overview['answer_correct_but_format_wrong']
                log_dict['eval_total_wrong'] = test_overview['total_wrong']  
                log_dict['eval_accuracy'] = test_overview['accuracy']
                log_dict['eval_wrong_rate'] = test_overview['wrong_rate']
                print(f"📊 Test results - Accuracy: {test_overview['accuracy']:.2%}, Correct: {test_overview['total_correct']}, Wrong: {test_overview['total_wrong']}")
                print(test_overview)

                log_dict['global_step'] = global_it
                wandb.log(log_dict,step=global_it)
            

            if it > 0 and (it % self.config.save_interval == 0 or it == self.config.max_iters-1):
                if self.config.cosine_schedule_iters > self.config.max_iters:
                    # 说明这是EI or GRPO迭代,不需要保存
                    print('No checkpoint saving during SFT phase (EI or GRPO training).')
                else:
                    print(f"\n💾 Saving checkpoint at iteration {it}...")
                    os.makedirs(self.config.save_dir,exist_ok=True)
                    save_it_dir = os.path.join(self.config.save_dir,f'checkpoint_{it}')
                    os.makedirs(save_it_dir,exist_ok=True)
                    self.save_checkpoint(save_it_dir)
                    print(f"✅ Checkpoint saved to {save_it_dir}\n")

    @torch.no_grad()
    def evaluate(self):
        return evaluate_vllm(
            self.vllm,
            self.test_dataset.prompts,
            self.test_dataset.answers, # 只需要答案
            self.config.sampling_params,# EI和SFT共用一个采样参数是不公平的
            self.config.reward_fn,
        )

    def save_checkpoint(self,path:str):
        self.model.save_pretrained(save_directory=path)
        self.tokenizer.save_pretrained(save_directory=path)
        print('save checkpoint successfully!')

        
# EIDataset: 在训练阶段用于替换SFTDataset
class EIDataset(SFTDataset):
    def __init__(
        self,
        vllm:LLM ,
        sampling_params: SamplingParams,# 保持有一定随机性
        reward_fn : Callable,
        json_path:str,# only training set
        prompt_template_path:str,
        sft_sample_size:int ,
    ):
        """
        从SFTDataset训练集中采集sft_sample_size个样本
        然后使用vllm进行rollout采样,保留答案为正确的responses
        """
        super().__init__(json_path,prompt_template_path)
        # vllm
        self.vllm = vllm
        self.sampling_params = sampling_params

        # rollout size is defined in sampling_params.n
        self.sft_sample_size = sft_sample_size
        self.reward_fn  = reward_fn

    @torch.no_grad()
    def sample_responses(self)->tuple[List[str],List[str]]:
        assert len(self.prompts) == len(self.answers), \
    f"数据集长度不匹配: prompts {len(self.prompts)} vs answers {len(self.answers)}"
        indices = range(len(self.prompts))
        sampled_indices = random.sample(indices,self.sft_sample_size)
        sampled_prompts = [self.prompts[idx] for idx in sampled_indices]
        sampled_answers = [self.answers[idx] for idx in sampled_indices]
        return sampled_prompts,sampled_answers

    @torch.no_grad()
    def get_ei_batch(
        self,
    )->tuple[List[str],List[str],List[str]]:
        sampled_questions,sampled_answers = self.sample_responses()
        rollouts: List[List[str]] = generate_rollouts(self.vllm, sampled_questions, self.sampling_params)

        rollout_prompts = []
        rollout_ground_truths = []
        rollout_answers = []

        for i in range(len(rollouts)):
            # ith prompt,ith answer
            for response in rollouts[i]:
                reward_dict = self.reward_fn(response,sampled_answers[i],fast=True)
                if reward_dict['reward'] == 1.0:
                    rollout_prompts.append(sampled_questions[i])
                    rollout_ground_truths.append(response)
                    rollout_answers.append(sampled_answers[i])
        return rollout_prompts, rollout_ground_truths, rollout_answers
        
                
class EITrainer:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer : AutoTokenizer,
        optimizer : torch.optim.Optimizer,
        config: EIConfig, 
        vllm: LLM,
    ):
        # 模型和优化器配置
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.config =  config
        
        # vllm：离线推理模型
        self.vllm = vllm
        self.ei_dataset =  EIDataset(
            vllm = vllm,
            sampling_params = self.config.ei_sampling_params,
            reward_fn = self.config.reward_fn,
            json_path = self.config.train_dataset_path,
            prompt_template_path=self.config.prompt_template_path,
            sft_sample_size = self.config.sft_sample_size
        )
    def train(self):
        global_start_step = 0
        for iteration in range(self.config.ei_iterations):
            load_policy_into_vllm_instance(self.model,self.vllm)# \theta_old
            rollout_prompts, rollout_ground_truths, rollout_answers = self.ei_dataset.get_ei_batch()# sample_responses -> vllm evaluate
            if len(rollout_prompts) == 0:
                print(f"Iteration {iteration+1}: No valid samples obtained for SFT training. Skipping this iteration.")
                continue

            print(f"Iteration {iteration+1}: obtained {len(rollout_prompts)} valid samples for SFT training.")

            # 从现有prompt和ground_truths中构建一个新的SFT数据集
            train_dataset = SFTDataset.from_prompts_and_ground_truths(
                prompts=rollout_prompts,
                ground_truths=rollout_ground_truths,
                answers=rollout_answers,
            )

            sft_trainer = SFTTrainer.from_ei_trainer(
                ei_trainer=self,
                train_dataset=train_dataset,
            )
            
            sft_trainer.train(global_start_step)
            global_start_step += self.config.max_iters # +=60

            # 缓存清理
            clear_gpu_memory()

            # 只保存最后一个迭代的模型
            if iteration == self.config.ei_iterations-1:
                # 显示调用save_checkpoint. train函数中save_interval设置为无穷大
                os.makedirs(self.config.save_dir, exist_ok=True)
                save_it_dir = os.path.join(self.config.save_dir,f'EI_iteration_{iteration+1}')
                os.makedirs(save_it_dir, exist_ok=True)
                sft_trainer.save_checkpoint(path=save_it_dir)

class GRPODataset(SFTDataset):
    def __init__(
        self,
        vllm:LLM ,
        config : GRPOConfig,
    ):
        """
        然后使用vllm进行rollout采样,保留advantages不为0的responses
        """
        # vllm
        self.vllm = vllm
        self.config = config
        super().__init__(self.config.train_dataset_path,self.config.prompt_template_path)

    @torch.no_grad()
    def sample_responses(self)->tuple[List[str],List[str]]:
        assert len(self.prompts) == len(self.answers), \
    f"数据集长度不匹配: prompts {len(self.prompts)} vs answers {len(self.answers)}"
        indices = range(len(self.prompts))
        sampled_indices = random.sample(indices,self.config.vllm_prompt_batch_size)
        sampled_prompts = [self.prompts[idx] for idx in sampled_indices]
        sampled_answers = [self.answers[idx] for idx in sampled_indices]
        return sampled_prompts,sampled_answers

    @torch.no_grad()
    def get_rollout_batch(self) -> tuple[list[str], list[str], list[str]]:
        """
        使用 vLLM 离线采样一批 GRPO 训练数据。

        逻辑：
        - while 循环：
            - 每次调用 sample_responses 随机采样 vllm_prompt_batch_size 个 (prompt, answer)
            - 用 vLLM 对这些 prompt rollout，得到每个 prompt 的 group_size 个 response
            - 计算每个 prompt 组的 advantages
            - 只保留 advantages 不全为 0 的整组：
                - prompt 重复 group_size 次
                - 对应的 answer 也重复 group_size 次
                - 收集对应的 responses
        - 直到收集到的样本数 >= rollout_batch_size 为止

        返回：
            rollout_prompts:  展平后的 prompt 列表，长度 = rollout_batch_size
            rollout_responses:展平后的 response 列表，长度 = rollout_batch_size
            rollout_answers:  展平后的 answer 列表（与 prompts/responses 一一对应），长度 = rollout_batch_size
        """
        assert (
            self.config.group_size == self.config.grpo_sampling_params.n
        ), "group_size must be equal to grpo_sampling_params.n for GRPO training."

        rollout_prompts: list[str] = []
        rollout_responses: list[str] = []
        rollout_answers: list[str] = []

        target_size: int = self.config.rollout_batch_size
        vllm_prompt_batch_size: int = self.config.vllm_prompt_batch_size

        max_loops: int = 1000  # 循环上限，防止理论上的无限循环
        loop_cnt: int = 0

        while len(rollout_prompts) < target_size:
            loop_cnt += 1
            if loop_cnt > max_loops:
                raise RuntimeError(
                    f"get_rollout_batch reached max_loops={max_loops} "
                    f"but only collected {len(rollout_prompts)} samples. "
                    "可能是 advantages 几乎总为 0，请检查 reward_fn 或采样配置。"
                )

            # 1. 随机采样一批 prompt / answer
            sampled_prompts, sampled_answers = self.sample_responses()
            assert len(sampled_prompts) == len(sampled_answers) == vllm_prompt_batch_size, \
                "sample_responses 返回的数量与 vllm_prompt_batch_size 不一致。"

            # 2. 用 vLLM 批量 rollout（每个 prompt 生成 group_size 个 response）
            responses_batch: list[list[str]] = generate_rollouts(
                vllm=self.vllm,
                prompts=sampled_prompts,
                sampling_params=self.config.grpo_sampling_params,
                use_tqdm=True,
            )  # 形状: [batch_size, group_size]

            # 3. 对这一批中的每个 prompt 分别计算 advantages 并筛选
            for p_j, ans_j, responses_per_prompt in zip(
                sampled_prompts, sampled_answers, responses_batch
            ):
                if len(responses_per_prompt) != self.config.group_size:
                    # 理论上不应该发生，防御性检查
                    continue

                repeated_answers: list[str] = [ans_j] * self.config.group_size

                advantages_per_prompt, raw_rewards_per_prompt, _ = compute_group_normalized_rewards(
                    reward_fn=self.config.reward_fn,
                    rollout_responses=responses_per_prompt,
                    repeated_ground_truths=repeated_answers,
                    group_size=self.config.group_size,
                    advantage_eps=self.config.advantage_eps,
                    normalize_by_std=self.config.normalize_by_std,
                )

                # 若这一组 advantages 全为 0，则跳过
                if advantages_per_prompt.sum().item() == 0.0:
                    del advantages_per_prompt, raw_rewards_per_prompt
                    continue

                # 否则保留这一整组：
                # - prompt 重复 group_size 次
                # - 对应的 answer 也重复 group_size 次
                # - 收集 responses
                rollout_prompts.extend([p_j] * self.config.group_size)
                rollout_responses.extend(responses_per_prompt)
                rollout_answers.extend(repeated_answers)

                del advantages_per_prompt, raw_rewards_per_prompt

                # 如果已经收集够了，就可以提前结束 while
                if len(rollout_prompts) >= target_size:
                    break

        # 截断到 rollout_batch_size，确保长度一致
        rollout_prompts = rollout_prompts[:target_size]
        rollout_responses = rollout_responses[:target_size]
        rollout_answers = rollout_answers[:target_size]

        assert len(rollout_prompts) == len(rollout_responses) == len(rollout_answers), \
            "Length mismatch among prompts, responses and answers."

        return rollout_prompts, rollout_responses, rollout_answers

class GRPOTrainer(SFTTrainer):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer : AutoTokenizer,
        optimizer : torch.optim.Optimizer,
        config: GRPOConfig,
        vllm: LLM,
    ):
        # 调用父类初始化（已设置 model, tokenizer, optimizer, config, vllm）
        super().__init__(model, tokenizer, optimizer, config, vllm)

        # 替换为 GRPODataset
        self.train_dataset = GRPODataset(
            vllm = self.vllm,
            config = self.config
        )

    def train_step(self,global_it:int)->dict[str:float,str:float]:
        # eval metrics
        total_batch_loss = 0.0
        total_batch_entropy = 0.0
        gradient_update_step: int = 0

        # old policy model
        load_policy_into_vllm_instance(self.model, self.vllm)

        # get rollout_batch List[str]: rollout_batch_size
        rollout_prompts, rollout_responses, rollout_answers = self.train_dataset.get_rollout_batch()

        # 总共需要迭代的次数 32 = 256 // 8
        n_micro_batch = self.config.rollout_batch_size // self.config.micro_batch_size
        
        # 获得所有old_log_probs的列表 on CPU 
        old_log_probs_list: List[torch.tensor] = []
        input_ids_list: List[torch.tensor] = []
        labels_list: List[torch.tensor] = []
        response_mask_list: List[torch.tensor] = []
        advantages_list: List[torch.tensor] = []
        rewards_list: List[torch.tensor] = []

        # 预先获得所有microbatch
        for i in range(n_micro_batch):
            start_idx = i * self.config.micro_batch_size
            end_idx = start_idx + self.config.micro_batch_size
            micro_batch_prompts = rollout_prompts[start_idx:end_idx]
            micro_batch_responses = rollout_responses[start_idx:end_idx]
            micro_batch_answers = rollout_answers[start_idx:end_idx]

            micro_batch_advantages, micro_batch_raw_rewards , _ = compute_group_normalized_rewards(
                reward_fn=self.config.reward_fn,
                rollout_responses=micro_batch_responses,
                repeated_ground_truths=micro_batch_answers,
                group_size=self.config.group_size,
                advantage_eps=self.config.advantage_eps,
                normalize_by_std=self.config.normalize_by_std,
            )
            micro_batch_advantages = micro_batch_advantages.to(self.model.device)
            micro_batch_raw_rewards = micro_batch_raw_rewards.to(self.model.device)

            # input_ids, labels, response_mask
            inputs = tokenize_prompt_and_output(
                micro_batch_prompts, micro_batch_responses, self.tokenizer
            )
            input_ids = inputs['input_ids'].to(self.model.device)
            labels = inputs['labels'].to(self.model.device)
            response_mask = inputs['response_mask'].to(self.model.device)


            # old_log_probs
            with torch.inference_mode():# fast inference, no_grad
                old_log_probs_token_entropy = get_response_log_probs(# require_grad == False
                    model= self.model,
                    input_ids = input_ids,
                    labels = labels,
                    return_token_entropy=False)
                # 去除计算图
            
            # NOTE: 不需要常驻GPU
            micro_batch_advantages = micro_batch_advantages.to('cpu')
            micro_batch_raw_rewards = micro_batch_raw_rewards.to('cpu')
            advantages_list.append(micro_batch_advantages)
            rewards_list.append(micro_batch_raw_rewards)
            input_ids = input_ids.to('cpu')
            labels = labels.to('cpu')
            response_mask = response_mask.to('cpu')
            input_ids_list.append(input_ids)
            labels_list.append(labels)
            response_mask_list.append(response_mask)
            old_log_probs = old_log_probs_token_entropy['log_probs'].detach().to('cpu')
            old_log_probs_list.append(old_log_probs)
            
        clear_gpu_memory()

        for epoch in range(self.config.n_train_steps_per_rollout_batch):
            for i in range(n_micro_batch):
                input_ids = input_ids_list[i].to(self.model.device)
                labels = labels_list[i].to(self.model.device)
                response_mask = response_mask_list[i].to(self.model.device)
                advantages = advantages_list[i].to(self.model.device)
                raw_rewards = rewards_list[i].to(self.model.device)
                old_log_probs = old_log_probs_list[i].to(self.model.device)

                # policy_log_probs: torch.tensor: b*g l
                policy_log_probs_token_entropy = get_response_log_probs(
                    model= self.model, 
                    input_ids = input_ids,
                    labels = labels,
                    return_token_entropy=False# save GPU memory   
                )
                policy_log_probs = policy_log_probs_token_entropy['log_probs'].to(self.model.device)
                
                # 函数包含反向传播, 更新策略模型 
                scaled_loss,loss_metadata = grpo_microbatch_train_step(
                    policy_log_probs = policy_log_probs,
                    response_mask = response_mask,
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                    loss_type = self.config.loss_type,
                    raw_rewards = raw_rewards,
                    advantages = advantages,
                    old_log_probs = old_log_probs,
                    cliprange= self.config.clip_range,
                    # 长度归一化 Dr.GRPO
                    normalize_by_length= self.config.normalize_by_length,
                    normalize_constant = self.config.normalize_constant
                )
                total_batch_loss += scaled_loss.item()
                gradient_update_step += 1

                # 每隔 gradient_accumulation_steps步进行一次梯度更新
                if gradient_update_step % self.config.gradient_accumulation_steps ==0:
                    # 梯度累积更新
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm = self.config.max_grad_norm,
                        norm_type=2
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)  
                del input_ids, labels, response_mask, advantages, raw_rewards
                del old_log_probs, policy_log_probs, policy_log_probs_token_entropy
                # clear_gpu_memory()# -6k
        
        avg_batch_loss = total_batch_loss 
        avg_batch_entropy = total_batch_entropy / self.config.gradient_accumulation_steps
        # 缓存清理
        clear_gpu_memory()
        return {
            'avg_batch_entropy':avg_batch_entropy,
            'avg_batch_loss':avg_batch_loss,
        }

    def train(self):
        print(f"🚀 Starting training from iteration {self.config.start_iters} to {self.config.max_iters}")
        print(f"📊 Training configuration: batch_size={self.config.batch_size}, gradient_accumulation_steps={self.config.gradient_accumulation_steps}")
        print(f"💾 Checkpoints will be saved every {self.config.save_interval} steps to {self.config.save_dir}")
        print(f"📈 Evaluation will be performed every {self.config.eval_interval} steps\n")
        log_dict = {}
        for it in range(self.config.start_iters,self.config.max_iters):
            it += 1
            self.model.train()
            self.update_lr(it,self.optimizer)
            train_step_log = self.train_step(global_it=it)

            if it % 10 == 0:  # 每10步打印一次训练状态
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"⏱️  Iteration {it}/{self.config.max_iters} | Loss: {train_step_log['avg_batch_loss']:.4f} | Entropy: {train_step_log['avg_batch_entropy']:.4f} | LR: {current_lr:.2e}")

            log_dict ['train_avg_loss'] = train_step_log['avg_batch_loss']
            log_dict ['train_avg_token_entropy'] = train_step_log['avg_batch_entropy']
            
            if it % self.config.eval_interval == 0 or it == self.config.max_iters:
                # 必须部署vllm
                load_policy_into_vllm_instance(self.model,self.vllm)
                sampled_prompts, sampled_answers = self.sample_responses()

                # 对于sampled的结果进行输出
                sampled_overview = log_generation(
                    sampled_prompts,
                    sampled_answers,
                    self.config.reward_fn,
                    self.model,
                    self.tokenizer,
                    self.vllm,
                    self.config.sampling_params,
                )
                sampled_summary = sampled_overview['summary']

                log_dict['sampled_avg_reward'] = sampled_summary['avg_reward']
                log_dict['sampled_avg_token_entropy'] = sampled_summary['avg_token_entropy']
                log_dict['sampled_avg_resp_len'] = sampled_summary['avg_resp_len']
                log_dict['sampled_avg_len_correct'] = sampled_summary['avg_len_correct']
                log_dict['sampled_avg_len_wrong'] = sampled_summary['avg_len_wrong']
                print(f"✨ Sampled responses summary: Avg Reward={sampled_summary['avg_reward']:.4f}, Avg Resp Len={sampled_summary['avg_resp_len']:.2f}")
                print(sampled_overview)

                print(f"🧪 Running full test evaluation...")
                test_overview = self.evaluate()
                log_dict['eval_answer_correct'] = test_overview['answer_correct']
                log_dict['eval_format_correct'] = test_overview['format_correct']
                log_dict['eval_total_correct'] = test_overview['total_correct']
                log_dict['eval_format_correct_but_answer_wrong'] = test_overview['format_correct_but_answer_wrong']
                log_dict['eval_answer_correct_but_format_wrong'] = test_overview['answer_correct_but_format_wrong']
                log_dict['eval_total_wrong'] = test_overview['total_wrong']  
                log_dict['eval_accuracy'] = test_overview['accuracy']
                log_dict['eval_wrong_rate'] = test_overview['wrong_rate']
                print(f"📊 Test results - Accuracy: {test_overview['accuracy']:.2%}, Correct: {test_overview['total_correct']}, Wrong: {test_overview['total_wrong']}")
                print(test_overview)

                log_dict['global_step'] = it
                wandb.log(log_dict,step=it)
            

            if  it % self.config.save_interval == 0 or it == self.config.max_iters:
                print(f"\n💾 Saving checkpoint at iteration {it}...")
                os.makedirs(self.config.save_dir,exist_ok=True)
                save_it_dir = os.path.join(self.config.save_dir,f'checkpoint_{it}')
                os.makedirs(save_it_dir,exist_ok=True)
                self.save_checkpoint(save_it_dir)
                print(f"✅ Checkpoint saved to {save_it_dir}\n")


        
        
