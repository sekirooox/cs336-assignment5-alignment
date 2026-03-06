from torch.utils.data import Dataset,DataLoader
from typing import Tuple, List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import itertools
from utils import load_template,get_r1_prompts,get_r1_ground_truths,get_r1_ground_truths_with_template
from dataclasses import dataclass
import torch
from vllm import LLM,SamplingParams
from vllm_utils import load_policy_into_vllm_instance,log_generation,evaluate_vllm
import random
from sft import *
import wandb
import os
from drgrpo_grader import r1_zero_reward_fn
from sft_config import SFTConfig

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
        return self.prompts[idx], self.ground_truths[idx]

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
        vllm: LLM = None,
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
        self.dataloader = DataLoader(
            dataset = self.train_dataset,
            batch_size = config.batch_size ,
            shuffle = True,
            drop_last = False,
            collate_fn= lambda batch: sft_collate_fn(batch,self.tokenizer)
        )
        # 变成循环迭代器
        self.data_iter = itertools.cycle(self.dataloader)
        
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
            self.config.cosine_schedule_iters
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
            self.model.train()
            self.update_lr(it,self.optimizer)
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
                wandb.log(log_dict,step=it)
            
            if it > 0 and (it % self.config.save_interval == 0 or it == self.config.max_iters-1):
                print(f"\n💾 Saving checkpoint at iteration {it}...")
                os.makedirs(self.config.save_dir,exist_ok=True)
                save_it_dir = os.path.join(self.config.save_dir,f'checkpoint_{it}')
                os.makedirs(save_it_dir,exist_ok=True)
                self.save_checkpoint(save_it_dir)
                print(f"✅ Checkpoint saved to {save_it_dir}\n")
                
        wandb.finish()
            


    @torch.no_grad()
    def evaluate(self):
        return evaluate_vllm(
            self.vllm,
            self.test_dataset.prompts,
            self.test_dataset.answers, # 只需要答案
            self.config.sampling_params,
            self.config.reward_fn,
        )

    def save_checkpoint(self,path:str):
        self.model.save_pretrained(save_directory=path)
        self.tokenizer.save_pretrained(save_directory=path)
        print('save checkpoint successfully!')

        


                



 



                


        
