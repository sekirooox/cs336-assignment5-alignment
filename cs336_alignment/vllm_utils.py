from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch
from vllm import LLM,SamplingParams
from transformers import PreTrainedModel,AutoTokenizer,AutoModelForCausalLM
from typing import Union,List,Callable,Dict,Any,Tuple,Literal,Optional
import torch
from sft import tokenize_prompt_and_output,get_response_log_probs
from utils import *

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    启动推理过程，此处使用vLLM将模型部署在与策略模型不同的GPU上。
    """
    vllm_set_random_seed(seed)
    # 从TRL借鉴的Monkeypatch：https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # 对vLLM进行补丁，确保：
    # （1）将vLLM模型部署到指定设备（world_size_patch）；
    # （2）跳过不适合当前场景的测试（profiling_patch）。
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    从https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670复制
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def generate_responses(
    vllm: LLM,
    prompts: Union[str, List[str]],
    sampling_params: SamplingParams
) -> List[str]:
    """
    使用vLLM生成响应
    """    
    # 生成响应
    outputs = vllm.generate(prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    return responses

# EI
@torch.no_grad()
def generate_rollouts(
    vllm:LLM,
    prompts:List[str],
    sampling_params:SamplingParams,
    use_tqdm:bool = True,
)->List[List[str]]:
    """
    对每个prompt生成多条响应（rollout）
    """
    outputs = vllm.generate(prompts, sampling_params,use_tqdm = use_tqdm)
    rollouts = [[o.text for o in output.outputs] for output in outputs]
    return rollouts



# BUG: 使用vllm获得old_log_probs的函数逻辑存在严重错误, 不建议使用
@torch.no_grad()
def get_response_log_probs_vllm(
        repeated_prompt_ids:List[List[int]],# b*g ?
        flatten_output_log_probs:List[List[float]],# b*g ?
        response_mask: torch.Tensor, # ... l
    )->torch.Tensor:
        """
        return : bg l
        """
        max_seq_len = response_mask.shape[-1]
        # logprob<=0 填充的无效值应该为整数
        old_log_probs = [
                [100]* len(prompt_ids) + old_log_probs + 
                [100]* (max_seq_len -len(prompt_ids) - len(old_log_probs)+1) # padding last token
                for prompt_ids, old_log_probs in zip(repeated_prompt_ids, flatten_output_log_probs)
        ]
        # b*g l+1 多出一个token,补充一个padding
        old_log_probs = torch.tensor(old_log_probs, device=response_mask.device)
        return old_log_probs[:,1:]# b*g l

# NOTE: vllm不能按照预期产生old_log_probs, 现在的返回值无效
@torch.no_grad()
def generate_grpo_samples(
    vllm: LLM,
    prompts: list[str],# unrepeated prompts
    sampling_params: SamplingParams,
) -> Dict[str, List[List[List[float]]]]:
    """
    使用 vLLM 为每个 prompt 生成多条响应（rollout），并返回：
    - 文本响应
    - 每个响应 token 的 log_prob（旧策略）
    - 每个 prompt 的 token_ids（用于确定 prompt 长度）

    """
    outputs = vllm.generate(prompts, sampling_params)
    responses: list[list[str]] = [
        [o.text for o in output_per_prompt.outputs]
        for output_per_prompt in outputs
    ]
    flatten_responses = [
        response_per_rollout
        for responses_per_prompt in responses
            for response_per_rollout in responses_per_prompt
    ]

    output_logprobs: list[list[list[float]]] = [
        [
            [
                logprobs_dict[token_id].logprob
                for token_id, logprobs_dict in zip(
                    output_per_rollout.token_ids,
                    output_per_rollout.logprobs,
                )
            ]
            for output_per_rollout in output_per_prompt.outputs
        ]
        for output_per_prompt in outputs
    ]
    flatten_output_logprobs = [
        output_logprobs_per_rollout
        for output_logprobs_per_prompt in output_logprobs
            for output_logprobs_per_rollout in output_logprobs_per_prompt
    ]

    prompt_token_ids: list[list[int]] = [
        output_per_prompt.prompt_token_ids
        for output_per_prompt in outputs
    ]
    repeated_prompt_token_ids = [
        prompt_token_ids_per_sample
        for prompt_token_ids_per_sample in prompt_token_ids
            for _ in range(sampling_params.n)  # 每个 prompt 重复 n 次（n 是 rollout 数量）
    ]
    unflatten_metdata = {
        "responses": responses,
        "output_logprobs": output_logprobs,
        "prompt_token_ids": prompt_token_ids,
    }
    flatten_metadata = {
        "responses": flatten_responses,
        "output_logprobs": flatten_output_logprobs,
        "prompt_token_ids": repeated_prompt_token_ids,
    }
    return unflatten_metdata, flatten_metadata

@torch.no_grad()
def evaluate_vllm(
    vllm:LLM,
    prompts:List[str],
    ground_truths:List[str],
    sampling_params:SamplingParams,
    reward_fn:Callable,# defined in drgrpo_grader.py 
)->Dict[str,int]:
    responses = generate_responses(vllm, prompts, sampling_params)
    rewards = [reward_fn(response, ground_truth) for response,ground_truth in zip(responses,ground_truths)]
    overview = {
        "sample_size": len(rewards),
        "answer_correct": 0,
        "format_correct": 0,
        "total_correct": 0,
        "format_correct_but_answer_wrong": 0,
        "answer_correct_but_format_wrong": 0,
        "total_wrong": 0,
        "accuracy": 0.0,
        'wrong_rate': 0.0,
        'contradictory_samples': 0,
    }

    for reward_dict in rewards:
        # 统计格式正确
        if reward_dict['format_reward'] == 1.0:
            overview['format_correct'] += 1
            
        # 统计答案正确
        if reward_dict['answer_reward'] == 1.0:
            overview['answer_correct'] += 1
        
        # 统计完全正确
        if reward_dict['reward'] == 1.0:
            overview['total_correct'] += 1
        
        # 统计格式正确但答案错误
        if reward_dict['format_reward'] == 1.0 and reward_dict['answer_reward'] == 0.0:
            overview['format_correct_but_answer_wrong'] += 1
        
        # 统计答案正确但格式错误
        if reward_dict['answer_reward'] == 1.0 and reward_dict['format_reward'] == 0.0:
            overview['answer_correct_but_format_wrong'] += 1

        if reward_dict['format_reward'] == 0.0 and reward_dict['answer_reward'] == 0.0:
            overview['total_wrong'] += 1

        # 统计自相矛盾的样本数
        if reward_dict['reward']==1.0 and (reward_dict['format_reward'] == 0.0 or reward_dict['answer_reward'] == 0.0):
            overview['contradictory_samples'] += 1

    # 计算准确率
    overview['accuracy'] = overview['total_correct'] / overview['sample_size']
    overview['wrong_rate'] = overview['total_wrong'] / overview['sample_size']
    return overview

@torch.no_grad()
def log_generation(
    prompts: list[str],
    ground_truths: list[str],
    reward_fn: Callable,
    model: AutoModelForCausalLM,
    tokenizer:AutoTokenizer,
    vllm:LLM,
    sampling_params,
):
    """
    同时需要vllm和model
    对于model的显存占用有一定影响
    """
    device = next(model.parameters()).device
    responses = generate_responses(
        vllm,
        prompts,
        sampling_params,
    )

    reward_dicts = [reward_fn(resp, gt) for resp, gt in zip(responses, ground_truths)]

    total_rewards = torch.tensor([float(d["reward"]) for d in reward_dicts])
    fmt_rewards = torch.tensor([float(d["format_reward"]) for d in reward_dicts])
    ans_rewards = torch.tensor([float(d["answer_reward"]) for d in reward_dicts])
    correct = total_rewards == 1.0

    # 必须使用model来生成token_entropy,因为vllm不支持返回所有词表的logprobs
    inputs = tokenize_prompt_and_output(
        prompts,
        responses,
        tokenizer,
    )
    input_ids, labels, response_mask = inputs["input_ids"], inputs["labels"], inputs["response_mask"]

    # 获得logprobs:
    model.eval()
    out = get_response_log_probs(
        model,
        input_ids=input_ids.to(device),
        labels=labels.to(device),
        return_token_entropy=True,
    )
    ent = out["token_entropy"].cpu()

    res_len = response_mask.sum(dim=1).type_as(total_rewards)  # Number of response tokens per sample
    avg_ent = (ent * response_mask.type_as(ent)).sum(dim=1) / res_len # b l-> b

    rows = [
        {
            "prompt": p,
            "response": r,
            "true_answer": gt,
            "total_reward": float(tr.item()),
            "format_reward": float(fr.item()),
            "answer_reward": float(ar.item()),
            "is_correct": bool(c.item()),
            "response_length": int(rl.item()),
            "avg_token_entropy": float(ae.item()),
        }
        for p, r, gt, tr, fr, ar, c, rl, ae in zip(
            prompts,
            responses,
            ground_truths,
            total_rewards,
            fmt_rewards,
            ans_rewards,
            correct,
            res_len,
            avg_ent,
        )
    ]

    summary = {
        "avg_reward": float(total_rewards.float().mean().item()),
        "avg_token_entropy": float(avg_ent.mean().detach().cpu().item()),
        "avg_resp_len": float(res_len.float().mean().item()),
        "avg_len_correct": float(res_len[correct].float().mean().item()) if correct.any() else 0.0,
        "avg_len_wrong": float(res_len[~correct].float().mean().item()) if (~correct).any() else 0.0,
        "n_examples": len(prompts),
    }

    clear_gpu_memory()
    model.train()
    return {"summary": summary, "rows": rows}
