from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch
from vllm import LLM,SamplingParams
from transformers import PreTrainedModel,AutoTokenizer,AutoModelForCausalLM
from typing import Union,List,Callable,Dict,Any,Tuple,Literal,Optional
import torch
from sft import tokenize_prompt_and_output,get_response_log_probs
from utils import *

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85) -> LLM:
    """
    启动推理过程，此处使用 vLLM 将模型部署在与策略模型不同的 GPU 上。

    Args:
        model_id: 模型 ID 或本地路径。
        device: GPU 设备字符串（如 "cuda:0"）。
        seed: 随机种子。
        gpu_memory_utilization: GPU 内存利用率，默认为 0.85。

    Returns:
        LLM: 初始化后的 vLLM 模型实例。
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

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM) -> None:
    """
    将训练好的策略模型权重加载到 vLLM 实例中。

    从 https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670 复制

    Args:
        policy: HuggingFace 预训练模型（训练中的策略模型）。
        llm: vLLM 模型实例。
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
    使用 vLLM 生成响应。

    Args:
        vllm: vLLM 模型实例。
        prompts: 单个 prompt 字符串或 prompt 列表。
        sampling_params: 采样参数。

    Returns:
        List[str]: 生成的响应字符串列表。
    """    
    # 生成响应
    outputs = vllm.generate(prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    return responses

@torch.no_grad()
def generate_rollouts(
    vllm: LLM,
    prompts: List[str],
    sampling_params: SamplingParams,
    use_tqdm: bool = True,
) -> List[List[str]]:
    """
    对每个 prompt 生成多条响应（rollout）。

    Args:
        vllm: vLLM 模型实例。
        prompts: prompt 列表。
        sampling_params: 采样参数。
        use_tqdm: 是否显示进度条，默认为 True。

    Returns:
        List[List[str]]: 二维列表，外层每个元素对应一个 prompt，内层每个元素对应一条响应。
    """
    outputs = vllm.generate(prompts, sampling_params,use_tqdm = use_tqdm)
    rollouts = [[o.text for o in output.outputs] for output in outputs]
    return rollouts

@torch.no_grad()
def evaluate_vllm(
    vllm: LLM,
    prompts: List[str],
    ground_truths: List[str],
    sampling_params: SamplingParams,
    reward_fn: Callable,
) -> Dict[str, int]:
    """
    使用 vLLM 评估模型在给定 prompts 上的表现。

    Args:
        vllm: vLLM 模型实例。
        prompts: prompt 列表。
        ground_truths: 真实答案列表。
        sampling_params: 采样参数。
        reward_fn: 奖励函数，用于评估响应。

    Returns:
        Dict[str, int]: 包含评估指标的字典，如 sample_size, answer_correct, format_correct 等。
    """
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
    tokenizer: AutoTokenizer,
    vllm: LLM,
    sampling_params: SamplingParams,
) -> Dict[str, Any]:
    """
    同时需要 vllm 和 model，对于 model 的显存占用有一定影响。

    Args:
        prompts: prompt 列表。
        ground_truths: 真实答案列表。
        reward_fn: 奖励函数。
        model: 训练中的策略模型。
        tokenizer: 分词器。
        vllm: vLLM 模型实例。
        sampling_params: 采样参数。

    Returns:
        Dict[str, Any]: 包含 summary 和 rows 的字典。
    """
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
