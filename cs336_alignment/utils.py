from typing import Iterable, List, Dict, Tuple, Union
import pandas as pd
import numpy as np
import random
import json
import torch
import gc

def seed_everything(seed: int = 42):
    """
    设置所有可能的随机种子以确保结果可重复
    
    Args:
        seed: 随机种子，默认42
    """
    # Python内置random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    
    return seed

def get_device(rank: int = 0) -> torch.device | None:
    """
    获取指定 GPU 设备。

    Args:
        rank: GPU 设备编号，默认 0。

    Returns:
        torch.device | None: 返回 CUDA 设备对象，若 CUDA 不可用或设备不存在则返回 None。
    """
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return None
    
    device_count = torch.cuda.device_count()
    
    if rank >= device_count:
        print(f"Device {rank} not available. Only {device_count} devices found.")
        return None
    
    return torch.device(f'cuda:{rank}')

# eval部分的代码
def load_jsonl(json_path: str) -> Iterable:
    """
    逐行读取 jsonl 文件。

    Args:
        json_path: jsonl 文件路径。

    Yields:
        dict: 每行解析后的 JSON 对象。
    """
    with open(json_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            yield json.loads(line)

def load_template(template_path: str) -> str:
    """
    加载 prompt 模板文件。

    Args:
        template_path: 模板文件路径。

    Returns:
        str: 模板内容。
    """
    with open(template_path,'r') as f:
        prompt_template = f.read()
    return prompt_template

def apply_r1_template(prompt: str, json_obj: dict) -> str:
    """
    将 JSON 对象应用到 R1 模板生成 prompt。

    Args:
        prompt: 模板字符串，包含 {question} 占位符。
        json_obj: 包含 question 字段的字典。

    Returns:
        str: 格式化后的 prompt。
    """
    return prompt.format(question=json_obj['question'])

def apply_r1_ground_truth_template(prompt: str, json_obj: dict) -> str:
    """
    将 JSON 对象应用到 R1 真实答案模板。

    Args:
        prompt: 模板字符串，包含 {cot} 和 {answer} 占位符。
        json_obj: 包含 cot 和 answer 字段的字典。

    Returns:
        str: 格式化后的真实答案模板。
    """
    return prompt.format(cot=json_obj['cot'],answer=json_obj['answer'])

def get_r1_prompts(json_path: str, prompt_template: str) -> list:
    """
    从 jsonl 文件加载 prompts 并应用 R1 模板。

    Args:
        json_path: jsonl 文件路径。
        prompt_template: prompt 模板字符串。

    Returns:
        list: 格式化后的 prompt 列表。
    """
    json_iter = load_jsonl(json_path)
    return [apply_r1_template(prompt_template,json_obj) for json_obj in json_iter]

def get_r1_ground_truths(json_path: str) -> list:
    """
    从 jsonl 文件加载 R1 格式的真实答案（仅 answer 部分）。

    Args:
        json_path: jsonl 文件路径。

    Returns:
        list: 答案字符串列表。
    """
    json_iter = load_jsonl(json_path)
    return [json_obj['answer'] for json_obj in json_iter]

def get_r1_ground_truths_with_template(
    json_path: str,
    ground_truth_template: str = "{cot} </think> <answer> {answer} </answer>",
) -> list:
    """
    从 jsonl 文件加载 R1 格式的真实答案（包含 cot 和 answer 模板）。

    Args:
        json_path: jsonl 文件路径。
        ground_truth_template: 真实答案模板字符串，包含 {cot} 和 {answer} 占位符。

    Returns:
        list: 格式化后的真实答案列表。
    """
    json_iter = load_jsonl(json_path)
    return [apply_r1_ground_truth_template(ground_truth_template,json_obj) for json_obj in json_iter]

def clear_gpu_memory():
    """
    清理 GPU 内存，包括 Python GC 和 CUDA 缓存。
    """
    gc.collect()
    # 4. 清理 CUDA 缓存（仅在 GPU 可用时调用）
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # 回收跨进程共享缓存（如果没有多进程，也不会有副作用）
        torch.cuda.ipc_collect()
