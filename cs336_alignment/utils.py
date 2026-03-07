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

def get_device(rank:int=0):
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return None
    
    device_count = torch.cuda.device_count()
    
    if rank >= device_count:
        print(f"Device {rank} not available. Only {device_count} devices found.")
        return None
    
    return torch.device(f'cuda:{rank}')

# eval部分的代码
def load_jsonl(
    json_path:str
)->Iterable:
    with open(json_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            yield json.loads(line)

def load_template(
    template_path:str
)->str:
    with open(template_path,'r') as f:
        prompt_template = f.read()
    return prompt_template

def apply_r1_template(
    prompt:str,
    json_obj:dict
)->str:#
    return prompt.format(question=json_obj['question'])

def apply_r1_ground_truth_template(
    prompt:str,
    json_obj:dict
)->str:
    return prompt.format(cot=json_obj['cot'],answer=json_obj['answer'])

def get_r1_prompts(
    json_path:str,
    prompt_template:str,
)->list:
    json_iter = load_jsonl(json_path)
    return [apply_r1_template(prompt_template,json_obj) for json_obj in json_iter]

def get_r1_ground_truths(
    json_path:str,
    # ground_truth_template:str='{cot}</think><answer>{answer}</answer>', # ground_truth不包含模板内容
)->list:
    json_iter = load_jsonl(json_path)
    return [json_obj['answer'] for json_obj in json_iter]

def get_r1_ground_truths_with_template(
    json_path:str,
    # 建议<answer>不加入空格
    ground_truth_template:str='{cot} </think> <answer> {answer} </answer>', # ground_truth不包含模板内容
)->list:
    json_iter = load_jsonl(json_path)
    return [apply_r1_ground_truth_template(ground_truth_template,json_obj) for json_obj in json_iter]

def clear_gpu_memory():
    gc.collect()
    # 4. 清理 CUDA 缓存（仅在 GPU 可用时调用）
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # 回收跨进程共享缓存（如果没有多进程，也不会有副作用）
        torch.cuda.ipc_collect()
