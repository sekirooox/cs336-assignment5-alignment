from torch.utils.data import Dataset,DataLoader
from typing import Tuple, List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import itertools
from utils import load_template,get_r1_prompts,get_r1_ground_truths_with_template,get_device,seed_everything
from dataclasses import dataclass
import torch
from vllm import LLM,SamplingParams
from vllm_utils import init_vllm,load_policy_into_vllm_instance,log_generation,evaluate_vllm
import random
from sft import *
import wandb
import os
from drgrpo_grader import r1_zero_reward_fn
from sft_trainer import SFTTrainer
from sft_config import SFTConfig
os.environ["WANDB_API_KEY"] = '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'
import argparse
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path",
        type=str,
        help="json path of the SFTConfig",
    )
    args = parser.parse_args()
    return args

# 使用args进行解析
args = parse()

# 配置config
config = SFTConfig.from_json(args.json_path)

# seed
seed = seed_everything(config.seed)

# wandb
import wandb
wandb.login()
wandb.init(
    project = config.project_name,
    name = config.name,
    config = config,
)


# 设备
device1 = get_device(0)
device2 = get_device(1)


# 训练模型
model_name ='model/Qwen2.5-Math-1.5B'
model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device1,# device 1
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.max_lr, weight_decay=config.weight_decay, betas = config.betas, eps = config.eps)

# vllm离线推理模型
vllm = init_vllm(model_name,device=device2,seed=config.seed)

# trainer
trainer = SFTTrainer(
    model,
    tokenizer,
    optimizer,
    config,
    vllm)



trainer.train()
