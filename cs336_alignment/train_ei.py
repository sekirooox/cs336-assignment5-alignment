from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from vllm import LLM,SamplingParams
from vllm_utils import *
from utils import *
from ei_config import EIConfig
from sft_config import SFTConfig
import os
from sft import *
from sft_trainer import *
from drgrpo_grader import r1_zero_reward_fn
os.environ["WANDB_API_KEY"] = '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'

import argparse
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path",
        type=str,
        help="json path of the EIConfig",
    )
    args = parser.parse_args()
    return args
# 使用args进行解析
args = parse()

# 配置config
config = EIConfig.from_json(args.json_path)

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
device1 = get_device(1)
device2 = get_device(2)

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

# vllm离线推理模型 GPU利用率不能太高
vllm = init_vllm(model_name,device=device2,seed=config.seed,gpu_memory_utilization=0.6)

trainer = EITrainer(
    model,
    tokenizer,
    optimizer,
    config,
    vllm,
)

trainer.train()