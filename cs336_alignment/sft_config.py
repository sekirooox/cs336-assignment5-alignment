from dataclasses import dataclass,field
from drgrpo_grader import r1_zero_reward_fn
from vllm import SamplingParams
from typing import Callable
from dataclasses import dataclass, field, asdict
from drgrpo_grader import r1_zero_reward_fn
from vllm import SamplingParams
from typing import Callable, Tuple, Optional, Dict, Any
import json
import os

@dataclass
class SFTConfig:
    # 随机数设置
    seed: int = 42

    # wandb设置
    project_name:str = 'cs336-assignment5'
    name:str = 'SFT-training'

    # 训练基本配置
    batch_size: int = 4
    gradient_accumulation_steps: int = 3
    max_iters: int = 500
    start_iters: int = 0

    # 优化器配置
    weight_decay: float = 1e-5
    betas: Tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-6
    
    # 学习率配置
    max_lr: float = 5e-6
    min_lr: float = 1e-6
    warmup_iters: int = 50
    cosine_schedule_iters: int = 450
    
    # 优化器配置
    max_grad_norm: float = 1.0
    normalize_constant: float = 1.0  # loss归一化常数
    
    # 数据集配置
    train_dataset_path: str = r"/root/autodl-tmp/assignment5-alignment/preprocessed/gsm8k/train.jsonl"
    test_dataset_path: str = r"/root/autodl-tmp/assignment5-alignment/preprocessed/gsm8k/test.jsonl"
    prompt_template_path: str = r"/root/autodl-tmp/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
    
    # 评估配置
    eval_interval: int = 50  # 每多少步评估一次
    sample_size: int = 4  # 采样评估的样本数量
    
    # 保存配置
    save_interval: int = 250  # 每多少步保存一次
    save_dir: str = "checkpoints/"
    
    # vLLM推理配置 - 这些参数用于创建SamplingParams SFT和EI不能共用一个SamplingParams
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 1024
    stop: list = field(default_factory=lambda: ["</answer>"])
    include_stop_str_in_output: bool = True
    
    # 奖励函数配置 - 存储函数名称
    reward_fn_name: str = "r1_zero_reward_fn"

    
    # 这些字段会在__post_init__中创建，不直接序列化
    def __post_init__(self):
        """初始化后创建实际的对象实例"""
        # 创建SamplingParams实例
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop=self.stop,
            seed=self.seed,
            include_stop_str_in_output=self.include_stop_str_in_output
        )
        
        # 根据函数名称设置reward_fn
        self.reward_fn = self._get_reward_fn(self.reward_fn_name)
    
    def _get_reward_fn(self, name: str) -> Callable:
        """根据名称获取奖励函数"""
        reward_functions = {
            "r1_zero_reward_fn": r1_zero_reward_fn,
            # 可以在这里添加更多奖励函数
        }
        
        if name not in reward_functions:
            raise ValueError(f"Unknown reward function: {name}. Available: {list(reward_functions.keys())}")
        return reward_functions[name]
    
    def to_json(self, filepath: str) -> None:
        """
        将配置保存为JSON文件
        
        Args:
            filepath: JSON文件保存路径
        """
        # 获取所有字段（排除特殊字段）
        config_dict = {
            'batch_size': self.batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'max_iters': self.max_iters,
            'start_iters': self.start_iters,
            'weight_decay': self.weight_decay,
            'betas': list(self.betas),  # Tuple转list以便JSON序列化
            'eps': self.eps,
            'max_lr': self.max_lr,
            'min_lr': self.min_lr,
            'warmup_iters': self.warmup_iters,
            'cosine_schedule_iters': self.cosine_schedule_iters,
            'max_grad_norm': self.max_grad_norm,
            'normalize_constant': self.normalize_constant,
            'train_dataset_path': self.train_dataset_path,
            'test_dataset_path': self.test_dataset_path,
            'prompt_template_path': self.prompt_template_path,
            'eval_interval': self.eval_interval,
            'sample_size': self.sample_size,
            'save_interval': self.save_interval,
            'save_dir': self.save_dir,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'max_tokens': self.max_tokens,
            'stop': self.stop,
            'include_stop_str_in_output': self.include_stop_str_in_output,
            'reward_fn_name': self.reward_fn_name,
        }
        
        # 创建目录
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        # 保存为JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)
        
        print(f"✅ Configuration saved to {filepath}")
    
    @classmethod
    def from_json(cls, filepath: str) -> 'SFTConfig':
        """
        从JSON文件加载配置
        
        Args:
            filepath: JSON文件路径
            
        Returns:
            SFTConfig实例
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # 处理特殊字段
        if 'betas' in config_dict and isinstance(config_dict['betas'], list):
            config_dict['betas'] = tuple(config_dict['betas'])
        
        # 创建实例
        config = cls(**config_dict)
        
        print(f"✅ Configuration loaded from {filepath}")
        return config
    

# 使用示例
if __name__ == "__main__":
    # 1. 创建默认配置
    config = SFTConfig.from_json('cs336_alignment/configs/sft.json')
    print(config)
    print(f"sampling_params type: {type(config.sampling_params)}")
    print(f"sampling_params seed: {config.sampling_params.seed}")
    print(f"sampling_params temperature: {config.sampling_params.temperature}")

    print(f"reward_fn type: {type(config.reward_fn)}")
    print(f"reward_fn name: {config.reward_fn.__name__}")
    print(f"reward_fn callable? {callable(config.reward_fn)}")

