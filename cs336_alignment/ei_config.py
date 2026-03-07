from sft_config import SFTConfig
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import json
import os
from sft_config import SFTConfig
from drgrpo_grader import r1_zero_reward_fn
from vllm import SamplingParams


@dataclass
class EIConfig(SFTConfig):
    """
    EI（Expert Iteration）配置类。

    在 SFTConfig 的基础上增加 EI 相关的配置：
    - ei_iterations: 进行多少轮 Expert Iteration
    - rollout_size:  每个 prompt 进行多少次 rollout 采样
    - 其余字段（batch_size / lr / dataset 路径 / sampling_params / reward_fn 等）
      完全复用 SFTConfig 的定义和 __post_init__。
    """

    # EI 相关配置
    ei_iterations: int = 3       # Expert Iteration 的外层循环次数
    rollout_size: int = 4        # 每个 prompt 的 rollout 次数（vllm 采样次数）
    sft_sample_size: int = 128 
    min_tokens: int = 32

    # 说明：sample_size(SFT评估阶段采样) vs sft_sample_size(SFT训练阶段样本数)

    # 其余字段全部从 SFTConfig 继承：
    def __post_init__(self) -> None:
        """
        重写版本：在 SFTConfig.__post_init__ 的基础上，使用 EIConfig 中的
        采样参数字段重新创建 sampling_params。
        区别：
        - 父类 SFTConfig.__post_init__ 中通常构造的是用于 SFT 评估 / 采样的
          SamplingParams（可能只有 temperature / top_p / max_tokens 等）。
        - 本类在调用父类 __post_init__ 后，会根据 EIConfig 中新增的：
            - sampling_min_tokens
            - sampling_n
          等字段，重新构造一个更适合 EI 采样的 sampling_params。
        - reward_fn 的构造逻辑保持完全一致：仍由 reward_fn_name 决定，
          不在此处修改。
        """
        # 先执行父类逻辑：包括 seed、reward_fn 等初始化
        super().__post_init__()

        # 基于 EIConfig 的字段重新构造 SamplingParams，用于 EI 采样
        # 注意：这里复用父类中的 temperature / top_p / max_tokens / stop 等字段
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            stop=self.stop,
            seed=self.seed,
            include_stop_str_in_output=self.include_stop_str_in_output,
            max_tokens=self.max_tokens,
            min_tokens=self.min_tokens,
            n=self.rollout_size,

        )
        # reward_fn 仍由 SFTConfig.__post_init__ 按 reward_fn_name 构造，
        # 比如默认为 r1_zero_reward_fn，不做修改。

    def to_json(self, filepath: str) -> None:
        """
        重写版本：在 SFTConfig.to_json 的基础上，额外保存 EI 相关字段。

        区别：
        - 父类只保存 SFT 训练相关字段；
        - 本类在此基础上再保存：
            - ei_iterations
            - rollout_size
        """
        # 先调用父类的实现生成基础配置字典
        # 注意：父类实现是直接写文件，这里我们重写为“构造 dict 再写文件”，
        # 以便附加 EI 字段。为保证简洁，这里不直接调用父类 to_json，
        # 而是基本复制其逻辑再扩展。
        config_dict: Dict[str, Any] = {
            'seed': self.seed,
            'project_name': self.project_name,
            'name': self.name,
            'batch_size': self.batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'max_iters': self.max_iters,
            'start_iters': self.start_iters,
            'weight_decay': self.weight_decay,
            'betas': list(self.betas),
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
            # EI 新增字段
            'ei_iterations': self.ei_iterations,
            'rollout_size': self.rollout_size,
            'sft_sample_size': self.sft_sample_size,
            'min_tokens':self.min_tokens,
        }
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)

        print(f"✅ EIConfig saved to {filepath}")

    @classmethod
    def from_json(cls, filepath: str) -> "EIConfig":
        """
        重写版本：在 SFTConfig.from_json 的基础上，额外加载 EI 相关字段。

        区别：
        - 父类只支持 SFT 字段；
        - 本类在读取 JSON 时，如果包含:
            - ei_iterations
            - rollout_size
            - sft_sample_size
          则会一并加载到 EIConfig 中。
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        # 兼容老配置文件：没有 EI 字段时使用默认值
        if 'ei_iterations' not in config_dict:
            config_dict['ei_iterations'] = cls.ei_iterations
        if 'rollout_size' not in config_dict:
            config_dict['rollout_size'] = cls.rollout_size
        if 'sft_sample_size' not in config_dict:
            config_dict['sft_sample_size'] = cls.sft_sample_size
        if 'min_tokens' not in config_dict:
            config_dict['min_tokens'] = cls.min_tokens

        # 处理父类中的特殊字段
        if 'betas' in config_dict and isinstance(config_dict['betas'], list):
            config_dict['betas'] = tuple(config_dict['betas'])

        config = cls(**config_dict)

        print(f"✅ EIConfig loaded from {filepath}")
        return config
if __name__ == "__main__":
    # 1. 创建默认配置
    config = EIConfig.from_json('cs336_alignment/configs/ei_math.json')
    print(config)
    print(f"sampling_params type: {type(config.sampling_params)}")
    print(f"sampling_params seed: {config.sampling_params.seed}")
    print(f"sampling_params temperature: {config.sampling_params.temperature}")

    print(f"reward_fn type: {type(config.reward_fn)}")
    print(f"reward_fn name: {config.reward_fn.__name__}")
    print(f"reward_fn callable? {callable(config.reward_fn)}")