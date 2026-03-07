from sft_config import SFTConfig
from dataclasses import dataclass, asdict
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

    # EI 专用采样配置（与 SFTConfig 中的 sampling 参数区分开）
    ei_temperature: float = 1.0
    ei_top_p: float = 1.0
    ei_max_tokens: int = 1024
    ei_stop: list = None
    ei_include_stop_str_in_output: bool = True
    ei_min_tokens: int = 32

    # 说明：sample_size(SFT评估阶段采样) vs sft_sample_size(SFT训练阶段样本数)
    # sampling_params参数不能继承SFTConfig, 采样需求不同
    

    # 其余字段全部从 SFTConfig 继承：
    def __post_init__(self) -> None:
        """
        重写版本：在 SFTConfig.__post_init__ 的基础上，使用 EIConfig 中的
        采样参数字段重新创建 sampling_params。
        """
        # 先执行父类逻辑：包括 seed、reward_fn 等初始化
        super().__post_init__()

        # 基于 EIConfig 的字段重新构造 SamplingParams，用于 EI 采样
        self.ei_sampling_params = SamplingParams(
            temperature=self.ei_temperature,
            top_p=self.ei_top_p,
            stop=self.ei_stop,
            seed=self.seed,
            include_stop_str_in_output=self.ei_include_stop_str_in_output,
            max_tokens=self.ei_max_tokens,
            min_tokens=self.ei_min_tokens,
            n=self.rollout_size,
        )
        # reward_fn 仍由 SFTConfig.__post_init__ 按 reward_fn_name 构造，
        # 比如默认为 r1_zero_reward_fn，不做修改。

    def pretty_print(self) -> None:
        """
        格式化打印 EIConfig 的所有字段、以及 SFT/EI 的 SamplingParams 细节。
        """
        cfg = asdict(self)

        print("=" * 80)
        print("EIConfig 全部配置")
        print("=" * 80)

        # 1. 基础/训练相关配置（排除温度等采样字段）
        print("\n[基础与训练配置]")
        for k, v in cfg.items():
            if k in {
                "temperature", "top_p", "max_tokens", "stop",
                "include_stop_str_in_output",
                "ei_temperature", "ei_top_p", "ei_max_tokens",
                "ei_stop", "ei_include_stop_str_in_output",
                "ei_min_tokens",
            }:
                continue
            print(f"  {k:32s}: {v}")

        # 2. SFT 采样配置（从 SFTConfig 继承来的温度等）
        print("\n[SFT 采样配置字段 (配置层面)]")
        print(f"  {'temperature':32s}: {self.temperature}")
        print(f"  {'top_p':32s}: {self.top_p}")
        print(f"  {'max_tokens':32s}: {self.max_tokens}")
        print(f"  {'stop':32s}: {self.stop}")
        print(f"  {'include_stop_str_in_output':32s}: {self.include_stop_str_in_output}")

        # 3. EI 采样配置（ei_* 字段）
        print("\n[EI 采样配置字段 (配置层面)]")
        print(f"  {'ei_temperature':32s}: {self.ei_temperature}")
        print(f"  {'ei_top_p':32s}: {self.ei_top_p}")
        print(f"  {'ei_max_tokens':32s}: {self.ei_max_tokens}")
        print(f"  {'ei_min_tokens':32s}: {self.ei_min_tokens}")
        print(f"  {'ei_stop':32s}: {self.ei_stop}")
        print(f"  {'ei_include_stop_str_in_output':32s}: {self.ei_include_stop_str_in_output}")
        print(f"  {'rollout_size (n)':32s}: {self.rollout_size}")

        # 4. 实际 SamplingParams 对象情况
        print("\n[SFT SamplingParams 实例 (self.sampling_params)]")
        sp = getattr(self, "sampling_params", None)
        if sp is None:
            print("  <None>")
        else:
            print(f"  {'temperature':32s}: {sp.temperature}")
            print(f"  {'top_p':32s}: {sp.top_p}")
            print(f"  {'max_tokens':32s}: {sp.max_tokens}")
            print(f"  {'stop':32s}: {sp.stop}")
            print(f"  {'seed':32s}: {sp.seed}")
            print(f"  {'include_stop_str_in_output':32s}: {sp.include_stop_str_in_output}")
            print(f"  {'n':32s}: {getattr(sp, 'n', None)}")

        print("\n[EI SamplingParams 实例 (self.ei_sampling_params)]")
        esp = getattr(self, "ei_sampling_params", None)
        if esp is None:
            print("  <None>")
        else:
            print(f"  {'temperature':32s}: {esp.temperature}")
            print(f"  {'top_p':32s}: {esp.top_p}")
            print(f"  {'max_tokens':32s}: {esp.max_tokens}")
            print(f"  {'stop':32s}: {esp.stop}")
            print(f"  {'seed':32s}: {esp.seed}")
            print(f"  {'include_stop_str_in_output':32s}: {esp.include_stop_str_in_output}")
            print(f"  {'n':32s}: {getattr(esp, 'n', None)}")

        print("=" * 80)

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
            # EI 专用采样字段
            'ei_temperature': self.ei_temperature,
            'ei_top_p': self.ei_top_p,
            'ei_max_tokens': self.ei_max_tokens,
            'ei_stop': self.ei_stop,
            'ei_include_stop_str_in_output': self.ei_include_stop_str_in_output,
            'ei_min_tokens': self.ei_min_tokens,
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

        # EI 专用采样字段的默认处理
        if 'ei_temperature' not in config_dict:
            config_dict['ei_temperature'] = cls.ei_temperature
        if 'ei_top_p' not in config_dict:
            config_dict['ei_top_p'] = cls.ei_top_p
        if 'ei_max_tokens' not in config_dict:
            config_dict['ei_max_tokens'] = cls.ei_max_tokens
        if 'ei_stop' not in config_dict:
            # 默认使用 SFT 的 stop，如果没有则用类默认 None，让 __post_init__ 再处理
            config_dict['ei_stop'] = config_dict.get('stop', None)
        if 'ei_include_stop_str_in_output' not in config_dict:
            config_dict['ei_include_stop_str_in_output'] = cls.ei_include_stop_str_in_output
        if 'ei_min_tokens' not in config_dict:
            # 若旧 JSON 里有 min_tokens，则复用，否则用类默认
            config_dict['ei_min_tokens'] = config_dict.get('ei_min_tokens', cls.ei_min_tokens)

        # 处理父类中的特殊字段
        if 'betas' in config_dict and isinstance(config_dict['betas'], list):
            config_dict['betas'] = tuple(config_dict['betas'])

        config = cls(**config_dict)

        print(f"✅ EIConfig loaded from {filepath}")
        return config

if __name__ == "__main__":
    # 1. 创建默认配置
    config = EIConfig.from_json('cs336_alignment/configs/ei_math.json')
    # print(config)
    # print(f"sampling_params type: {type(config.sampling_params)}")
    # print(f"sampling_params seed: {config.sampling_params.seed}")
    # print(f"sampling_params temperature: {config.sampling_params.temperature}")

    # print(f"reward_fn type: {type(config.reward_fn)}")
    # print(f"reward_fn name: {config.reward_fn.__name__}")
    # print(f"reward_fn callable? {callable(config.reward_fn)}")

    config.pretty_print()