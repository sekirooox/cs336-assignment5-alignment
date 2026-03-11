from dataclasses import dataclass,field
from drgrpo_grader import r1_zero_reward_fn
from vllm import SamplingParams
from dataclasses import dataclass, field, asdict
from typing import Callable, Tuple, Optional, Dict, Any, Literal
import json
import os

@dataclass
class SFTConfig:
    """
    SFTConfig
    """
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


@dataclass
class GRPOConfig(SFTConfig):
    """
    GRPO（Group Relative Policy Optimization）训练配置。
    """
    # -------- GRPO / PPO 相关超参 --------
    group_size: int = 4
    clip_range: float = 0.2
    advantage_eps: float = 1e-8
    normalize_by_std: bool = True
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = 'grpo_clip'
    n_train_steps_per_rollout_batch:int = 1 # On-policy default

    # GRPO 专用采样参数（与 SFTConfig 中的 sampling_params 区分）
    grpo_temperature: float = 1.0
    grpo_top_p: float = 1.0
    grpo_max_tokens: int = 1024
    grpo_min_tokens: int = 1
    grpo_stop: Optional[list] = None
    grpo_include_stop_str_in_output: bool = True

    def __post_init__(self) -> None:
        """
        先调用 SFTConfig.__post_init__ 保持 SFT 行为，
        然后基于 GRPO 字段构造单独的 grpo_sampling_params。
        """
        super().__post_init__()

        # 如果没有显式设置 grpo_stop，默认与 SFT 的 stop 一致
        stop = self.grpo_stop if self.grpo_stop is not None else self.stop

        # 为 GRPO 采样单独构造 SamplingParams
        self.grpo_sampling_params = SamplingParams(
            temperature=self.grpo_temperature,
            top_p=self.grpo_top_p,
            max_tokens=self.grpo_max_tokens,
            min_tokens=self.grpo_min_tokens,
            stop=stop,
            seed=self.seed,
            include_stop_str_in_output=self.grpo_include_stop_str_in_output,
            n=self.group_size,
        )

    def to_json(self, filepath: str) -> None:
        """
        将 GRPOConfig 保存为 JSON 文件，包含：
        - SFTConfig 中的通用训练字段
        - GRPOConfig 新增字段
        """
        config_dict: Dict[str, Any] = {
            # ---------- 从 SFTConfig 继承的字段 ----------
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

            # ---------- GRPOConfig 新增字段 ----------
            'group_size': self.group_size,
            'clip_range': self.clip_range,
            'advantage_eps': self.advantage_eps,
            'normalize_by_std': self.normalize_by_std,
            'loss_type': self.loss_type,
            'n_train_steps_per_rollout_batch': self.n_train_steps_per_rollout_batch,

            # GRPO 采样字段
            'grpo_temperature': self.grpo_temperature,
            'grpo_top_p': self.grpo_top_p,
            'grpo_max_tokens': self.grpo_max_tokens,
            'grpo_min_tokens': self.grpo_min_tokens,
            'grpo_stop': self.grpo_stop,
            'grpo_include_stop_str_in_output': self.grpo_include_stop_str_in_output,
        }

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)

        print(f"✅ GRPOConfig saved to {filepath}")

    @classmethod
    def from_json(cls, filepath: str) -> "GRPOConfig":
        """
        从 JSON 文件中加载 GRPOConfig，兼容缺失 GRPO 字段的老配置：
        若某些字段不存在，则使用类默认值。
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        # ---------- 兼容老配置：为缺失字段填默认值 ----------
        default = cls()  # 用于获取类默认值

        # SFTConfig 中需要特殊处理的字段
        if 'betas' in config_dict and isinstance(config_dict['betas'], list):
            config_dict['betas'] = tuple(config_dict['betas'])

        # GRPO 相关字段（仅限当前类中真实存在的字段）
        for field_name in [
            'group_size',
            'clip_range',
            'advantage_eps',
            'normalize_by_std',
            'loss_type',
            'n_train_steps_per_rollout_batch',
            'grpo_temperature',
            'grpo_top_p',
            'grpo_max_tokens',
            'grpo_min_tokens',
            'grpo_stop',
            'grpo_include_stop_str_in_output',
        ]:
            if field_name not in config_dict:
                config_dict[field_name] = getattr(default, field_name)

        config = cls(**config_dict)
        print(f"✅ GRPOConfig loaded from {filepath}")
        return config

    def pretty_print(self) -> None:
        """
        格式化打印 GRPOConfig 的所有关键字段。
        """
        cfg = asdict(self)

        print("=" * 80)
        print("GRPOConfig 全部配置")
        print("=" * 80)

        # 1. 基础 / SFT 训练相关
        print("\n[基础与 SFT 训练配置]")
        for k in [
            'seed', 'project_name', 'name',
            'batch_size', 'gradient_accumulation_steps',
            'max_iters', 'start_iters',
            'weight_decay', 'betas', 'eps',
            'max_lr', 'min_lr', 'warmup_iters', 'cosine_schedule_iters',
            'max_grad_norm', 'normalize_constant',
            'train_dataset_path', 'test_dataset_path', 'prompt_template_path',
            'eval_interval', 'sample_size',
            'save_interval', 'save_dir',
            'temperature', 'top_p', 'max_tokens', 'stop',
            'include_stop_str_in_output', 'reward_fn_name',
        ]:
            print(f"  {k:32s}: {cfg.get(k)}")

        # 2. GRPO 训练超参
        print("\n[GRPO / PPO 超参数]")
        for k in [
            'group_size',
            'clip_range',
            'advantage_eps',
            'normalize_by_std',
            'loss_type',
            'n_train_steps_per_rollout_batch',
        ]:
            print(f"  {k:32s}: {cfg.get(k)}")

        # 3. GRPO 采样配置字段
        print("\n[GRPO 采样配置字段 (配置层面)]")
        for k in [
            'grpo_temperature', 'grpo_top_p',
            'grpo_max_tokens', 'grpo_min_tokens',
            'grpo_stop', 'grpo_include_stop_str_in_output',
        ]:
            print(f"  {k:32s}: {cfg.get(k)}")

        # 4. 实际 SamplingParams 实例
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

        print("\n[GRPO SamplingParams 实例 (self.grpo_sampling_params)]")
        gsp = getattr(self, "grpo_sampling_params", None)
        if gsp is None:
            print("  <None>")
        else:
            print(f"  {'temperature':32s}: {gsp.temperature}")
            print(f"  {'top_p':32s}: {gsp.top_p}")
            print(f"  {'max_tokens':32s}: {gsp.max_tokens}")
            print(f"  {'min_tokens':32s}: {getattr(gsp, 'min_tokens', None)}")
            print(f"  {'stop':32s}: {gsp.stop}")
            print(f"  {'seed':32s}: {gsp.seed}")
            print(f"  {'include_stop_str_in_output':32s}: {gsp.include_stop_str_in_output}")
            print(f"  {'n':32s}: {getattr(gsp, 'n', None)}")

        print("=" * 80)

# 使用示例
if __name__ == "__main__":
    # config = SFTConfig.from_json('cs336_alignment/configs/sft.json')
    # print(config)
    # print(f"sampling_params type: {type(config.sampling_params)}")
    # print(f"sampling_params seed: {config.sampling_params.seed}")
    # print(f"sampling_params temperature: {config.sampling_params.temperature}")

    # print(f"reward_fn type: {type(config.reward_fn)}")
    # print(f"reward_fn name: {config.reward_fn.__name__}")
    # print(f"reward_fn callable? {callable(config.reward_fn)}")

    config = GRPOConfig.from_json('cs336_alignment/configs/grpo.json')
    config.pretty_print()