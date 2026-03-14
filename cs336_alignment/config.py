from dataclasses import dataclass, field, fields, asdict
from drgrpo_grader import r1_zero_reward_fn
from vllm import SamplingParams
from typing import Callable, Tuple, Optional, Dict, Any, Type
import json
import os


# ============ 工具函数 ============

def _serialize_config(obj: Any) -> Dict[str, Any]:
    """将配置对象序列化为字典，处理特殊类型（如 Tuple）。"""
    config_dict = {}
    for f in fields(obj):
        value = getattr(obj, f.name)
        # Tuple 转换为 list 以便 JSON 序列化
        if isinstance(value, tuple):
            value = list(value)
        config_dict[f.name] = value
    return config_dict

def _load_json_common(cls: Type, filepath: str, extra_defaults: Optional[Dict[str, Any]] = None) -> Any:
    """
    通用的 JSON 加载方法。

    Args:
        cls: 配置类
        filepath: JSON 文件路径
        extra_defaults: 额外的默认字段字典（子类特有字段）
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)

    # 处理特殊字段：betas (list -> tuple)
    if 'betas' in config_dict and isinstance(config_dict['betas'], list):
        config_dict['betas'] = tuple(config_dict['betas'])

    # 使用类的默认值填充缺失字段
    default_obj = cls()
    for f in fields(default_obj):
        if f.name not in config_dict:
            config_dict[f.name] = getattr(default_obj, f.name)

    # 处理额外的默认值
    if extra_defaults:
        for key, value in extra_defaults.items():
            if key not in config_dict:
                config_dict[key] = value

    config = cls(**config_dict)
    print(f"✅ {cls.__name__} loaded from {filepath}")
    return config


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
        config_dict = _serialize_config(self)

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
        return _load_json_common(cls, filepath)

    def pretty_print(self) -> None:
        """格式化打印配置"""
        cfg = asdict(self)

        print("=" * 80)
        print("SFTConfig 全部配置")
        print("=" * 80)

        for k, v in cfg.items():
            print(f"  {k:32s}: {v}")

        print("\n[SamplingParams 实例 (self.sampling_params)]")
        sp = getattr(self, "sampling_params", None)
        if sp:
            print(f"  {'temperature':32s}: {sp.temperature}")
            print(f"  {'top_p':32s}: {sp.top_p}")
            print(f"  {'max_tokens':32s}: {sp.max_tokens}")
            print(f"  {'stop':32s}: {sp.stop}")
            print(f"  {'seed':32s}: {sp.seed}")
            print(f"  {'include_stop_str_in_output':32s}: {sp.include_stop_str_in_output}")
            print(f"  {'n':32s}: {getattr(sp, 'n', None)}")

        print("=" * 80)


# ============ EIConfig ============

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

    def pretty_print(self) -> None:
        """
        格式化打印 EIConfig 的所有字段、以及 SFT/EI 的 SamplingParams 细节。
        """
        cfg = asdict(self)

        print("=" * 80)
        print("EIConfig 全部配置")
        print("=" * 80)

        # 1. 基础/训练相关配置（排除采样字段）
        print("\n[基础与训练配置]")
        for k, v in cfg.items():
            if k.startswith('ei_'):
                continue
            print(f"  {k:32s}: {v}")

        # 2. EI 采样配置
        print("\n[EI 采样配置]")
        for k, v in cfg.items():
            if k.startswith('ei_'):
                print(f"  {k:32s}: {v}")

        # 3. SamplingParams 实例
        print("\n[SFT SamplingParams 实例]")
        sp = getattr(self, "sampling_params", None)
        if sp:
            print(f"  {'temperature':32s}: {sp.temperature}")
            print(f"  {'n':32s}: {getattr(sp, 'n', None)}")

        print("\n[EI SamplingParams 实例]")
        esp = getattr(self, "ei_sampling_params", None)
        if esp:
            print(f"  {'temperature':32s}: {esp.temperature}")
            print(f"  {'n':32s}: {getattr(esp, 'n', None)}")

        print("=" * 80)

    @classmethod
    def from_json(cls, filepath: str) -> "EIConfig":
        """
        从JSON文件加载配置

        Args:
            filepath: JSON文件路径

        Returns:
            EIConfig实例
        """
        # EI 字段的默认值
        extra_defaults = {
            'ei_iterations': cls.ei_iterations,
            'rollout_size': cls.rollout_size,
            'sft_sample_size': cls.sft_sample_size,
            'ei_temperature': cls.ei_temperature,
            'ei_top_p': cls.ei_top_p,
            'ei_max_tokens': cls.ei_max_tokens,
            'ei_stop': cls.ei_stop,
            'ei_include_stop_str_in_output': cls.ei_include_stop_str_in_output,
            'ei_min_tokens': cls.ei_min_tokens,
        }
        return _load_json_common(cls, filepath, extra_defaults)


# ============ GRPOConfig ============

@dataclass
class GRPOConfig(SFTConfig):
    """
    GRPO（Group Relative Policy Optimization）训练配置。
    继承 SFTConfig 的通用字段，并新增 GRPO 相关超参。
    """

    # -------- GRPO / PPO 相关超参 --------
    # 一次 vLLM rollout 展平后的样本数（= n_prompts * group_size）
    rollout_batch_size: int = 128
    # 逻辑上的训练 batch 大小（本作业中可与 rollout_batch_size 相同）
    train_batch_size: int = 128
    # 每次送给 vLLM 的 prompt 数量（GRPODataset.sample_responses 使用）
    vllm_prompt_batch_size: int = 128
    # 单个 micro batch 样本数（对应你代码里的 micro_batch_size）
    micro_batch_size: int = 8
    # 每个 rollout_batch 在离线阶段训练的 epoch 数（Off‑policy 程度）
    n_train_steps_per_rollout_batch: int = 1
    # 每个 prompt 的生成条数（group_size），也是 grpo_sampling_params.n
    group_size: int = 4
    # PPO / GRPO 裁剪范围
    clip_range: float = 0.2
    # advantage 归一化时的 eps
    advantage_eps: float = 1e-8
    # 是否按 std 归一化 advantage
    normalize_by_std: bool = True
    # 是否按 length 归一化 logprobs
    normalize_by_length: bool = True
    # 损失类型（你的 grpo_microbatch_train_step 会用到）
    loss_type: str = "grpo_clip"

    # -------- 额外与 GRPO 相关、在代码中访问到的字段 --------
    # NOTE: SFTConfig 中已经有：
    # - max_grad_norm
    # - normalize_constant
    # - reward_fn / reward_fn_name
    # - sampling_params（SFT 用）
    # 这里不需要重复定义，只要在 __post_init__ 里构造 grpo_sampling_params 即可。

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
            n=self.group_size,  # 必须与 group_size 对齐，GRPODataset/Trainer 里有断言
        )

        # 根据 rollout_batch_size 和 micro_batch_size 推导一个默认的梯度累积步数，
        # 供你在外部计算/检查时使用。
        if self.micro_batch_size > 0:
            self.gradient_accumulation_steps = max(
                1, self.rollout_batch_size // self.micro_batch_size
            )

    @classmethod
    def from_json(cls, filepath: str) -> "GRPOConfig":
        """
        从 JSON 文件加载 GRPOConfig，兼容缺失 GRPO 字段的老配置。
        """
        # GRPO 字段的默认值
        extra_defaults = {
            'rollout_batch_size': cls.rollout_batch_size,
            'train_batch_size': cls.train_batch_size,
            'vllm_prompt_batch_size': cls.vllm_prompt_batch_size,
            'micro_batch_size': cls.micro_batch_size,
            'n_train_steps_per_rollout_batch': cls.n_train_steps_per_rollout_batch,
            'group_size': cls.group_size,
            'clip_range': cls.clip_range,
            'advantage_eps': cls.advantage_eps,
            'normalize_by_std': cls.normalize_by_std,
            'normalize_by_length': cls.normalize_by_length,
            'loss_type': cls.loss_type,
            'grpo_temperature': cls.grpo_temperature,
            'grpo_top_p': cls.grpo_top_p,
            'grpo_max_tokens': cls.grpo_max_tokens,
            'grpo_min_tokens': cls.grpo_min_tokens,
            'grpo_stop': cls.grpo_stop,
            'grpo_include_stop_str_in_output': cls.grpo_include_stop_str_in_output,
        }
        return _load_json_common(cls, filepath, extra_defaults)

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
        for k, v in cfg.items():
            if not k.startswith('grpo_'):
                print(f"  {k:32s}: {v}")

        # 2. GRPO 训练超参
        print("\n[GRPO / PPO 超参数]")
        for k in [
            "rollout_batch_size", "train_batch_size", "vllm_prompt_batch_size",
            "micro_batch_size", "group_size", "clip_range", "advantage_eps",
            "normalize_by_std", "normalize_by_length", "loss_type",
            "n_train_steps_per_rollout_batch"
        ]:
            print(f"  {k:32s}: {cfg.get(k)}")

        # 3. GRPO 采样配置
        print("\n[GRPO 采样配置]")
        for k, v in cfg.items():
            if k.startswith('grpo_'):
                print(f"  {k:32s}: {v}")

        # 4. SamplingParams 实例
        print("\n[SFT SamplingParams 实例]")
        sp = getattr(self, "sampling_params", None)
        if sp:
            print(f"  {'temperature':32s}: {sp.temperature}")
            print(f"  {'n':32s}: {getattr(sp, 'n', None)}")

        print("\n[GRPO SamplingParams 实例]")
        gsp = getattr(self, "grpo_sampling_params", None)
        if gsp:
            print(f"  {'temperature':32s}: {gsp.temperature}")
            print(f"  {'n':32s}: {getattr(gsp, 'n', None)}")

        print("=" * 80)


# 使用示例
if __name__ == "__main__":
    # config = SFTConfig.from_json('cs336_alignment/configs/sft.json')
    # config.pretty_print()

    config = GRPOConfig.from_json('cs336_alignment/configs/grpo_math_no_r_std_no_lnorm.json')
    config.pretty_print()
