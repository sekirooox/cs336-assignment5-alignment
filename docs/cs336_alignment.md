# cs336_alignment 文件说明

本文件夹包含 LLM 对齐训练的核心实现代码。

---



## 目录
>tests/adapters/ 目录中的测试不能全部通过是正常的，本项目没有实现补充作业

>sft.py、grpo.py中的核心代码都已经验证正确，但是可能存在引用问题，导致Module Not Found问题，可忽略并放心使用。
1. [sft.py - SFT 核心函数](#1-sftpy---sft-核心函数)
2. [grpo.py - GRPO 损失函数](#2-grpopy---grpopy-损失函数)
3. [trainer.py - 主训练器类](#3-trainerpyp---主训练器类)
4. [config.py - 配置数据类](#4-configpy---配置数据类)
5. [drgrpo_grader.py - 奖励函数](#5-drgrpopy---奖励函数)
6. [vllm_utils.py - VLLM 推理工具](#6-vllm_utilspy---vllm-推理工具)
7. [utils.py - 通用工具函数](#7-utilspy---通用工具函数)
8. [preprocess.py - 数据预处理](#8-preprocesspy---数据预处理)
9. [训练入口脚本](#9-训练入口脚本)
10. [配置文件 (configs/)](#10-配置文件-configs)
11. [Prompt 模板 (prompts/)](#11-prompt-模板-prompts)

---

## 1. sft.py - SFT 核心函数

### sft/tokenize_prompt_and_output

**主要用法**：对提示和输出字符串进行分词，并构建一个掩码，标记响应 token（值为 1），其余（提示或填充）为 0。

**输入参数**：
- `prompt_strs: List[str]` — 提示字符串列表
- `output_strs: List[str]` — 输出字符串列表
- `tokenizer: PreTrainedTokenizer` — 用于分词的分词器

**输出参数**：
- `dict[str, torch.Tensor]`，包含：
  - `input_ids`: `torch.Tensor`，shape `(batch_size, max_len - 1)`，拼接后的 token 序列（去掉最后一个 token）
  - `labels`: `torch.Tensor`，shape 同 input_ids，为 input_ids 右移一位
  - `response_mask`: `torch.Tensor`，shape 同 input_ids，响应 token 对应位置为 True，其余为 False

**使用位置**：
- `trainer.py/sft_collate_fn`: 调用此函数进行数据分词
- `trainer.py/GRPOTrainer.train_step`: 在 GRPO 训练中对 rollout 响应分词
- `vllm_utils.py/log_generation`: 对生成的响应分词
- `tests/adapters.py`: 测试适配器

---

### sft/compute_entropy

**主要用法**：获取下一个 token 预测的熵（即在词汇表维度上的熵）。

**输入参数**：
- `logits: torch.Tensor`，shape `(batch_size, sequence_length, vocab_size)`，包含未归一化的 logits

**输出参数**：
- `torch.Tensor`，shape `(batch_size, sequence_length)`，表示每个下一个 token 预测的熵

**使用位置**：
- `sft.py/get_response_log_probs`: 当 `return_token_entropy=True` 时调用

---

### sft/get_response_log_probs

**主要用法**：获取响应的条件对数概率，可选择返回逐 token 熵。

**输入参数**：
- `model: AutoModelForCausalLM` — HuggingFace 预训练模型
- `input_ids: torch.Tensor`，shape `(batch_size, sequence_length)`，分词后的输入
- `labels: torch.Tensor`，shape `(batch_size, sequence_length)`，标签
- `return_token_entropy: bool = False` — 是否返回逐 token 熵

**输出参数**：
- `dict[str, torch.Tensor]`，包含：
  - `log_probs`: `torch.Tensor`，shape `(batch_size, sequence_length)`，条件对数概率
  - `token_entropy`: `torch.Tensor` 或 `None`，shape `(batch_size, sequence_length)`

**使用位置**：
- `trainer.py/SFTTrainer.train_step`: 计算训练步的 log_probs
- `trainer.py/GRPOTrainer.train_step`: 计算 policy 和 old policy 的 log_probs
- `vllm_utils.py/log_generation`: 计算响应的 token 熵

---

### sft/masked_normalize

**主要用法**：带掩码的归一化，仅对掩码值为 1 的元素求和并除以常数。

**输入参数**：
- `tensor: torch.Tensor` — 需求和并归一化的张量
- `mask: torch.Tensor`，与 tensor 形状相同 — 值为 1 的位置会被纳入求和范围
- `normalize_constant: float` — 用于归一化的除数常数
- `dim: int | None = None` — 归一化前要求和的维度

**输出参数**：
- `torch.Tensor` — 归一化后的和

**使用位置**：
- `sft.py/sft_microbatch_train_step`: 计算 SFT 损失
- `grpo.py/grpo_microbatch_train_step`: 计算 GRPO 损失（长度归一化时）

---

### sft/sft_microbatch_train_step

**主要用法**：对微批次执行前向传播和反向传播，计算 SFT 损失。

**输入参数**：
- `policy_log_probs: torch.Tensor`，shape `(batch_size, sequence_length)` — 来自待训练策略的逐 token 对数概率
- `response_mask: torch.Tensor`，shape `(batch_size, sequence_length)` — 响应 token 位置为 1，提示词/填充位置为 0
- `gradient_accumulation_steps: int` — 每个优化器步骤对应的微批次数量
- `normalize_constant: float = 1.0` — 用于除法归一化的常数

**输出参数**：
- `tuple[torch.Tensor, dict[str, torch.Tensor]]`：
  - `loss`: 标量张量，已根据梯度累积进行调整
  - `metadata`: 字典，包含 `loss` 和 `unscaled_loss`

**使用位置**：
- `trainer.py/SFTTrainer.train_step`: 执行 SFT 训练步

---

## 2. grpo.py - GRPO 损失函数

实现 Group Relative Policy Optimization 相关的损失计算。

### grpo/compute_group_normalized_rewards

**主要用法**：为每组 rollout 响应计算奖励，按组归一化（减去均值，可选除以标准差）。

**输入参数**：
- `reward_fn: Callable[[str, str], dict[str, float]]` — 奖励函数
- `rollout_responses: list[str]` — 策略生成的 rollout 响应列表
- `repeated_ground_truths: list[str]` — 每个样本对应的标准答案列表
- `group_size: int` — 每个问题生成的响应数量
- `advantage_eps: float` — 用于归一化时避免除零的小常数
- `normalize_by_std: bool` — 是否用标准差归一化

**输出参数**：
- `tuple[torch.Tensor, torch.Tensor, dict[str, float]]`：
  - `advantages: torch.Tensor`，shape `(rollout_batch_size,)` — 组内归一化奖励
  - `raw_rewards: torch.Tensor`，shape `(rollout_batch_size,)` — 原始未归一化奖励
  - `metadata: dict` — 奖励统计信息（均值、标准差、最大/最小值）

**使用位置**：
- `trainer.py/GRPODataset.get_rollout_batch`: 计算每个 prompt 组的 advantages
- `trainer.py/GRPOTrainer.train_step`: 在训练前预计算 advantages

---

### grpo/compute_grpo_clip_loss

**主要用法**：GRPO 裁剪损失（PPO 风格），防止策略更新过大。

**输入参数**：
- `advantages: torch.Tensor`，shape `(batch_size, 1)` — 每个样本的优势值
- `policy_log_probs: torch.Tensor`，shape `(batch_size, sequence_length)` — 待训练策略的逐 token 对数概率
- `old_log_probs: torch.Tensor`，shape `(batch_size, sequence_length)` — 旧策略的逐 token 对数概率
- `cliprange: float` — 裁剪参数 ε（例如 0.2）

**输出参数**：
- `tuple[torch.Tensor, dict[str, torch.Tensor]]`：
  - `loss: torch.Tensor`，shape `(batch_size, sequence_length)` — 逐 token 裁剪损失
  - `metadata: dict` — 包含 `is_clipped` 和 `is_clipped_ratio`

**使用位置**：
- `grpo.py/compute_policy_gradient_loss`: 当 `loss_type="grpo_clip"` 时调用

---

### grpo/compute_naive_policy_gradient_loss

**主要用法**：基础策略梯度损失，不使用基线。

**输入参数**：
- `raw_rewards_or_advantages: torch.Tensor`，shape `(batch_size, 1)` — 原始奖励或优势值
- `policy_log_probs: torch.Tensor`，shape `(batch_size, sequence_length)` — 策略的逐 token 对数概率

**输出参数**：
- `torch.Tensor`，shape `(batch_size, sequence_length)` — 逐 token 策略梯度损失

**使用位置**：
- `grpo.py/compute_policy_gradient_loss`: 当 `loss_type="no_baseline"` 或 `"reinforce_with_baseline"` 时调用

---

### grpo/compute_policy_gradient_loss

**主要用法**：策略梯度损失包装器，支持三种损失类型。

**输入参数**：
- `policy_log_probs: torch.Tensor`，shape `(batch_size, sequence_length)` — 策略的逐 token 对数概率
- `loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"]` — 损失类型
- `advantages: torch.Tensor | None = None` — 优势值
- `raw_rewards: torch.Tensor | None = None` — 原始奖励
- `old_log_probs: torch.Tensor | None = None` — 旧策略的对数概率
- `cliprange: float | None = None` — 裁剪参数

**输出参数**：
- `tuple[torch.Tensor, dict[str, torch.Tensor]]` — 损失和元数据

**使用位置**：
- `grpo.py/grpo_microbatch_train_step`: 计算策略梯度损失

---

### grpo/masked_mean

**主要用法**：带掩码的张量均值计算。

**输入参数**：
- `tensor: torch.Tensor` — 输入张量
- `mask: torch.Tensor` — 布尔掩码
- `dim: int | None = None` — 沿哪个维度计算均值

**输出参数**：
- `torch.Tensor` — 带掩码的均值

**使用位置**：
- `grpo.py/grpo_microbatch_train_step`: 当 `normalize_by_length=False` 时使用

---

### grpo/grpo_microbatch_train_step

**主要用法**：GRPO 微批次训练步，执行前向传播和反向传播。

**输入参数**：
- `policy_log_probs: torch.Tensor`，shape `(batch_size, sequence_length)` — 策略的逐 token 对数概率
- `response_mask: torch.Tensor`，shape `(batch_size, sequence_length)` — 响应 token 掩码
- `gradient_accumulation_steps: int` — 梯度累积步数
- `loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"]` — 损失类型
- `raw_rewards: torch.Tensor | None = None` — 原始奖励
- `advantages: torch.Tensor | None = None` — 优势值
- `old_log_probs: torch.Tensor | None = None` — 旧策略对数概率
- `cliprange: float | None = None` — 裁剪参数
- `normalize_by_length: bool = True` — 是否按长度归一化
- `normalize_constant: float = 1.0` — 归一化常数

**输出参数**：
- `tuple[torch.Tensor, dict[str, torch.Tensor]]`：
  - `scaled_loss: torch.Tensor` — 缩放后的损失
  - `metadata: dict` — 包含 `scaled_loss`、`loss` 等

**使用位置**：
- `trainer.py/GRPOTrainer.train_step`: 执行 GRPO 训练步

---

## 3. trainer.py - 主训练器类

包含所有训练器实现（SFT、GRPO、EI）。

### trainer/sft_collate_fn

**主要用法**：SFT 数据集的批处理函数，将一批样本整理成模型输入格式。

**输入参数**：
- `batch: List[Tuple[str, str]]` — 批处理数据，每项为 (prompt, ground_truth) 元组
- `tokenizer: AutoTokenizer` — 分词器

**输出参数**：
- `dict[str, Any]`，包含 `input_ids`、`labels`、`response_mask`

**使用位置**：
- `trainer.py/SFTDataset._create_dataloader`: 作为 DataLoader 的 collate_fn

---

### trainer/SFTDataset

**主要用法**：监督微调数据集类，从 JSONL 文件加载数据。

**初始化参数**：
- `json_path: str` — JSONL 文件路径
- `prompt_template_path: str` — Prompt 模板路径

**主要方法**：
- `__len__()`: 返回数据集长度
- `__getitem__(idx)`: 返回 `(prompt, ground_truth, answer)` 三元组
- `from_prompts_and_ground_truths(cls, prompts, ground_truths, answers)`: 类方法，从列表创建数据集

**使用位置**：
- `trainer.py/SFTTrainer.__init__`: 创建训练和测试数据集
- `trainer.py/EITrainer`: 用于创建 SFT 训练数据集

---

### trainer/get_lr_cosine_schedule_with_warmup

**主要用法**：带预热（warmup）的余弦退火学习率调度函数。

**输入参数**：
- `it: int` — 当前步数
- `max_lr: float` — 最大学习率
- `min_lr: float` — 最小学习率
- `warmup_iters: int` — 预热步数
- `cosine_schedule_iters: int` — 余弦退火总步数

**输出参数**：
- `float` — 当前学习率

**使用位置**：
- `trainer.py/SFTTrainer.update_lr`: 更新学习率
- `trainer.py/GRPOTrainer.update_lr`: 更新学习率

---

### trainer/SFTTrainer

**主要用法**：SFT 训练器类，包含完整的训练循环、评估和检查点保存功能。

**初始化参数**：
- `model: AutoModelForCausalLM` — 训练模型
- `tokenizer: AutoTokenizer` — 分词器
- `optimizer: torch.optim.Optimizer` — 优化器
- `config: SFTConfig` — 配置对象
- `vllm: LLM` — VLLM 推理实例

**主要方法**：
- `train_step()`: 执行一个训练步，包含梯度累积
- `train(global_start_step=0)`: 主训练循环
- `evaluate()`: 在测试集上评估模型
- `sample_responses()`: 随机采样测试样本
- `update_lr(it, optimizer)`: 更新学习率
- `save_checkpoint(path)`: 保存模型检查点
- `from_ei_trainer(cls, ei_trainer, train_dataset)`: 类方法，从 EI 训练器创建

**使用位置**：
- `train_sft.py`: 直接实例化并调用 train 方法
- `train_ei.py`: 通过 EITrainer 内部使用

---

### trainer/EIDataset

**主要用法**：Expert Iteration 数据集类，使用 vLLM 进行 rollout 采样。

**初始化参数**：
- `vllm: LLM` — VLLM 实例
- `sampling_params: SamplingParams` — 采样参数
- `reward_fn: Callable` — 奖励函数
- `json_path: str` — 训练数据路径
- `prompt_template_path: str` — Prompt 模板路径
- `sft_sample_size: int` — 采样大小

**主要方法**：
- `sample_responses()`: 随机采样问题和答案
- `get_ei_batch()`: 获取 EI 训练批次，保留答案为正确的 responses

**使用位置**：
- `trainer.py/EITrainer.__init__`: 创建 EI 数据集

---

### trainer/EITrainer

**主要用法**：Expert Iteration 训练器类，执行迭代训练。

**初始化参数**：
- `model: AutoModelForCausalLM` — 训练模型
- `tokenizer: AutoTokenizer` — 分词器
- `optimizer: torch.optim.Optimizer` — 优化器
- `config: EIConfig` — 配置对象
- `vllm: LLM` — VLLM 实例

**主要方法**：
- `train()`: 主训练循环，执行多轮 Expert Iteration

**使用位置**：
- `train_ei.py`: 直接实例化并调用 train 方法

---

### trainer/GRPODataset

**主要用法**：GRPO 数据集类，生成 rollout 批次。

**初始化参数**：
- `vllm: LLM` — VLLM 实例
- `config: GRPOConfig` — GRPO 配置

**主要方法**：
- `sample_responses()`: 随机采样问题和答案
- `get_rollout_batch()`: 获取 GRPO 训练批次，只保留 advantages 不全为 0 的组

**使用位置**：
- `trainer.py/GRPOTrainer.__init__`: 创建 GRPO 数据集

---

### trainer/GRPOTrainer

**主要用法**：GRPO 训练器类，继承自 SFTTrainer。

**初始化参数**：
- `model: AutoModelForCausalLM` — 训练模型
- `tokenizer: AutoTokenizer` — 分词器
- `optimizer: torch.optim.Optimizer` — 优化器
- `config: GRPOConfig` — 配置对象
- `vllm: LLM` — VLLM 实例

**主要方法**：
- `train_step(global_it)`: 执行一个 GRPO 训练步
- `train()`: 主训练循环

**使用位置**：
- `train_grpo.py`: 直接实例化并调用 train 方法

---

## 4. config.py - 配置数据类

定义训练配置的数据类。

### config/SFTConfig

**主要用法**：SFT 训练配置，包含所有超参数、数据路径、采样参数等。

**主要字段**：
- 随机数设置：`seed: int = 42`
- wandb 设置：`project_name`, `name`
- 训练配置：`batch_size`, `gradient_accumulation_steps`, `max_iters`, `start_iters`
- 优化器配置：`weight_decay`, `betas`, `eps`, `max_grad_norm`
- 学习率配置：`max_lr`, `min_lr`, `warmup_iters`, `cosine_schedule_iters`
- 数据集配置：`train_dataset_path`, `test_dataset_path`, `prompt_template_path`
- 评估配置：`eval_interval`, `sample_size`
- 保存配置：`save_interval`, `save_dir`
- VLLM 配置：`temperature`, `top_p`, `max_tokens`, `stop`
- 奖励函数：`reward_fn_name`

**主要方法**：
- `__post_init__()`: 初始化 SamplingParams 和 reward_fn
- `to_json(filepath)`: 保存配置到 JSON 文件
- `from_json(filepath)`: 从 JSON 文件加载配置
- `pretty_print()`: 格式化打印配置

**使用位置**：
- `train_sft.py`: 加载配置并创建 SFTTrainer
- `trainer.py/SFTTrainer`: 接收配置参数

---

### config/EIConfig

**主要用法**：Expert Iteration 配置，继承自 SFTConfig。

**新增字段**：
- `ei_iterations: int = 3` — Expert Iteration 外层循环次数
- `rollout_size: int = 4` — 每个 prompt 的 rollout 次数
- `sft_sample_size: int = 128` — SFT 训练采样大小
- EI 专用采样配置：`ei_temperature`, `ei_top_p`, `ei_max_tokens`, `ei_stop`, `ei_sampling_params`

**使用位置**：
- `train_ei.py`: 加载配置并创建 EITrainer

---

### config/GRPOConfig

**主要用法**：GRPO 训练配置，继承自 SFTConfig。

**新增字段**：
- `rollout_batch_size: int = 128` — 展平后的样本数
- `train_batch_size: int = 128` — 训练 batch 大小
- `vllm_prompt_batch_size: int = 128` — 每次 vLLM 的 prompt 数量
- `micro_batch_size: int = 8` — 微批次大小
- `n_train_steps_per_rollout_batch: int = 1` — 每个 rollout 的训练 epoch 数
- `group_size: int = 4` — 每个 prompt 生成的响应数
- `clip_range: float = 0.2` — PPO/GRPO 裁剪范围
- `advantage_eps: float = 1e-8` — advantage 归一化 eps
- `normalize_by_std: bool = True` — 是否按标准差归一化
- `normalize_by_length: bool = True` — 是否按长度归一化
- `loss_type: str = "grpo_clip"` — 损失类型
- GRPO 专用采样配置：`grpo_temperature`, `grpo_top_p`, `grpo_max_tokens`, `grpo_sampling_params`

**使用位置**：
- `train_grpo.py`: 加载配置并创建 GRPOTrainer

---

## 5. drgrpo_grader.py - 奖励函数

实现 R1-Zero 风格的奖励评分函数。

### drgrpo_grader/r1_zero_reward_fn

**主要用法**：R1-Zero 风格奖励函数，检查格式（</think> <answer>）和答案正确性。

**输入参数**：
- `response: str` — 模型生成的响应
- `ground_truth: str` — 真实答案
- `fast: bool = True` — 是否使用快速模式（跳过 math_verify 验证）

**输出参数**：
- `dict[str, float]`，包含：
  - `format_reward: float` — 格式奖励（格式正确为 1.0，否则为 0.0）
  - `answer_reward: float` — 答案奖励（答案正确为 1.0，否则为 0.0）
  - `reward: float` — 综合奖励（格式和答案都正确为 1.0，否则为 0.0）

**使用位置**：
- `trainer.py/GRPODataset.get_rollout_batch`: 计算 rollout 奖励
- `trainer.py/EIDataset.get_ei_batch`: 筛选正确样本
- `vllm_utils.py/evaluate_vllm`: 评估模型表现
- `vllm_utils.py/log_generation`: 记录生成结果

---

### drgrpo_grader/question_only_reward_fn

**主要用法**：仅检查答案正确性的奖励函数，不强制格式要求。

**输入参数**：
- `response: str` — 模型生成的响应
- `ground_truth: str` — 真实答案
- `fast: bool = True` — 是否使用快速模式

**输出参数**：
- `dict[str, float]` — 格式奖励、答案奖励、综合奖励

**使用位置**：
- 可作为 `config.reward_fn` 的替代选项

---

### drgrpo_grader/grade

**主要用法**：答案评分函数，支持 sympy 和 LaTeX 解析。

**输入参数**：
- `model_answer: str` — 模型生成的答案
- `gt_answer: str` — 标准答案
- `fast: bool = True` — 是否使用快速模式

**输出参数**：
- `bool` — 答案是否正确

**使用位置**：
- `drgrpo_grader/r1_zero_reward_fn`: 内部调用
- `drgrpo_grader/question_only_reward_fn`: 内部调用

---

### drgrpo_grader/extract_answer

**主要用法**：从 LaTeX \boxed 命令中提取答案。

**输入参数**：
- `passage: str` — 包含 \boxed 的文本

**输出参数**：
- `str | None` — 提取的答案，如果不存在则返回 None

**使用位置**：
- `drgrpo_grader/r1_zero_reward_fn`: 提取 <answer> 标签中的答案
- `drgrpo_grader/grade_answer_sympy`: 提取 \boxed 中的答案

---

### drgrpo_grader/grade_answer_sympy

**主要用法**：使用 sympy 进行数学答案评分。

**输入参数**：
- `given_answer: str` — 模型生成的答案
- `ground_truth: str` — 标准答案

**输出参数**：
- `bool` — 答案是否正确

**使用位置**：
- `drgrpo_grader/grade`: 内部调用

---

### drgrpo_grader/grade_answer_mathd

**主要用法**：使用 mathd 方法进行答案评分。

**输入参数**：
- `given_answer: str` — 模型生成的答案
- `ground_truth: str` — 标准答案

**输出参数**：
- `bool` — 答案是否正确

**使用位置**：
- `drgrpo_grader/grade`: 内部调用

---

### drgrpo_grader/is_latex_equal

**主要用法**：检查 LaTeX 格式答案是否相等（慢速模式使用）。

**输入参数**：
- `given_answer: str` — 模型生成的答案
- `ground_truth: str` — 标准答案

**输出参数**：
- `bool` — 答案是否相等

**使用位置**：
- `drgrpo_grader/grade`: 当 `fast=False` 时调用

---

## 6. vllm_utils.py - VLLM 推理工具

定义与 vLLM 相关的基本操作。

### vllm_utils/init_vllm

**主要用法**：初始化 vLLM 模型实例。

**输入参数**：
- `model_id: str` — 模型 ID 或本地路径
- `device: str` — GPU 设备字符串（如 "cuda:0"）
- `seed: int` — 随机种子
- `gpu_memory_utilization: float = 0.85` — GPU 内存利用率

**输出参数**：
- `LLM` — 初始化后的 vLLM 实例

**使用位置**：
- `train_sft.py`: 初始化推理模型
- `train_grpo.py`: 初始化推理模型
- `train_ei.py`: 初始化推理模型
- `eval.py`: 初始化推理模型

---

### vllm_utils/load_policy_into_vllm_instance

**主要用法**：将训练好的策略模型权重加载到 vLLM 实例。

**输入参数**：
- `policy: PreTrainedModel` — HuggingFace 预训练模型（训练中的策略模型）
- `llm: LLM` — vLLM 实例

**输出参数**：
- `None`（原地修改 vLLM 模型权重）

**使用位置**：
- `trainer.py/SFTTrainer.train`: 评估前加载权重
- `trainer.py/GRPOTrainer.train_step`: 每次训练步前加载权重
- `train_ei.py/EITrainer.train`: 每次迭代前加载权重

---

### vllm_utils/generate_responses

**主要用法**：使用 vLLM 生成单条响应。

**输入参数**：
- `vllm: LLM` — vLLM 模型实例
- `prompts: str | List[str]` — 单个或多个 prompt
- `sampling_params: SamplingParams` — 采样参数

**输出参数**：
- `List[str]` — 生成的响应字符串列表

**使用位置**：
- `vllm_utils.py/evaluate_vllm`: 生成评估响应
- `vllm_utils.py/log_generation`: 生成记录响应
- `preprocess.py`: 数据预处理时生成响应

---

### vllm_utils/generate_rollouts

**主要用法**：对每个 prompt 生成多条响应（rollout）。

**输入参数**：
- `vllm: LLM` — vLLM 模型实例
- `prompts: List[str]` — prompt 列表
- `sampling_params: SamplingParams` — 采样参数（n > 1）
- `use_tqdm: bool = True` — 是否显示进度条

**输出参数**：
- `List[List[str]]` — 二维列表，外层每个元素对应一个 prompt，内层每个元素对应一条响应

**使用位置**：
- `trainer.py/EIDataset.get_ei_batch`: 生成 EI 训练样本
- `trainer.py/GRPODataset.get_rollout_batch`: 生成 GRPO rollout

---

### vllm_utils/evaluate_vllm

**主要用法**：使用 vLLM 评估模型在给定 prompts 上的表现。

**输入参数**：
- `vllm: LLM` — vLLM 模型实例
- `prompts: List[str]` — prompt 列表
- `ground_truths: List[str]` — 真实答案列表
- `sampling_params: SamplingParams` — 采样参数
- `reward_fn: Callable` — 奖励函数

**输出参数**：
- `dict[str, int]`，包含：
  - `sample_size`: 样本数量
  - `answer_correct`: 答案正确数
  - `format_correct`: 格式正确数
  - `total_correct`: 完全正确数
  - `accuracy`: 准确率
  - 等其他统计信息

**使用位置**：
- `trainer.py/SFTTrainer.evaluate`: 评估 SFT 模型
- `trainer.py/GRPOTrainer.evaluate`: 评估 GRPO 模型
- `eval.py`: 独立评估脚本

---

### vllm_utils/log_generation

**主要用法**：生成并记录响应，包含详细统计信息。

**输入参数**：
- `prompts: list[str]` — prompt 列表
- `ground_truths: list[str]` — 真实答案列表
- `reward_fn: Callable` — 奖励函数
- `model: AutoModelForCausalLM` — 训练中的策略模型
- `tokenizer: AutoTokenizer` — 分词器
- `vllm: LLM` — vLLM 模型实例
- `sampling_params: SamplingParams` — 采样参数

**输出参数**：
- `dict[str, Any]`，包含：
  - `summary: dict` — 统计摘要（平均奖励、平均熵、平均响应长度等）
  - `rows: list[dict]` — 每条样本的详细信息

**使用位置**：
- `trainer.py/SFTTrainer.train`: 记录采样结果
- `trainer.py/GRPOTrainer.train`: 记录采样结果

---

## 7. utils.py - 通用工具函数

常用辅助函数。

### utils/seed_everything

**主要用法**：设置所有随机种子（Python、NumPy、PyTorch），确保结果可重复。

**输入参数**：
- `seed: int = 42` — 随机种子

**输出参数**：
- `int` — 设置的种子值

**使用位置**：
- 所有训练脚本（train_sft.py, train_grpo.py, train_ei.py）

---

### utils/get_device

**主要用法**：获取指定 GPU 设备。

**输入参数**：
- `rank: int = 0` — GPU 设备编号

**输出参数**：
- `torch.device | None` — CUDA 设备对象，若不可用则返回 None

**使用位置**：
- 所有训练脚本

---

### utils/load_jsonl

**主要用法**：逐行读取 JSONL 文件。

**输入参数**：
- `json_path: str` — JSONL 文件路径

**输出参数**：
- `Iterable` — 每行解析后的 JSON 对象生成器

**使用位置**：
- `utils.py/get_r1_prompts`: 内部调用
- `utils.py/get_r1_ground_truths`: 内部调用

---

### utils/load_template

**主要用法**：加载 prompt 模板文件。

**输入参数**：
- `template_path: str` — 模板文件路径

**输出参数**：
- `str` — 模板内容

**使用位置**：
- `trainer.py/SFTDataset.__init__`: 加载 prompt 模板

---

### utils/apply_r1_template

**主要用法**：将 JSON 对象应用到 R1 模板生成 prompt。

**输入参数**：
- `prompt: str` — 模板字符串，包含 `{question}` 占位符
- `json_obj: dict` — 包含 question 字段的字典

**输出参数**：
- `str` — 格式化后的 prompt

**使用位置**：
- `utils.py/get_r1_prompts`: 内部调用

---

### utils/get_r1_prompts

**主要用法**：从 JSONL 文件加载 prompts 并应用 R1 模板。

**输入参数**：
- `json_path: str` — JSONL 文件路径
- `prompt_template: str` — prompt 模板字符串

**输出参数**：
- `list` — 格式化后的 prompt 列表

**使用位置**：
- `trainer.py/SFTDataset.__init__`: 加载训练/测试 prompts

---

### utils/get_r1_ground_truths

**主要用法**：从 JSONL 文件加载真实答案（仅 answer 部分）。

**输入参数**：
- `json_path: str` — JSONL 文件路径

**输出参数**：
- `list` — 答案字符串列表

**使用位置**：
- `trainer.py/SFTDataset.__init__`: 加载答案

---

### utils/get_r1_ground_truths_with_template

**主要用法**：从 JSONL 文件加载完整答案（包含 cot 和 answer）。

**输入参数**：
- `json_path: str` — JSONL 文件路径
- `ground_truth_template: str = "{cot} </think> <answer> {answer} </answer>"` — 答案模板

**输出参数**：
- `list` — 格式化后的答案列表

**使用位置**：
- `train_sft.py`: 加载训练数据的 ground truth

---

### utils/clear_gpu_memory

**主要用法**：清理 GPU 内存，包括 Python GC 和 CUDA 缓存。

**输入参数**：
- 无

**输出参数**：
- 无

**使用位置**：
- `trainer.py/SFTTrainer.train_step`: 训练步结束后清理
- `trainer.py/GRPOTrainer.train_step`: 训练步结束后清理
- `vllm_utils.py/log_generation`: 记录完成后清理

---

## 8. preprocess.py - 数据预处理

数据预处理相关函数。

### preprocess/load_json

**主要用法**：加载 JSON 文件。

**输入参数**：
- `json_path: str` — JSON 文件路径

**输出参数**：
- `Iterable[Dict]` — JSON 对象生成器

**使用位置**：
- `preprocess.py/convert2template`: 内部调用

---

### preprocess/convert2template

**主要用法**：将数据转换为指定模板格式。

**输入参数**：
- `data: Iterable[Dict]` — 原始数据
- `template: str` — 模板字符串

**输出参数**：
- `list[Dict]` — 转换后的数据列表

**使用位置**：
- `preprocess.py/preprocess_data`: 内部调用

---

### preprocess/preprocess_data

**主要用法**：预处理数据，转换为模板格式并保存。

**输入参数**：
- `input_path: str` — 输入文件路径
- `output_path: str` — 输出文件路径
- `template: str` — 模板字符串
- `has_cot: bool = True` — 是否包含 cot（思考过程）

**输出参数**：
- 无（直接写入输出文件）

**使用位置**：
- 作为独立脚本使用

---

### preprocess/filter_data

**主要用法**：过滤数据，保留指定条件的样本。

**输入参数**：
- `input_path: str` — 输入文件路径
- `output_path: str` — 输出文件路径
- `filter_fn: Callable` — 过滤函数

**使用位置**：
- 作为独立脚本使用

---

### preprocess/filter_correct_data

**主要用法**：过滤数据，只保留答案正确的样本（用于 Expert Iteration）。

**输入参数**：
- `input_path: str` — 输入文件路径
- `output_path: str` — 输出文件路径
- `vllm: LLM` — VLLM 实例
- `sampling_params: SamplingParams` — 采样参数

**使用位置**：
- 作为独立脚本使用

---

## 9. 训练入口脚本

### train_sft.py

**主要用法**：SFT 训练入口脚本。

**功能**：
1. 解析命令行参数（json_path）
2. 加载配置
3. 初始化模型和分词器
4. 初始化 vLLM
5. 创建 SFTTrainer
6. 执行训练循环

**使用函数**：
- `config/SFTConfig.from_json`: 加载配置
- `utils/seed_everything`: 设置随机种子
- `utils/get_device`: 获取设备
- `vllm_utils/init_vllm`: 初始化 vLLM
- `trainer/SFTTrainer`: 执行训练

---

### train_grpo.py

**主要用法**：GRPO 训练入口脚本。

**功能**：
1. 解析命令行参数（json_path）
2. 加载配置
3. 初始化模型和分词器
4. 初始化 vLLM
5. 创建 GRPOTrainer
6. 执行训练循环

**使用函数**：
- `config/GRPOConfig.from_json`: 加载配置
- `utils/seed_everything`: 设置随机种子
- `utils/get_device`: 获取设备
- `vllm_utils/init_vllm`: 初始化 vLLM
- `trainer/GRPOTrainer`: 执行训练

---

### train_ei.py

**主要用法**：Expert Iteration 训练入口脚本。

**功能**：
1. 解析命令行参数
2. 加载配置
3. 初始化模型和分词器
4. 初始化 vLLM
5. 创建 EITrainer
6. 执行迭代训练循环

**使用函数**：
- `config/EIConfig.from_json`: 加载配置
- `utils/seed_everything`: 设置随机种子
- `utils/get_device`: 获取设备
- `vllm_utils/init_vllm`: 初始化 vLLM
- `trainer/EITrainer`: 执行训练

---

### eval.py

**主要用法**：模型评估脚本。

**功能**：
1. 解析命令行参数（json_path）
2. 加载配置
3. 初始化 vLLM
4. 加载测试数据
5. 使用 vLLM 评估模型
6. 输出评估结果

**使用函数**：
- `utils/load_jsonl`: 加载测试数据
- `vllm_utils/init_vllm`: 初始化 vLLM
- `vllm_utils/evaluate_vllm`: 评估模型

---

## 10. 配置文件 (configs/)

| 配置文件 | 用途 |
|---------|------|
| `sft.json` | SFT 训练配置 |
| `grpo.json` | GRPO 训练配置 |
| `grpo_math.json` | GRPO 数学任务配置 |
| `grpo_math_no_r_std.json` | GRPO 无奖励标准差归一化 |
| `grpo_math_no_r_std_no_lnorm.json` | GRPO 无奖励标准差和长度归一化 |
| `sft_math.json` | SFT 数学任务配置 |
| `ei_math_*.json` | Expert Iteration 配置 |

---

## 11. Prompt 模板 (prompts/)

| 模板文件 | 用途 |
|---------|------|
| `r1_zero.prompt` | R1-Zero 风格模板，包含 <think> 和 </answer> 标签 |
| `alpaca_sft.prompt` | Alpaca SFT 格式模板 |
| `question_only.prompt` | 仅问题格式模板 |
| `zero_shot_system_prompt.prompt` | 零样本提示模板 |
