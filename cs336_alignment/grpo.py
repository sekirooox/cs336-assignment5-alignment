import torch
from typing import Callable,List,Optional,Union,Literal
from einops import rearrange

@torch.no_grad()
def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
):
    """为每组 rollout 响应计算奖励，并按组进行归一化。

    参数：
    reward_fn: Callable[[str, str], dict[str, float]]  
        用于将 rollout 响应与标准答案（ground truth）进行比较并打分的函数，返回一个字典，
        包含键 "reward"、"format_reward" 和 "answer_reward"。
    
    rollout_responses: list[str]  
        策略生成的 rollout 响应列表。该列表长度为 rollout_batch_size，
        即 rollout_batch_size = n_prompts_per_rollout_batch * group_size。
    
    repeated_ground_truths: list[str]  
        每个样本对应的标准答案列表。该列表长度也为 rollout_batch_size，
        因为每个问题的标准答案被重复了 group_size 次（与每个问题对应的多个响应对齐）。
    
    group_size: int  
        每个问题（即每组）生成的响应数量。
    
    advantage_eps: float  
        用于归一化时避免除零的小常数。
    
    normalize_by_std: bool  
        若为 True，则用每组奖励的标准差进行归一化（即减去均值后除以标准差）；
        否则仅减去组内均值。

    返回：
    tuple[torch.Tensor, torch.Tensor, dict[str, float]]
        - advantages: shape (rollout_batch_size,)，每条 rollout 响应的组内归一化奖励（即优势值）。
        - raw_rewards: shape (rollout_batch_size,)，每条 rollout 响应的原始未归一化奖励。
        - metadata: 用户自定义的其他统计信息，可用于日志记录（例如奖励的均值、标准差、最大/最小值等）。
    """
    batch_size = len(rollout_responses) // group_size

    # 获得每一组的奖励 b*g,
    raw_rewards = [reward_fn(r,g)['reward'] for r,g in zip(rollout_responses,repeated_ground_truths)] # list[float]
    raw_rewards = torch.tensor(raw_rewards) # shape: b*g,

    # 获得优势函数
    rewards = rearrange(raw_rewards,'(b g)->b g',b=batch_size,g=group_size)
    avg_rewards = rewards.mean(dim=-1,keepdim=True)
    if normalize_by_std:
        std_rewards = rewards.std(dim=-1,keepdim=True)
        advantages = (rewards - avg_rewards) / (std_rewards + advantage_eps)
    else:
        advantages = rewards - avg_rewards
    
    advantages  = rearrange(advantages,'b g->(b g)',b=batch_size,g=group_size)# b*g
    metadata = {
        'avg_reward':raw_rewards.mean().item(),
        'std_reward':raw_rewards.std().item(),
        'max_reward':raw_rewards.max().item(),
        'min_reward':raw_rewards.min().item(),
    }
    return advantages, raw_rewards, metadata

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    计算每个token的策略梯度损失，其中raw_rewards_or_advantages可为原始奖励或已归一化的优势值
    
    参数：
        raw_rewards_or_advantages: 形状为(batch_size, 1)的张量，每个滚动响应的标量奖励/优势值
        policy_log_probs: 形状为(batch_size, sequence_length)的张量，每个token的对数概率
    
    返回：
        形状为(batch_size, sequence_length)的张量，逐token策略梯度损失（将在训练循环中跨批次和序列维度聚合）
    """
    return - policy_log_probs * raw_rewards_or_advantages

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """    
    参数：
        advantages: 形状为(batch_size, 1)的张量，每个样本的优势值A
        policy_log_probs: 形状为(batch_size, sequence_length)的张量，待训练策略的逐token对数概率
        old_log_probs: 形状为(batch_size, sequence_length)的张量，旧策略的逐token对数概率
        cliprange: 裁剪参数ε（例如0.2）
    
    返回：
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            loss: 形状为(batch_size, sequence_length)的张量，逐token裁剪损失
            metadata: 需记录的元数据（建议记录每个token是否被裁剪，即min函数右侧的裁剪后损失是否小于左侧）
    """

    # ratio
    ratio = torch.exp(policy_log_probs - old_log_probs) # shape: b l

    # unclipped
    unclipped_part = ratio * advantages # b l

    # clipped 
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange) # b l
    clipped_part = clipped_ratio * advantages # b l

    loss = -torch.min(unclipped_part, clipped_part) # b l

    is_clipped = (clipped_part > unclipped_part).float() # b l
    is_clipped_ratio = is_clipped.sum() / is_clipped.numel()
    metadata = {
        'is_clipped': is_clipped,# b l: float
        'is_clipped_ratio': is_clipped_ratio.item(), # 记录被裁剪的比例
    }
    return loss, metadata

from typing import Literal
def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    advantages: torch.Tensor | None = None,
    raw_rewards: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        assert raw_rewards is not None,"raw_rewards must be provided for no_baseline loss type"
        metadata = {
            'raw_rewards': raw_rewards,
        }
        if raw_rewards.ndim == 1:
            raw_rewards = raw_rewards.unsqueeze(-1) # b -> b 1
        return compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards,
            policy_log_probs=policy_log_probs,
        ), metadata

    if loss_type == "reinforce_with_baseline":
        assert advantages is not None,"advantages must be provided for reinforce_with_baseline loss type"
        metadata = {
            'advantages': advantages,
        }
        if advantages.ndim == 1:
            advantages = advantages.unsqueeze(-1) # b -> b 1
        return compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages,
            policy_log_probs=policy_log_probs,
        ), metadata
    
    if loss_type == "grpo_clip":
        assert advantages is not None,"advantages must be provided for grpo_clip loss type"
        assert old_log_probs is not None,"old_log_probs must be provided for grpo_clip loss type"
        assert cliprange is not None,"cliprange must be provided for grpo_clip loss type"
        metadata = {
            'cliprange':cliprange,
            'advantages':advantages,
            'old_log_probs':old_log_probs,
        }
        if advantages.ndim == 1:
            advantages = advantages.unsqueeze(-1) # b -> b 1
        loss,grpo_metadata = compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )
        metadata.update(grpo_metadata)
        return loss,metadata
        # dict_keys(['cliprange', 'advantages', 'old_log_probs', 'is_clipped', 'is_clipped_ratio'])
def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,# bool
    dim: int | None = None,
) -> torch.Tensor:
    """
    这个代码是错误的,会影响均值: 
    masked_tensor = tensor * mask
    return masked_tensor.mean(dim=dim)
    """
    valid_elements = mask.float().sum(dim=dim)# b l-> b
    masked_tensor = tensor * mask
    masked_sum = masked_tensor.sum(dim=dim) # b l -> b
    masked_sum[valid_elements == 0.0] = 0.0
    return masked_sum / valid_elements

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        advantages=advantages,
        raw_rewards=raw_rewards,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    # 这一步与标准的PPO/GRPO实现不同,\sum{i=1}^T log \pi(a_i|s_i)
    masked_loss = masked_mean(
        tensor = loss,
        mask = response_mask,
        dim = -1,
    ).mean() # b -> 1

    # 梯度累积和反向传播
    scaled_loss = masked_loss / gradient_accumulation_steps
    scaled_loss.backward()
    
    metadata.update({
        'scaled_loss': scaled_loss.item(),# 1
        'loss': masked_loss.item(),# 1
    })
    return scaled_loss, metadata
    # dict_keys(['cliprange', 'advantages', 'old_log_probs', 'is_clipped', 'is_clipped_ratio', 'scaled_loss', 'loss'])