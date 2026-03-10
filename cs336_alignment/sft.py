from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List,Dict,Literal,Union,Optional

def tokenize_prompt_and_output(prompt_strs:List[str], output_strs:List[str], tokenizer)->dict[str,torch.Tensor]:
    """
    对提示和输出字符串进行分词，并构建一个掩码，标记响应 token（值为 1），其余（提示或填充）为 0。

    Args:
        prompt_strs: List[str] —— 提示字符串列表。
        output_strs: List[str] —— 输出字符串列表。
        tokenizer: PreTrainedTokenizer —— 用于分词的分词器。

    Returns:
        dict[str, torch.Tensor]：
            设 prompt_and_output_lens 为各拼接后序列的长度列表，
            返回字典包含以下键：
            - input_ids: shape (batch_size, max(prompt_and_output_lens) - 1)
                         拼接后的 token 序列（去掉最后一个 token）
            - labels: shape 同 input_ids，为 input_ids 右移一位（即去掉第一个 token）
            - response_mask: shape 同 input_ids，响应 token 对应位置为 True，其余为 False
    """
    """
    需要注意的点:
    1. tokenizer(prompt)+tokenizer(output)!=tokenizer(prompt+output)
    2. response_mask是针对labels的,需要右移
    3. 所有掩码一律使用二进制, 填充的token要使用pad_id(0可能是有用的token)
    """
    # 填充要填充指定的pad_id,掩码中1代表有用的token,0代表无用的token
    prompt_ids = tokenizer(prompt_strs, 
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,).input_ids
    output_ids = tokenizer(output_strs,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,).input_ids
    input_ids = []
    prompt_masks = []

    # 拼接输入和输出,获得prompt_masks
    for p, o in zip(prompt_ids, output_ids):
        input_ids.append(p + o)
        prompt_masks.append([0] * len(p) + [1] * len(o))

    # 获取最长长度
    max_length = max([len(ids) for ids in input_ids])

    # 获取padding_mask b ? -> b l
    padding_masks = []
    pad_id = tokenizer.pad_token_id
    for i in range(len(input_ids)):
        # [1,1,0]
        padding_masks.append([1] * len(input_ids[i]) + [0] * (max_length - len(input_ids[i])))
        # [0 1 0]
        prompt_masks[i] = prompt_masks[i] + [0] * (max_length - len(prompt_masks[i]))
        # [x x y]
        input_ids[i] = input_ids[i] + [pad_id] * (max_length - len(input_ids[i]))
    
    padding_masks = torch.tensor(padding_masks)
    prompt_masks = torch.tensor(prompt_masks)
    input_ids = torch.tensor(input_ids)
    mask = (padding_masks & prompt_masks).bool()
    return {
        'input_ids': input_ids[:,:-1],# 去掉最后一个token(no label)
        'labels': input_ids[:, 1:].clone(), 
        'response_mask': mask[:,1:]# b l
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
	"""
	功能：获取下一个 token 预测的熵（即在词汇表维度上的熵）。
	
	参数：
	- logits: torch.Tensor，形状为 (batch_size, sequence_length, vocab_size)，包含未归一化的 logits。
	
	返回值：
	- torch.Tensor，形状为 (batch_size, sequence_length)，表示每个下一个 token 预测的熵。
	"""
	log_probs = torch.log_softmax(logits,dim=-1)
	probs = torch.exp(log_probs)
	entropy = - log_probs * probs
	return entropy.sum(dim=-1)

def get_response_log_probs(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    参数:
    - model:PreTrainedModel,用于评分的HuggingFace模型（若无需计算梯度,需放置在正确设备上并处于推理模式）。
    - input_ids:torch.Tensor,形状为（batch_size, sequence_length）,由分词方法生成的拼接后的提示词+响应token。
    - labels:torch.Tensor,形状为（batch_size, sequence_length）,由分词方法生成的标签。
    - return_token_entropy:bool,若为True,通过调用`compute_entropy`额外返回逐token熵。

    返回值:
    - dict[str, torch.Tensor]:
    - "log_probs":形状为（batch_size, sequence_length）,条件对数概率\(log p_{\theta}(x_t | x_{<<t})\)。
    - "token_entropy"（可选）:形状为（batch_size, sequence_length）,每个位置的逐token熵（仅当return_token_entropy=True时存在）。
    """
    logits = model(input_ids).logits  # b l v
    log_probs_all = torch.log_softmax(logits, dim=-1)  # b l v
    
    log_probs = torch.gather(
        log_probs_all, 
        dim=-1, 
        index=labels.unsqueeze(-1)
    ).squeeze(-1)  # b l
    
    if return_token_entropy:
        entropy = compute_entropy(logits)  # b l
        return {
            "log_probs": log_probs,
            "token_entropy": entropy,
        }
    else:
        return {
            "log_probs": log_probs,
            "token_entropy": None,
        }

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """
    对指定维度求和并通过常数归一化，仅考虑掩码中值为1的元素。
    sum(tensor)/constant

    参数：
    - tensor：torch.Tensor，需求和并归一化的张量。
    - mask：torch.Tensor，与tensor形状相同；值为1的位置会被纳入求和范围。
    - normalize_constant：float，用于归一化的除数常数。
    - dim：int | None，归一化前要求和的维度；若为None，对所有维度求和。

    返回值：
    - torch.Tensor，归一化后的和，其中掩码元素（mask == 0）不参与求和。
    """
    masked_tensor = tensor * mask
    summed = masked_tensor.sum(dim=dim,keepdim=True)
    norm = summed / normalize_constant
    return norm.sum(dim=dim)

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    对微批次执行前向传播和反向传播。
	
	参数：
	- policy_log_probs：形状为（batch_size, sequence_length），来自待训练监督微调（SFT）策略的逐token对数概率。
	- response_mask：形状为（batch_size, sequence_length），响应token对应位置为1，提示词/填充token对应位置为0。
	- gradient_accumulation_steps：每个优化器步骤对应的微批次数量。
	- normalize_constant：用于除法归一化的常数，默认设为1.0即可。
	
	返回值：
	- tuple[torch.Tensor, dict[str, torch.Tensor]]：
	  - loss：标量张量，微批次损失（已根据梯度累积进行调整），返回该值用于日志记录。
	  - metadata：字典，包含底层损失调用的元数据及其他需记录的统计信息。
    """
    # SFT的目标函数:-log(y|x)

    loss = - masked_normalize(
        tensor = policy_log_probs,
        mask = response_mask,
        normalize_constant = normalize_constant,
        dim=-1,
    ).mean()
    scaled_loss = loss / gradient_accumulation_steps
    scaled_loss.backward()
    
    return scaled_loss,{
        'loss': scaled_loss,
        'unscaled_loss': loss,
    }