import json
import os
from typing import Dict, Iterable, List, Literal

"""
gms8k:
dict_keys(['question', 'answer']) answer: cot \n#### answer
math:
dict_keys(['problem', 'solution', 'answer', 'subject', 'level', 'unique_id'])

ri-zero template:
question
cot
answer
"""

def load_json(json_path: str) -> Iterable[Dict]:
    """
    逐行读取 jsonl 文件，并将每一行解析为 Python 字典。

    :param json_path: jsonl 文件路径
    :return: 生成器，每次 yield 一条样本（dict）
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        # 一行一个 JSON，适用于 jsonl 格式
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            yield json.loads(line)


def convert2template(
    files: Iterable[Dict],
    mode: Literal['gms8k', 'math'] = 'gsm8k'
) -> List[Dict[str, str]]:
    """
    将原始数据转换为统一的 ri-zero 模板格式：
    {
        "question": 问题,
        "cot": 思考过程/解题步骤,
        "answer": 最终答案
    }

    :param files: 原始样本的可迭代对象（如 load_json 的返回值）
    :param mode: 数据集类型，'gms8k' 或 'math'
    :return: 格式化后的样本列表
    """
    formatted_files: List[Dict[str, str]] = []
    for file in files:
        if mode == 'gsm8k':
            # gsm8k: answer 字段格式为 "cot 文本\n#### 最终答案"
            raw_answer: str = file['answer']
            # 以 '\n####' 分割为 推理过程 + 最终答案
            parts = raw_answer.split('\n####', maxsplit=1)
            if len(parts) == 2:
                cot_part = parts[0].strip()
                ans_part = parts[1].strip()
            else:
                # 兜底：如果没有按预期分割，就把整个当作 cot，答案留空
                cot_part = raw_answer.strip()
                ans_part = ""

            formatted_files.append({
                'question': file['question'],
                'cot': cot_part,
                'answer': ans_part,
            })

        elif mode == 'math':
            # math 数据集字段名不同
            formatted_files.append({
                'question': file['problem'],
                'cot': file['solution'],
                'answer': file['answer'],
            })
        else:
            raise NotImplementedError(f"Unsupported mode: {mode}")
    return formatted_files


def preprocess_data(
    json_dir: str,
    mode: str,
    out_dir: str,
) -> None:
    """
    读取目录下所有 jsonl 文件，按给定 mode 转换为 r1-zero 模板，
    然后将所有格式化后的样本写入 jsonl 文件（每行一个 JSON）。

    :param json_dir: 原始 jsonl 文件所在目录
    :param mode: 数据集类型，'gms8k' 或 'math'
    :param out_dir: 输出的 jsonl 文件所在目录
    """
    # 确保输出目录存在
    os.makedirs(out_dir, exist_ok=True)

    for json_name in os.listdir(json_dir):
        json_path = os.path.join(json_dir, json_name)

        # 只处理 .jsonl 文件（根据需要可调整）
        if not json_name.endswith('.jsonl'):
            continue

        # 使用生成器逐行读取原始 jsonl
        data_iter = load_json(json_path)
        # 转为统一模板
        formatted_files = convert2template(data_iter, mode=mode)

        # 输出仍然是同名的 .jsonl 文件
        output_path = os.path.join(out_dir, json_name)

        # 逐条写入到输出 jsonl：一条样本一行
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in formatted_files:
                line = json.dumps(item, ensure_ascii=False)
                f.write(line + '\n')

import random
def filter_data(
    train_json_path: str,
    num_samples: int = 128,
) -> None:
    """
    从训练数据中随机采样指定数量的样本，并保存为新的 jsonl 文件。

    Args:
        train_json_path: 训练数据文件路径。
        num_samples: 随机采样的样本数量，默认为 128。
    """
    json_iter = load_json(train_json_path)
    train_data = [json_obj for json_obj in json_iter]
    filtered_data = random.sample(train_data,k=num_samples)
    with open(train_json_path.replace('.jsonl',f'_{num_samples}_filtered.jsonl'),'w',encoding='utf-8') as f:
        for item in filtered_data:
            line = json.dumps(item, ensure_ascii=False)
            f.write(line + '\n')
    print(f"Filtered {num_samples} samples and saved to {train_json_path.replace('.jsonl',f'_{num_samples}_filtered.jsonl')}")

from vllm import LLM,SamplingParams
from vllm_utils import init_vllm,generate_responses
from utils import *
from drgrpo_grader import r1_zero_reward_fn

def filter_correct_data(
    train_json_path: str,
    model_name: str,
    device: str = "cuda",
    seed: int = 42,
    prompt_template_path: str = "cs336_alignment/prompts/r1_zero.prompt",
) -> None:
    """
    读取 train_json_path (r1-zero 预处理后的 jsonl: 包含 question/cot/answer)，
    用 vLLM 生成响应，使用 r1_zero_reward_fn 评估，只保留 reward == 1.0 的样本，
    保存到同目录下 *_correct.jsonl。
    """
    # 1. 读取数据
    data_iter = load_json(train_json_path)
    data_list = [json_obj for json_obj in data_iter]

    if len(data_list) == 0:
        print(f"[filter_correct_data] No data found in {train_json_path}")
        return

    # 2. 构造 prompt 和 ground_truth_answer
    prompt_template = load_template(prompt_template_path)
    prompts: List[str] = []
    answers: List[str] = []
    for obj in data_list:
        # r1-zero 统一模板: question/cot/answer
        prompts.append(apply_r1_template(prompt_template, obj))
        # r1_zero_reward_fn 只需要纯答案
        answers.append(obj["answer"])

    # 3. 初始化 vLLM
    llm: LLM = init_vllm(model_name, device=device, seed=seed)

    # 4. 定义采样参数（与你代码中的 sampling_params 一致）
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        seed=seed,
        include_stop_str_in_output=True,
    )

    # 5. 用 vLLM 生成响应
    responses = generate_responses(llm, prompts, sampling_params)

    # 6. 计算奖励，并筛选 reward == 1.0 的样本
    assert len(responses) == len(data_list) == len(answers)
    correct_items: List[dict] = []
    for obj, resp, gt in zip(data_list, responses, answers):
        reward_dict = r1_zero_reward_fn(resp, gt)
        if reward_dict.get("reward", 0.0) == 1.0:
            correct_items.append(obj)

    # 7. 写回 *_correct.jsonl
    out_path = train_json_path.replace(".jsonl", "_correct.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for item in correct_items:
            line = json.dumps(item, ensure_ascii=False)
            f.write(line + "\n")

    print(
        f"[filter_correct_data] Kept {len(correct_items)}/{len(data_list)} samples. "
        f"Saved to {out_path}"
    )


if __name__ == "__main__":
    train_json_path = 'preprocessed/math/train.jsonl'
    # 过滤样本,仅对math数据集进行消融研究
    # num_samples_list = [128,256,512,1024]
    # for num_samples in num_samples_list:
    #     filter_data(train_json_path,num_samples=num_samples)

    # 过滤回答正确的样本
    filter_correct_data(
        train_json_path,
        'model/Qwen2.5-Math-1.5B',
        seed=42,
        prompt_template_path='cs336_alignment/prompts/r1_zero.prompt',
    )
