from vllm import LLM, SamplingParams
from utils import load_jsonl
from typing import Iterable, List,Callable
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
from utils import *
from vllm_utils import init_vllm, evaluate_vllm




if __name__=='__main__':
    # init llm,
    # model_path ='checkpoints/checkpoint_499' # local path
    # model_path ='model/Qwen2.5-Math-1.5B' # online path
    model_path = 'model/EI_iteration_5'
    llm = init_vllm(model_path,device=get_device(5),seed=42,gpu_memory_utilization=0.6)
    print('init LLM successfully!')

    # load_template
    template_path = 'cs336_alignment/prompts/r1_zero.prompt'
    r1_template = load_template(template_path)

    # generation
    import os 
    json_path  = 'preprocessed/gsm8k/test.jsonl'
    prompts = get_r1_prompts(json_path,r1_template)
    ground_truths = get_r1_ground_truths(json_path)

    # evaluation
    sampling_params = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    max_tokens=1024,
    min_tokens=32,
    stop = ["</answer>"],
    include_stop_str_in_output = True
    )
    overview =evaluate_vllm(llm,prompts,ground_truths,sampling_params,r1_zero_reward_fn)
    print(overview)

    """
    math:0.466
    gsm8k:0.582
    """
        
    

    