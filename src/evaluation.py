from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset
import torch
import pandas as pd
import re, os, json
from tqdm import tqdm
from .mgsm_exemplars import MGSM_EXEMPLARS
from .utils import *
from .evaluation_utils import *
from .infrastructure import *
import pdb

from typing import Literal

def construct_prompts_gsm(lang, shots=4, examplar=MGSM_EXEMPLARS, path='./'):
    l_path = path + 'mgsm_' + lang + '.tsv'
    df = pd.read_csv(l_path, sep="\t", names=["question","answer"]) 
    df["prompt"] = df["question"].apply(lambda x: few_shot_gsm(x, examplar=examplar, lang=lang, n=shots))
    return df

def construct_prompts_mmlu(lang, shots=4, path="./"):
    mapping = {"zh":"ZH_CN", "de":"DE_DE", "fr":"FR_FR","sw":"SW_KE"}
    #path="openai/MMMLU"
    #l_path = path + "mmlu"
    l_path = path
    dataset = load_dataset(l_path, mapping[lang], split="test")
    df = dataset.to_pandas()
    df = df.rename(columns={"Answer": "answer"})
    prompts = []
    for i, row in df.iterrows():
        examplar = df[df['Subject'] == row['Subject']].sample(n=shots, random_state=42)
        prompts.append(few_shot_mmlu(row, examplar, lang))
    df["prompt"] = prompts
    return df

def construct_inputs_ppl(lang, path="./"):
    l_path = path + lang + '.txt'
    dataset = load_dataset("text", data_files=l_path, split="train")
    return dataset

def evaluate(model_name, mode: Literal["sequential", "prallel", "perplexity"] = "sequential",
             dataset: Literal["gsm", "mmlu", "ppl"] = "gsm", lang: Literal["zh", "de", "fr", "sw", "th", "en"] = "zh"
             , full_record=False, shots=8, bsz=16, suffix="before-training", log_name="model", path="./"):
    
    if dataset == "gsm":
        df = construct_prompts_gsm(lang, shots=shots, path=path + 'mgsm/')
        mnt = 100
    elif dataset == "mmlu":
        mnt = 2
        df = construct_prompts_mmlu(lang, shots=shots, path=path + 'mmlu')
    elif dataset == "ppl":
        df = construct_inputs_ppl(lang, path='./corpus_all/')
        if not mode == "perplexity":
            print("Parallel inference for ppl test not implemented. Switching to sequential.")
            mode = "perplexity"
        mnt = 2048
    else:
        return NotImplementedError("")

    if mode == "sequential":
        model, tokenizer = load_model_from_name(model_name)
        all_responses = sequential_inference_hf(model, tokenizer, df["prompt"], max_new_tokens=mnt, batch_size=bsz)
        df["generated_answer"] = all_responses
    elif mode == "parallel":
        llm, sampling_params, tokenizer = prepare_vllm(model_name, temperature=0.3, top_p=0.9, max_tokens=mnt, tensor_parallel_size=2)
        responses = parallel_inference_vllm(llm, sampling_params, df["prompt"])
        
        df["generated_answer"] = responses
    elif mode == "perplexity":
        llm, sampling_params, tokenizer = prepare_vllm(model_name, temperature=0.3, top_p=0.9,
                                                       max_tokens=mnt, tensor_parallel_size=2, return_logprob=1)
        def filter_by_length(example):
            tokenized = tokenizer(
                example["text"],  # 替换为你自己的字段名
                truncation=False, # 不截断，让我们知道真实长度
                add_special_tokens=True
            )
            return len(tokenized["input_ids"]) <= mnt
        filtered_dataset = df.filter(
            filter_by_length,
            batched=False,  # 单条过滤
            desc="Filtering samples longer than max_length"
        )
        nll, tokens, ppl = parallel_ppl_vllm(llm, sampling_params, filtered_dataset["text"])
        log_path = './output/eval_log/' + log_name + '_' + lang + '_' + suffix + '_ppl.json'   
        with open(log_path, "w") as f:
            json.dump({
                "nll":nll,
                "tokens":tokens,
                "ppl":ppl
            }, f)
        return ppl
    else:
        return NotImplementedError("")    
    
    if dataset == "gsm":
        log_path = './output/eval_log/' + log_name + '_' + lang + '_' + suffix + '_gsm.json'
    elif dataset == "mmlu":
        log_path = './output/eval_log/' + log_name + '_' + lang + '_' + suffix + '_mmlu.json'
    correct, mapped = answer_mapping_batch(df, lang, dataset)
    acc = correct.count(1)
    with open(log_path, 'w') as f:
        json.dump([{"correct": acc, "wrong": len(correct) - acc, "accuracy": acc / len(correct)}], f)

    if full_record:
        with open(log_path, 'r') as f:
            log = json.load(f)
        for i, b in df.iterrows():
            log.append({"INPUT": b['prompt'], "GT": b['answer'], "GENERATED": b['generated_answer'], "MAPPED": mapped[i]})
        with open(log_path, 'w') as f:
            json.dump(log, f)
    
    return acc / len(correct)
        
    
    
    