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

def construct_prompts_gsm(lang, shots=4, examplar=MGSM_EXEMPLARS):
    path = './url-nlp/mgsm/mgsm_' + lang + '.tsv'
    df = pd.read_csv(path, sep="\t", names=["question","answer"]) 
    df["prompt"] = df["question"].apply(lambda x: few_shot_gsm(x, examplar=examplar, lang=lang, n=shots))
    return df

def construct_prompts_mmlu(lang, shots=4):
    mapping = {"zh":"ZH_CN", "de":"DE_DE", "fr":"FR_FR","sw":"SW_KE"}
    path="openai/MMMLU"
    dataset = load_dataset(path, mapping[lang], split="test")
    df = dataset.to_pandas()
    df = df.rename(columns={"Answer": "answer"})
    examplar = df.sample(n=shots)
    prompts = []
    for i, row in df.iterrows():
        prompts.append(few_shot_mmlu(row, examplar, lang))
    df["prompt"] = prompts
    return df

def evaluate(model_name, mode="sequential", dataset="gsm", lang="zh", full_record=False, shots=8, bsz=16, suffix="before-training", log_name="model"):
    
    if dataset == "gsm":
        df = construct_prompts_gsm(lang, shots=shots)
        mnt = 100
    elif dataset == "mmlu":
        mnt = 20
        df = construct_prompts_mmlu(lang, shots=shots)
    else:
        return NotImplementedError("")

    if mode == "sequential":
        model, tokenizer = load_model_from_name(model_name)
        all_responses = sequential_inference_hf(model, tokenizer, df["prompt"], max_new_tokens=mnt, batch_size=bsz)
        df["generated_answer"] = all_responses
    elif mode == "parallel":
        llm, sampling_params, tokenizer = prepare_vllm(model_name, temperature=0.3, top_p=0.9, max_tokens=mnt)
        responses = parallel_inference_vllm(llm, sampling_params, df["prompt"])
        
        df["generated_answer"] = responses
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
        
    
    
    