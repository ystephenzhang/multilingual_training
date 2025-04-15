import os, json
import torch
import subprocess

from typing import Literal
from transformers import Trainer, DataCollatorForLanguageModeling, TrainingArguments
from datasets import load_dataset
import random
import pdb
from tqdm import tqdm
from datasets import Dataset
import re

from itertools import islice
from .utils import *

args_pt = """
NPROC_PER_NODE={n} CUDA_VISIBLE_DEVICES={n_devices} swift pt \
    --model {base} \
    --model_type {type} \
    --output_dir {output} \
    --task_type causal_lm \
    --train_type full \
    --dataset {token} \
    --activate_path {activate_path} \
    --activate_layers {activate_layers} \
    --activate_types {activate_types} \
    --log_grad {log_grad} \
    --torch_dtype float16 \
    --streaming false \
    --gradient_checkpointing false \
    --num_train_epochs 1 \
    --per_device_train_batch_size {b_size} \
    --learning_rate {lr} \
    --gradient_accumulation_steps {g_acc} \
    --warmup_ratio 0.03 \
    --eval_steps {e_step} \
    --save_steps {s_step} \
    --save_total_limit 2 \
    --logging_steps 5 \
    --deepspeed {deepspeed} \
    --max_length {max_len} \
    --max_steps -1
"""

def enhanced_training(model, tokenizer, lang=None, args=None, data_path="/mnt/file1/zhangyang/multilingual_data/data", output_path="/mnt/file1/zhangyang/multilingual_data/models/", top_k = 600, corpus_size = 5000):
    if not lang:
        print("Vanilla training.")
        output_dir = './model/' + model.name_or_path.split('/')[-1] + '_vanilla'
        if not args:
            args = TrainingArguments(
                        per_device_train_batch_size=8,
                        gradient_accumulation_steps=4,
                        gradient_checkpointing = True,
                        max_grad_norm= 0.3,
                        num_train_epochs=1, 
                        learning_rate=2e-6,
                        bf16=True,
                        save_steps=500,
                        save_total_limit=0,
                        logging_steps=10,
                        output_dir=output_dir,
                        optim="paged_adamw_32bit",
                        lr_scheduler_type="cosine",
                        warmup_ratio=0.05,
                        activate_neuron=None,
                    )
    else:
        neuron_path = "./output/" + model.name_or_path.split('/')[-1] + '_' + lang + '.json'
        assert os.path.exists(neuron_path)
        activate_neuron = read_neuron(neuron_path, top_k = top_k)
        output_dir = './models/' + model.name_or_path.split('/')[-1] + '_' + lang
        if not args:
            args = TrainingArguments(
                        per_device_train_batch_size=8,
                        gradient_accumulation_steps=4,
                        gradient_checkpointing = False,
                        max_grad_norm= 0.3,
                        num_train_epochs=1, 
                        learning_rate=2e-6,
                        bf16=True,
                        save_steps=500,
                        save_total_limit=0,
                        logging_steps=10,
                        output_dir=output_dir,
                        logging_dir=output_dir + "/logs",
                        optim="paged_adamw_32bit",
                        lr_scheduler_type="cosine",
                        warmup_ratio=0.05,
                    )
            args.activate_neuron = activate_neuron
        else:
            args.output_dir = output_dir
            args.activate_neuron = activate_neuron
    
    pretrain_tokens = load_dataset("text", data_files=data_path + lang + ".txt")
    pretrain_tokens = pretrain_tokens['train'].select(range(corpus_size))

    tokenizer.pad_token = tokenizer.eos_token
    def tokenize_function(examples):
        sample = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
        sample["labels"] = sample["input_ids"].copy()
        return sample
    tokenized_datasets = pretrain_tokens.map(tokenize_function, batched=True, remove_columns=["text"])

    trainer = Trainer(model=model,
                    args=args,
                    train_dataset=tokenized_datasets,
                    tokenizer=tokenizer,
                    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)  # 这里 `mlm=False`，因为 LLaMA 不是 BERT
    )
    trainer.train()

    return trainer.model

def reverse_training(model_name, n_lang="english", lang=None, mode: Literal["hf", "swift"]="hf", data_path="./assets/", output_path="/mnt/file1/zhangyang/multilingual_data/models/", top_k=-1, training_size=10000, kwargs=None):
    mother_path = "./output/" + model_name.split('/')[-1] + '_' + n_lang + '.json'
    output_dir = output_path + model_name.split('/')[-1] + '_' + n_lang + '-to-' + lang
    if mode == "hf":
        activate_neuron = read_neuron(mother_path, top_k = top_k)
        model, tokenizer = load_model_from_name(model_name)
        args = TrainingArguments(
                        per_device_train_batch_size=4,
                        gradient_accumulation_steps=4,
                        gradient_checkpointing = False,
                        max_grad_norm= 0.3,
                        num_train_epochs=1, 
                        learning_rate=5e-6,
                        bf16=False,
                        save_steps=600,
                        save_total_limit=0,
                        logging_steps=10,
                        optim="paged_adamw_8bit",
                        output_dir=output_dir,
                        logging_dir=output_dir + "/logs",
                        fp16=True,
                        lr_scheduler_type="cosine",
                        warmup_ratio=0.05,
                    )
        args.activate_neuron = activate_neuron
        args.log_grad = False
    
        #pretrain_tokens = load_dataset("text", data_files=data_path + lang + ".txt")
        pretrain_tokens = load_dataset("text", data_files="./corpus_all/" + lang + ".txt")
        #pretrain_tokens = pretrain_tokens['train'].shuffle(seed=42).select(range(training_size))
        pretrain_tokens = pretrain_tokens['train']

        tokenizer.pad_token = tokenizer.eos_token
        def tokenize_function(examples):
            sample = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
            sample["labels"] = sample["input_ids"].copy()
            return sample
        tokenized_datasets = pretrain_tokens.map(tokenize_function, batched=True, remove_columns=["text"])

        trainer = Trainer(model=model,
                        args=args,
                        train_dataset=tokenized_datasets,
                        tokenizer=tokenizer,
                        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)  # 这里 `mlm=False`，因为 LLaMA 不是 BERT
        )
        trainer.train()

        del trainer
        torch.cuda.empty_cache()  # 清空缓存
        with torch.no_grad():
            torch.cuda.reset_peak_memory_stats()  # 重置显存统计信息
            torch.cuda.synchronize()  # 确保所有 CUDA 操作完成`
        
    elif mode=="swift":
        assert kwargs is not None 
        kwargs["output"] = output_dir
        kwargs["base"] = model_name
        kwargs["type"] = "llama3" if "Llama-3-" in model_name else "llama3_2"
        kwargs["token"] = data_path + lang + ".txt"
        
        bash = args_pt.format(**kwargs)
        subprocess.run(bash, shell=True, executable='/bin/bash')
    
    return output_dir
       