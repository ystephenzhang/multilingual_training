import os, json
import torch

from transformers import Trainer, DataCollatorForLanguageModeling, TrainingArguments
from datasets import load_dataset
import random
import pdb
from tqdm import tqdm
from datasets import Dataset
import re

from .utils import *

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

def reverse_training(model_name, n_lang="english", lang=None, args=None, data_path="./assets/", output_path="/mnt/file1/zhangyang/multilingual_data/models/", top_k=-1):
    model, tokenizer = load_model_from_name(model_name)
    mother_path = "./output/" + model.name_or_path.split('/')[-1] + '_' + n_lang + '.json'
    activate_neuron = read_neuron(mother_path, top_k = top_k)
    output_dir = output_path + model.name_or_path.split('/')[-1] + '_' + n_lang + '-to-' + lang
    if not args:
        args = TrainingArguments(
                        per_device_train_batch_size=6,
                        gradient_accumulation_steps=4,
                        gradient_checkpointing = False,
                        max_grad_norm= 0.3,
                        num_train_epochs=1, 
                        learning_rate=5e-6,
                        bf16=True,
                        save_steps=600,
                        save_total_limit=0,
                        logging_steps=10,
                        optim="paged_adamw_8bit",
                        output_dir=output_dir,
                        logging_dir=output_dir + "/logs",
                        fp16=False,
                        lr_scheduler_type="cosine",
                        warmup_ratio=0.05,
                    )
        args.activate_neuron = activate_neuron
        args.log_grad = False
    else:
        args.output_dir = output_dir
        args.activate_neuron = activate_neuron
    pretrain_tokens = load_dataset("text", data_files=data_path + lang + ".txt")
    #pretrain_tokens = pretrain_tokens['train'].select(range(corpus_size))
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
        torch.cuda.synchronize()  # 确保所有 CUDA 操作完成