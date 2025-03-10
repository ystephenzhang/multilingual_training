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

def enhanced_training(model, tokenizer, lang=None, args=None, data_path="./corpus_all/"):
    if not lang:
        print("Vanilla training.")
        output_dir = './model/' + model.name_or_path.split('/')[-1] + '_vanilla'
        if not args:
            args = TrainingArguments(
                        per_device_train_batch_size=8,
                        gradient_accumulation_steps=4,
                        gradient_checkpointing =True,
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
        activate_neuron = read_neuron(neuron_path)
        output_dir = './model/' + model.name_or_path.split('/')[-1] + '_' + lang
        if not args:
            args = TrainingArguments(
                        per_device_train_batch_size=8,
                        gradient_accumulation_steps=4,
                        gradient_checkpointing =True,
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
                    )
            args.activate_neuron = activate_neuron
        else:
            args.output_dir = output_dir
            args.activate_neuron = activate_neuron
    
    pretrain_tokens = load_dataset("text", data_files=data_path + lang + ".txt")
    pretrain_tokens = pretrain_tokens['train'].select(range(1000))

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