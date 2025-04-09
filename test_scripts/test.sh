#!/bin/bash
nproc_per_node=2

NPROC_PER_NODE=$nproc_per_node \
CUDA_VISIBLE_DEVICES=0,1 \
swift pt \
    --model ./models/base/Llama-3.2-1B \
    --model_type llama3_2 \
    --task_type causal_lm \
    --train_type full \
    --dataset /home/zhangyang/multilingual/multilingual_training/corpus_all/french.txt \
    --activate_path ./output/Llama-3.2-1B_english.json \
    --log_grad true \
    --torch_dtype float16 \
    --streaming false \
    --gradient_checkpointing false \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --learning_rate 2e-6 \
    --gradient_accumulation_steps 1 \
    --warmup_ratio 0.03 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --deepspeed zero3 \
    --ddp_backend nccl \
    --max_length 4096 \
    --max_steps -1