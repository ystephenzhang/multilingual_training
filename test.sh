#!/bin/bash
nproc_per_node=4

NPROC_PER_NODE=$nproc_per_node \
CUDA_VISIBLE_DEVICES=2,3,4,5 \
swift pt \
    --model ./models/base/Llama-3-8B \
    --model_type llama3 \
    --task_type causal_lm \
    --train_type full \
    --dataset /home/zhangyang/multilingual/multilingual_training/corpus_all/sw.txt \
    --torch_dtype float16 \
    --streaming false \
    --gradient_checkpointing true \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
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