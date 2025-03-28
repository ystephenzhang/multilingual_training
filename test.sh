nproc_per_node=8

NPROC_PER_NODE=$nproc_per_node \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift pt \
    --model ./models/base/Llama-3-8B \
    --model_type llama3 \
    --task_type causal_lm \
    --train_type full \
    --dataset ./corpus_all/french.txt \
    --torch_dtype bfloat16 \
    --streaming true \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps $(expr 256 / $nproc_per_node) \
    --warmup_ratio 0.03 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --deepspeed zero3 \
    --max_length 8192 \
    --max_steps 100000