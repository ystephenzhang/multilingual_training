# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/usr/local/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/usr/local/conda/etc/profile.d/conda.sh" ]; then
        . "/usr/local/conda/etc/profile.d/conda.sh"
    else
        export PATH="/usr/local/conda/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
cd /mnt/workspace/workgroup/workgroup_v100/yiran/ms-swift
source activate /mnt/workspace/workgroup/workgroup_v100/yiran/BabelTrain
export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export NCCL_IB_TIMEOUT=36000
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=36000
export NCCL_LAUNCH_MODE=PARALLEL
export TOKENIZERS_PARALLELISM=false
export NCCL_TIMEOUT=36000
NNODES=4 NODE_RANK=$RANK MASTER_ADDR=$MASTER_ADDR NPROC_PER_NODE=8 
swift pt \
--model /mnt/workspace/workgroup/workgroup_v100/yiran/Babel/Add_Layer/Train_9.2B_Stage1_dlc_32_cont/v0-20250110-075355/checkpoint-8000 \
--output_dir /mnt/workspace/workgroup_data/yiran/Train_9.2B_Stage1_dlc_32_All_round3_205G_Cont_4e-6_new_back \
--deepspeed zero1 \
--dataset /mnt/workspace/workgroup/workgroup_v100/yiran/Babel_Language_Shuffled_Sampled_1/round3_all/round3_new/round3_all_205G_new_shuffle_format.jsonl \
--num_train_epochs 1 \
--streaming true \
--train_type full \
--model_type qwen2 \
--learning_rate 4e-06 \
--max_steps 50000 \
--per_device_train_batch_size=1 \
--gradient_accumulation_steps=16 \
--max_length 4096 \
--save_steps=2000 \
--ddp_backend nccl \
--warmup_ratio=0.1 \
--lr_scheduler_type "cosine"\
--ddp_timeout 999999999999 \
--freeze_vit true