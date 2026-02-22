#!/bin/bash
set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen_05_sp2.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=data/sft/train_sft.parquet \
    data.val_files=data/sft/train_sft.parquet \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.micro_batch_size=4 \
    data.max_length=16384 \
    data.truncation=right \
    model.partial_pretrain=Qwen/Qwen3-14B \
    trainer.default_local_dir=$save_path \
    trainer.project_name=evods \
    trainer.experiment_name=evods-sft-qwen3-14b \
    trainer.logger=wandb \
    trainer.total_epochs=3 \
    trainer.save_freq=200 \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true
