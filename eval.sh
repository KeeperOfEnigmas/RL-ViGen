#!/bin/bash

env=dmc
task_name=walker_walk

CUDA_VISIBLE_DEVICES=0 python locodmc_eval.py \
    env=${env} \
    task=${task_name} \
    model_dir="${model_path}" \
    seed=1 \
    action_repeat=2 \
    use_wandb=False \
    use_tb=False \
    save_snapshot=False \
    save_video=True \
    wandb_group=walker_walk
