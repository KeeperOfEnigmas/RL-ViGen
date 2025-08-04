#!/bin/bash

env=dmc
task_list=(pendulum_swingup)
algorithm=(svea)
seed_list=(1 2 3 4 5)
# seed_list=(1)
augmentation=("cutmix" "cutout" "no_aug" "default" "cropping" "window" "rotation" "flip_v" "flip_h" "convolution" "mix")
# augmentation=("cutmix")
eval_augmentation=("convolution")

for algo in ${algorithm[@]}; do
    for task in ${task_list[@]}; do
        for seed in ${seed_list[@]}; do
            for aug in ${augmentation[@]}; do
                for eval_aug in ${eval_augmentation[@]}; do
                    model_path="/home/weiyi/RL-ViGen/exp_local/${algo}/${task}/${seed}/${aug}/snapshot.pt"
                    CUDA_VISIBLE_DEVICES=0 python locodmc_eval.py \
                                                    env=${env} \
                                                    task=${task} \
                                                    model_dir=${model_path} \
                                                    seed=${seed} \
                                                    action_repeat=2 \
                                                    use_wandb=False \
                                                    use_tb=False \
                                                    save_snapshot=False \
                                                    save_video=True \
                                                    wandb_group=walker_walk \
                                                    +eval_aug=${eval_aug} \

                    done
                done
            done
        done
    done

