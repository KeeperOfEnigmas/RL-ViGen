#!/bin/bash

env=dmc
task_list=(pendulum_swingup)
algorithm=(svea)
seed_list=(1 2 3 4 5)
# seed_list=(1)
augmentation=("cutmix" "cutout" "no_aug" "default" "cropping" "window" "rotation" "flip_v" "flip_h" "convolution" "mix")
# augmentation=("cutmix")
eval_augmentation=("distortion" "cutmix" "cutout" "overlay" "cropping" "window" "rotation" "flip_h" "flip_v" "convolution")
# eval_type=("original" "color" "video")
eval_type=("original")
eval_difficulty=("easy" "hard")

for algo in "${algorithm[@]}"; do
    for task in "${task_list[@]}"; do
        for seed in "${seed_list[@]}"; do
            for aug in "${augmentation[@]}"; do
                for type in "${eval_type[@]}"; do
                    if  [ "$type" == "original" ]; then
                        for eval_aug in "${eval_augmentation[@]}"; do
                            model_path="/home/weiyi/RL-ViGen/exp_local/${algo}/${task}/${seed}/${aug}/snapshot.pt"
                            CUDA_VISIBLE_DEVICES=0 python locodmc_eval.py \
                                                            env=${env} \
                                                            task=${task} \
                                                            seed=${seed} \
                                                            action_repeat=2 \
                                                            use_wandb=False \
                                                            use_tb=False \
                                                            save_snapshot=False \
                                                            save_video=True \
                                                            wandb_group=walker_walk \
                                                            +eval_aug=${eval_aug} \
                                                            model_dir=${model_path} \
                                                            +eval_type=original \
                                                            +eval_difficulty=easy \
                        
                        done
                    else
                        for difficulty in "${eval_difficulty[@]}"; do
                            model_path="/home/weiyi/RL-ViGen/exp_local/${algo}/${task}/${seed}/${aug}/snapshot.pt"
                            CUDA_VISIBLE_DEVICES=0 python locodmc_eval.py \
                                                            env=${env} \
                                                            task=${task} \
                                                            seed=${seed} \
                                                            action_repeat=2 \
                                                            use_wandb=False \
                                                            use_tb=False \
                                                            save_snapshot=False \
                                                            save_video=True \
                                                            wandb_group=walker_walk \
                                                            model_dir=${model_path} \
                                                            +eval_type=${type} \
                                                            +eval_difficulty=${difficulty} \
                        
                        done
                    fi
                done
            done
        done
    done
done


