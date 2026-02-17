#!/bin/bash
export MUJOCO_GL=glfw
# env=dmc
env=robosuite
# task_list=(Door)
task_list=(TwoArmLift)
algorithm=(pieg)
# seed_list=(1 2 3 4 5)
# seed_list=(5)
seed_list=(1)
# augmentation=("cutmix" "cutout" "no_aug" "overlay" "cropping" "window" "rotation" "flip_v" "flip_h" "convolution" "mix")
# augmentation=("cutmix" "cutout" "no_aug" "cropping" "window")
# augmentation=("rotation" "flip_v" "flip_h" "convolution" "mix")
augmentation=("no_aug")
eval_augmentation=("no_aug")
# eval_augmentation=("vignette" "distortion" "cutmix" "cutout" "overlay" "cropping" "window" "rotation" "flip_h" "flip_v" "convolution")
# eval_type=("original" "video" "color")
# eval_type=("video" "color")
eval_type=("original")
# eval_difficulty=("easy" "hard")
eval_difficulty=("easy")
mode=("eval-easy" "eval-medium" "eval-hard")
mode=("train")

for algo in "${algorithm[@]}"; do
    for task in "${task_list[@]}"; do
        for seed in "${seed_list[@]}"; do
            for aug in "${augmentation[@]}"; do
                for type in "${eval_type[@]}"; do
                    if [ "$env" == "dmc" ]; then
                        if  [ "$type" == "original" ]; then
                            for eval_aug in "${eval_augmentation[@]}"; do
                                # model_path="/home/weiyi/RL-ViGen/exp_local/${algo}/${task}/${seed}/${aug}/snapshot.pt"
                                CUDA_VISIBLE_DEVICES=0 python locodmc_eval.py \
                                                                env=${env} \
                                                                task=${task} \
                                                                seed=${seed} \
                                                                action_repeat=2 \
                                                                use_tb=False \
                                                                save_snapshot=False \
                                                                +eval_aug=${eval_aug} \
                                                                +aug=${aug} \
                                                                wandb_group=walker_walk \
                                                                +mode=eval-hard \
                                                                # save_video=True \
                                                                # use_wandb=False \
                                                                # +eval_type=original \
                                                                # +eval_difficulty=easy \
                                                                # model_dir=${model_path} \
                                                                # +aug=${aug} \
                            
                            done
                        else
                            for difficulty in "${eval_difficulty[@]}"; do
                                # model_path="/home/weiyi/RL-ViGen/exp_local/${algo}/${task}/${seed}/${aug}/snapshot.pt"
                                CUDA_VISIBLE_DEVICES=0 python locodmc_eval.py \
                                                                env=${env} \
                                                                task=${task} \
                                                                seed=${seed} \
                                                                action_repeat=2 \
                                                                use_tb=False \
                                                                save_snapshot=False \
                                                                wandb_group=walker_walk \
                                                                eval_type=${type} \
                                                                eval_difficulty=${difficulty} \
                                                                +eval_aug=${type}_${difficulty} \
                                                                +aug=${aug} \
                                                                # save_video=True \
                                                                # use_wandb=False \
                                                                # +eval_type=${type} \
                                                                # +eval_difficulty=${difficulty} \
                                                                # model_dir=${model_path} \
                                                                # +aug=${aug} \
                            
                            done
                        fi
                    else
                        for m in "${mode[@]}"; do
                                # model_path="/home/weiyi/RL-ViGen/exp_local/${algo}/${task}/${seed}/${aug}/snapshot.pt"
                                CUDA_VISIBLE_DEVICES=0 python locodmc_eval.py \
                                                                env=${env} \
                                                                task=${task} \
                                                                seed=${seed} \
                                                                action_repeat=2 \
                                                                use_tb=False \
                                                                save_snapshot=False \
                                                                wandb_group=walker_walk \
                                                                +eval_aug=no_aug \
                                                                +aug=${aug} \
                                                                +mode=${m} \

                            done
                    fi
                done
            done
        done
    done
done

# redo pieg/pendulum_swingup/1, 2, 3, all evaluation augmentations
