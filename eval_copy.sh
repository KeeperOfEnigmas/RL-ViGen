# !/bin/bash
export MUJOCO_GL=glfw
# env=dmc
env=robosuite
# task_list=("cheetah_run")
task_list=("Lift")
algorithm=(pieg)
# seed_list=(1 2 3 4 5)
# seed_list=(5)
seed_list=(1)
# augmentation=("cutmix" "cutout" "no_aug" "overlay" "cropping" "window" "rotation" "flip_v" "flip_h" "convolution" "mix")
# augmentation=("rotation" "flip_v" "flip_h")
augmentation=("overlay")
eval_augmentation=("vignette")
# eval_augmentation=("vignette" "distortion" "cutmix" "cutout" "overlay" "cropping" "window" "rotation" "flip_h" "flip_v" "convolution")
# eval_type=("original" "video" "color")
# eval_type=("video")
# eval_type=("original" "mode")
eval_type=("mode")
# eval_difficulty=("easy")
# eval_difficulty=("easy" "hard")
mode=("eval-hard")
# mode=("eval-easy" "eval-medium" "eval-hard")
# mode=("train")
action_repeat=2
# action_repeat=1 # 1 for robotic, 2 otherwise

for algo in "${algorithm[@]}"; do
    for task in "${task_list[@]}"; do
        for seed in "${seed_list[@]}"; do
            for aug in "${augmentation[@]}"; do
                for type in "${eval_type[@]}"; do
                    if  [ "$type" == "original" ]; then
                        for eval_aug in "${eval_augmentation[@]}"; do
                            # model_path="/home/weiyi/RL-ViGen/exp_local/${algo}/${task}/${seed}/${aug}/snapshot.pt"
                            CUDA_VISIBLE_DEVICES=0 python locodmc_eval.py \
                                                            env=${env} \
                                                            task=${task} \
                                                            seed=${seed} \
                                                            action_repeat=${action_repeat} \
                                                            use_tb=False \
                                                            save_snapshot=False \
                                                            +eval_aug=${eval_aug} \
                                                            +aug=${aug} \
                                                            wandb_group=walker_walk \
                                                            save_video=True \
                                                            mode=train \
                                                            # use_wandb=False \
                                                            # +eval_type=original \
                                                            # +eval_difficulty=easy \
                                                            # model_dir=${model_path} \
                        done
                    else
                        if [ "$env" == "dmc" ]; then
                            for difficulty in "${eval_difficulty[@]}"; do
                                # model_path="/home/weiyi/RL-ViGen/exp_local/${algo}/${task}/${seed}/${aug}/snapshot.pt"
                                CUDA_VISIBLE_DEVICES=0 python locodmc_eval.py \
                                                                env=${env} \
                                                                task=${task} \
                                                                seed=${seed} \
                                                                action_repeat=${action_repeat} \
                                                                use_tb=False \
                                                                save_snapshot=False \
                                                                wandb_group=walker_walk \
                                                                eval_type=${type} \
                                                                eval_difficulty=${difficulty} \
                                                                +aug=${aug} \
                                                                save_video=True \
                                                                # use_wandb=False \
                                                                # +eval_type=${type} \
                                                                # +eval_difficulty=${difficulty} \
                                                                # model_dir=${model_path} \
                            done
                        else
                            for m in "${mode[@]}"; do
                                # model_path="/home/weiyi/RL-ViGen/exp_local/${algo}/${task}/${seed}/${aug}/snapshot.pt"
                                CUDA_VISIBLE_DEVICES=0 python locodmc_eval.py \
                                                                env=${env} \
                                                                task=${task} \
                                                                seed=${seed} \
                                                                action_repeat=${action_repeat} \
                                                                use_tb=False \
                                                                save_snapshot=False \
                                                                wandb_group=walker_walk \
                                                                +eval_aug=no_aug \
                                                                +aug=${aug} \
                                                                mode=${m} \
                                                                save_video=True \

                            done
                        fi
                    fi
                done
            done
        done
    done
done