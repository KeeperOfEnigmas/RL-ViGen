# !/bin/bash
# export MUJOCO_GL=glfw
env=dmc
# env=robosuite
task_list=('walker_walk')
algorithm=("svea")
seed_list=(1 2 3 4 5)
augmentation=("cutmix" "cutout" "no_aug" "overlay" "cropping" "window" "rotation" "flip_v" "flip_h" "convolution" "mix")
eval_augmentation=("vignette" "distortion" "cutmix" "cutout" "overlay" "cropping" "window" "rotation" "flip_h" "flip_v" "convolution")
eval_type=("original" "video" "color")
eval_difficulty=("easy" "hard")
mode=("eval-easy" "eval-medium" "eval-hard")

for algo in "${algorithm[@]}"; do
    for task in "${task_list[@]}"; do
        for seed in "${seed_list[@]}"; do
            for aug in "${augmentation[@]}"; do
                for type in "${eval_type[@]}"; do
                    if  [ "$type" == "original" ]; then
                        for eval_aug in "${eval_augmentation[@]}"; do
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
                                                            # save_video=True \
                                                            # use_wandb=False \
                                                            # +eval_type=original \
                                                            # +eval_difficulty=easy \
                                                            # model_dir=${model_path} \
                        done
                    else
                        if [ "$env" == "dmc" ]; then
                            for difficulty in "${eval_difficulty[@]}"; do
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
                            done
                        else
                            for m in "${mode[@]}"; do
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
                                                                mode=${m} \

                            done
                        fi
                    fi
                done
            done
        done
    done
done
