easy_task_list=('pendulum_swingup' 'walker_walk')
seed_list=(1 2 3 4 5)
algorithm=("svea" "pieg")
eval_type=("video" "color")
eval_difficulty=("hard" "easy")
#augmentation=("default" "cropping" "window" "rotation" "flip_v" "flip_h" "convolution" "cutout" "cutmix" "no_aug") 
augmentation=("cutmix" "cutout" "no_aug" "default" "cropping" "window" "rotation" "flip_v" "flip_h" "convolution" "mix") 
frames=1001000
feature_dim=50
sgqn_quantile=0.93
action_repeat=2
aux_lr=8e-5
env=dmc

for task_name in ${easy_task_list[@]}; do
    for seed in ${seed_list[@]}; do
        for aug in ${augmentation[@]}; do
            for algo in ${algorithm[@]}; do
                for type in ${eval_type[@]}; do
                    for difficulty in ${eval_difficulty[@]}; do
                        CUDA_VISIBLE_DEVICES=0  python train.py \
                                                    env=${env} \
                                                    task_name=${task_name} \
                                                    seed=${seed} \
                                                    action_repeat=${action_repeat} \
                                                    use_wandb=False \
                                                    use_tb=False \
                                                    num_train_frames=${frames} \
                                                    save_snapshot=True \
                                                    save_video=False \
                                                    feature_dim=${feature_dim} \
                                                    agent._target_=algos.${algo}.${algo^^}Agent \
                                                    eval_type=${type} \
                                                    eval_difficulty=${difficulty} \
                                                    +aug=${aug} \
                    #                             agent.sgqn_quantile=${sgqn_quantile} \
                    #                             agent.aux_lr=${aux_lr}
                        done
                    done
                done
            done
        done
    done

# python train.py env=dmc task_name='walker_walk' seed=5 action_repeat=2 use_wandb=False use_tb=False num_train_frames=1001000 save_snapshot=True save_video=False feature_dim=50