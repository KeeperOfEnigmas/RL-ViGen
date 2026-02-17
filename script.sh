# export MUJOCO_GL=glfw
# easy_task_list=('pendulum_swingup' 'cheetah_run' 'humanoid_walk' 'Door' 'Lift')
easy_task_list=('pendulum_swingup')
# easy_task_list=('Door')
# seed_list=(3)
seed_list=(4)
# seed_list=(2 3 4 5)
# algorithm=("svea" "pieg")
algorithm=("pieg")
# eval_type=("video" "color")
# eval_difficulty=("hard" "easy")
augmentation=("no_aug") 
# augmentation=("window" "flip_v" "flip_h") 
# augmentation=("cutmix" "cutout" "no_aug" "overlay" "cropping" "window" "rotation" "flip_v" "flip_h" "convolution" "mix") 
frames=1001000
feature_dim=50
sgqn_quantile=0.93
action_repeat=2
aux_lr=8e-5
env=dmc
# env=robosuite

for algo in ${algorithm[@]}; do
    for task_name in ${easy_task_list[@]}; do
        for seed in ${seed_list[@]}; do
            for aug in ${augmentation[@]}; do
                # for type in ${eval_type[@]}; do
                    # for difficulty in ${eval_difficulty[@]}; do
                        CUDA_VISIBLE_DEVICES=0  python train.py \
                                                    env=${env} \
                                                    task_name=${task_name} \
                                                    seed=${seed} \
                                                    action_repeat=${action_repeat} \
                                                    num_train_frames=${frames} \
                                                    feature_dim=${feature_dim} \
                                                    agent._target_=algos.${algo}.${algo^^}Agent \
                                                    +aug=${aug} \
                                                    # use_wandb=False \
                                                    # use_tb=False \
                                                    # save_snapshot=True \
                                                    # save_video=False \
                                                    # eval_type=${type} \
                                                    # eval_difficulty=${difficulty} \
                    #                             agent.sgqn_quantile=${sgqn_quantile} \
                    #                             agent.aux_lr=${aux_lr}
                    # done
                # done
            done
        done
    done
done

# python train.py env=dmc task_name='walker_walk' seed=5 action_repeat=2 use_wandb=False use_tb=False num_train_frames=1001000 save_snapshot=True save_video=False feature_dim=50