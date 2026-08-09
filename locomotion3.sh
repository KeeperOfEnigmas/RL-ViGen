# export MUJOCO_GL=glfw
easy_task_list=('cheetah_run')
seed_list=(2)
algorithm=("pieg")
augmentation=("overlay" "mix")
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
            done
        done
    done
done