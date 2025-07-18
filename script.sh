easy_task_list=('walker_walk')
frames=1001000
feature_dim=50
sgqn_quantile=0.93
action_repeat=2
aux_lr=8e-5
env=dmc

for task_name in ${easy_task_list[@]};
do
    CUDA_VISIBLE_DEVICES=0  python train.py \
                                env=${env} \
                                task_name=${task_name} \
                                seed=5 \
                                action_repeat=${action_repeat} \
                                use_wandb=False \
                                use_tb=False \
                                num_train_frames=${frames} \
                                save_snapshot=True \
                                save_video=False \
                                feature_dim=${feature_dim}
#                             agent.sgqn_quantile=${sgqn_quantile} \
#                             agent.aux_lr=${aux_lr}
done

# python train.py env=dmc task_name='walker_walk' seed=5 action_repeat=2 use_wandb=False use_tb=False num_train_frames=1001000 save_snapshot=True save_video=False feature_dim=50