export MUJOCO_GL=glfw


python train.py env=robosuite task_name=Door seed=3 action_repeat=1 num_train_frames=501500 feature_dim=50 agent._target_=algos.pieg.PIEGAgent +aug=cropping hydra.run.dir="D:/Git/RL-ViGen/exp_local/cropping"