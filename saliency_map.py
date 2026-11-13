import torch
import numpy as np
import utils
from PIL import Image
import matplotlib.pyplot as plt
import train
import hydra



def saliency_map(self, obs: torch.tensor, step, save_dir=None, base_filename="saliency_map.png"):
    if obs.ndim == 3:
        obs = obs.unsqueeze(0)  # shape: (1, C, H, W), (1, 9, 84, 84)

    obs = obs.clone().detach().requires_grad_(True)

    # Forward pass
    features = self.encoder(obs)

    stddev = utils.schedule(self.stddev_schedule, step)
    dist = self.actor(features, stddev)
    # action = dist.mean.detach()
    action = dist.sample(clip=self.stddev_clip)

    q1, q2 = self.critic(features, action)
    # score = torch.min(q1, q2).mean()
    score = torch.min(q1, q2)

    # Backward pass
    # self.encoder.zero_grad()
    # self.critic.zero_grad()
    score.backward()

    # Compute saliency
    saliency = obs.grad.detach().abs()

    # Plot
    # Take max over RGB channels of first frame
    saliency = saliency[0, :3].max(dim=0)[0]

    saliency = saliency.detach().cpu().numpy()
    # threshold = 0.2  # Set a threshold for visualization
    # saliency = np.where(saliency >= threshold, saliency, 0)

    # obs is (1, C, H, W). take first batch then first 3 channels
    print("obs shape: ", obs.shape)
    obs = obs[0][:3].detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    print("obs shape: ", obs.shape)

    fig = plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Saliency Map")
    plt.imshow(saliency, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title("Input Image")
    plt.imshow(obs)
    plt.axis('off')
    plt.tight_layout()

    plt.show()
    # if save_dir is not None:
    #     fig.savefig(os.path.join(save_dir, base_filename))
    # else:
    #     plt.show()
    # plt.close(fig)


@hydra.main(config_path='cfgs', config_name='svea_config')
# @hydra.main(config_path='cfgs', config_name='pieg_config')
def load_model(cfg):
    snapshot = torch.load('D:/Git/RL-ViGen/exp_local/svea/walker_walk/1/overlay/snapshot.pt', map_location='cpu')
    # snapshot = torch.load('D:/Git/RL-ViGen/exp_local/pieg/cheetah_run/1/overlay/snapshot.pt', map_location='cpu')
    agent_state = snapshot['agent']
    step = snapshot['_global_step']

    import wrappers.loco_wrapper as dmc
    env = dmc.make("walker_walk", frame_stack=cfg.frame_stack, action_repeat=2, seed=1)
    # env = dmc.make("cheetah_run", frame_stack=cfg.frame_stack, action_repeat=2, seed=1)
    agent = train.make_agent(env.observation_spec(), env.action_spec(), cfg.agent)

    # Load the state dicts for each component
    agent.encoder.load_state_dict(agent_state.encoder.state_dict())
    agent.actor.load_state_dict(agent_state.actor.state_dict())
    agent.critic.load_state_dict(agent_state.critic.state_dict())
    agent.critic_target.load_state_dict(agent_state.critic_target.state_dict())

    obs = load_obs()
    saliency_map(agent, torch.as_tensor(obs, device=agent.device).float(), step, save_dir="D:/Git/RL-ViGen/result/svea/saliency_map")

    return agent


def load_obs():
    path = "D:/Git/RL-ViGen/result/svea/miscellaneous/images/input.png"
    img = Image.open(path).convert("RGB").resize((84, 84))
    print(img.size)
    img = np.array(img).transpose(2, 0, 1)
    print(img.shape)
    img = np.repeat(img, repeats=3, axis=0)
    print(img.shape)

    return img


if __name__ == "__main__":
    load_model()
    print("test")