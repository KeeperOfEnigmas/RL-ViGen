# Augmentations
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import utils

class RandomCropAug(nn.Module):
    """
    Locate a random crop in the input tensor x, than scale it back to the original size.
    
    Arguments:
        x (torch.Tensor): Input tensor of shape (n, c, h, w).
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: (n, c, h, w)
        n, c, h, w = x.size()
        assert h == w, "Input tensor must be square (height == width)"
        random_minus = random.randint(h//40, h//10)
        th, tw = (h - random_minus, w - random_minus)
        if h == th and w == tw:
            return x
        i = torch.randint(0, h - th + 1, (1,)).item()
        j = torch.randint(0, w - tw + 1, (1,)).item()
        crop = x[:, :, i:i+th, j:j+tw]
        resized = F.interpolate(crop, size=(h, w), mode='bilinear', align_corners=False)
        
        print("Class random crop!")
        return resized
    
def random_crop(x):
    """
    Locate a random crop in the input tensor x, than scale it back to the original size.
    
    Arguments:
        x (torch.Tensor): Input tensor of shape (n, c, h, w).
    """
    n, c, h, w = x.size()
    assert h==w, "Input tensor must be square (height == width)"
    random_minus = random.randint(h//40, h//10)
    th, tw = (h - random_minus, w - random_minus)
    if h == th and w == tw:
        return x
    i = torch.randint(0, h - th + 1, (1,)).item()
    j = torch.randint(0, w - tw + 1, (1,)).item()
    crop = x[:, :, i:i+th, j:j+tw]
    resized = F.interpolate(crop, size=(h, w), mode='bilinear', align_corners=False)

    # print("Simple function random crop!")
    return resized


def random_rot(x):
    """
    Randomly rotate the input tensor by -90, +90, or 180 degrees.
    Arguments:
        x (torch.Tensor): Input tensor of shape (n, c, h, w).
    """
    k = random.choice([-1, 0, 1])

    if k==0:
        # Rotate 180 degrees = rotate 90 degrees twice
        return torch.rot90(torch.rot90(x, 1, dims=[2, 3]), 1, dims=[2, 3])
    else:
        return torch.rot90(x, k, dims=[2, 3])
    

def random_flip_h(x):
    return torch.flip(x, dims=[2])  # Flip horizontally


def random_flip_v(x):
    return torch.flip(x, dims=[3])  # Flip horizontally


def random_window(x):
    crop = random_crop(x)

    _, _, hx, wx = x.size()
    _, _ ,hcrop, wcrop = crop.size()
    pad_top = (hx - hcrop) // 2
    pad_bottom = (hx - hcrop) - pad_top
    pad_right = (wx - wcrop) // 2
    pad_left = (wx - wcrop) - pad_right
    # to pad the last 2 dimensions of the input tensor, then use (padding_left,padding_right, padding_top, padding_bottom)
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    
    output = F.pad(crop, padding, 'replicate')

    # print("Simple function random window!")
    return output


class RandomOverlayAug(nn.Module):
    """
    Blend input tensor x with overlay_img using alpha.
    Both x and overlay_img should be (N, C, H, W) and in [0, 255] float.
    """
    def __init__(self, image_size, alpha=0.5):
        super().__init__()
        self.overlay_path = 'test/blend_image.jpg'
        self.alpha = alpha

        # Pre-load and preprocess the overlay image
        self.overlay_img = img_to_tensor(self.overlay_path, image_size)

    def forward(self, x):
        # x: (N, C, H, W)
        overlay_img = self.overlay_img.expand_as(x)

        print("Class random overlay!")
        return ((1 - self.alpha) * (x / 255.) + self.alpha * (overlay_img / 255.)) * 255.
    
def random_overlay(x, size, alpha=0.5):
    overlay_img = img_to_tensor('test/overlay_places365.jpg', size)

    print("Simple function random overlay!")
    return ((1 - alpha) * (x/ 255.) + alpha * (overlay_img / 255.)) * 255.


def random_conv(x):
    """Applies a random conv2d, deviates slightly from https://arxiv.org/abs/1910.05396"""
    n, c, h, w = x.size()
    for i in range(n):
        weights  = torch.randn(3, 3, 3, 3).to(x.device)
        temp_x = x[i:i+1].reshape(-1, 3, h, w)/255.
        temp_x = F.pad(temp_x, pad=[1]*4, mode='replicate')
        out = torch.sigmoid(F.conv2d(temp_x, weights))*255.
        total_out = out if i == 0 else torch.cat([total_out, out], axis=0)

    print("simple function random conv!")
    return total_out.reshape(n, c, h, w)


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


def view_input(obs, save=False, sample_idx=128):
    if isinstance(obs, torch.Tensor):
        obs = obs.detach().cpu().numpy()
    # If batch dimension exists, select the sample
    if obs.ndim == 4:
        obs = obs[sample_idx]  # shape (9, H, W)
    if isinstance(obs, np.ndarray):
        for i in range(0, obs.shape[0], 3):
            img = obs[i:i+3].transpose(1, 2, 0)
            fig = plt.figure(figsize=(8, 4))
            plt.imshow(img.astype('uint8'))
            plt.title(f"Frame stack {i//3}")
            if not save:
                plt.show() 
            elif save:
                save_unique_image(fig, "D:/Git/RL-ViGen/result/svea/miscellaneous/images", "input.png")
                plt.close(fig)


def random_color_slight(img, scale_range=(0.8, 1.2), bias_range=(-30, 30)):
    """
    Slightly adjust the color intensity of each channel in the input image.
    """
    assert img.ndim == 3 and img.shape[0] % 3 == 0, "Input must be (3*n, H, W)"
    c, h, w = img.shape
    out = np.empty_like(img)
    for i in range(0, c, 3):
        frame = img[i:i+3].astype(np.float32)  # (3, H, W)
        # Random scale and bias for each channel
        scale = np.random.uniform(*scale_range, size=(3, 1, 1))
        bias = np.random.uniform(*bias_range, size=(3, 1, 1))
        frame = frame * scale + bias
        frame = np.clip(frame, 0, 255)
        out[i:i+3] = frame.astype(img.dtype)
    return out

def random_color_intense(img, scale_range=(0.5, 1.5), bias_range=(-50, 50)):
    """
    Drastically change the color intensity of each channel in the input image.
    """
    assert img.ndim == 3 and img.shape[0] % 3 == 0, "Input must be (3*n, H, W)"
    c, h, w = img.shape
    out = np.empty_like(img)
    for i in range(0, c, 3):
        frame = img[i:i+3].astype(np.float32)  # (3, H, W)
        # Random scale and bias for each channel
        scale = np.random.uniform(*scale_range, size=(3, 1, 1))
        bias = np.random.uniform(*bias_range, size=(3, 1, 1))
        frame = frame * scale + bias
        frame = np.clip(frame, 0, 255)
        out[i:i+3] = frame.astype(img.dtype)
    return out


def saliency_map(self, obs: torch.tensor, step, save_dir=None, base_filename="svea_saliency_map.png"):
    """
    Visualize the saliency map for the input image.
    obs: torch.Tensor, shape (B, C, H, W) or (C, H, W)
    idx: int or None, index of the batch to visualize (if batch size > 1)
    """
    import matplotlib.pyplot as plt

    # Ensure obs is batched
    if obs.ndim == 3:
        obs = obs.unsqueeze(0)  # (1, C, H, W)
    obs = obs.clone().requires_grad_(True)

    # Forward pass
    # output = self.encoder(obs)
    # score, class_idx = output.max(dim=1)
    # score = score.sum()  # sum over batch if needed
    features = self.encoder(obs)
    stddev = utils.schedule(self.stddev_schedule, step) 
    dist = self.actor(features, stddev)
    action = dist.mean
    q1, q2 = self.critic(features, action)
    score = torch.min(q1, q2)[0, 0]  # scalar

    # Backward pass
    score.backward()

    # Get saliency: max over channels, abs value
    # saliency = obs.grad.abs()
    # saliency = saliency[0][:3].max(dim=0)[0].cpu().numpy()
    # img = obs[0][:3].cpu().detach().numpy().transpose(1, 2, 0) / 255.0
    saliency = obs.grad.data.abs().squeeze()[:3,:,:].max(dim=0)[0]
    saliency = saliency.cpu().data.numpy()
    img = obs.cpu().data.numpy().squeeze().transpose(1, 2, 0)[:,:,:3]/255.0

    saliency = saliency / (saliency.max() + 1e-8) # Normalize saliency map
    threshold = 0.2  # Set a threshold for visualization
    saliency = np.where(saliency >= threshold, saliency, 0)

    # Plot
    fig = plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Saliency Map")
    plt.imshow(saliency, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title("Input Image")
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()

    if save_dir is not None:
        save_unique_image(fig, save_dir, base_filename)
    else:
        plt.show()
    plt.close(fig)

def save_unique_image(fig, directory, base_filename):
    """
    Save a matplotlib figure to directory with a unique incremented filename.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    name, ext = os.path.splitext(base_filename)
    filename = base_filename
    i = 1

    while os.path.exists(os.path.join(directory, filename)):
        filename = f"{name}_{i}{ext}"
        i += 1
    fig.savefig(os.path.join(directory, filename))
    plt.close(fig)




# Testing section
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import train
import hydra



def test_augmentation(augmentation, path, simple_function=False):
    # Image size
    size = (84, 84)

    # Load image
    img_tensor = img_to_tensor(path, size)
	
    # List of augmentations
    list = {}
    list["random_shift"] = RandomShiftsAug(pad=4)
    list["random_overlay"] = RandomOverlayAug(size, alpha=0.3)
    list["random_crop"] = RandomCropAug()

    # Pass to augmentation
    if simple_function:
        if augmentation=="random_overlay":
            augmented_img = random_overlay(img_tensor, size, alpha=0.3) 
        elif augmentation=="random_crop":
            augmented_img = random_crop(img_tensor)
        elif augmentation=="random_conv":
            augmented_img = random_conv(img_tensor)
        elif augmentation=="random_window":
            augmented_img = random_window(img_tensor)
        elif augmentation=="random_rot":
            augmented_img = random_rot(img_tensor)
        elif augmentation=="random_flip_h":
            augmented_img = random_flip_h(img_tensor)
        elif augmentation=="random_flip_v":
            augmented_img = random_flip_v(img_tensor)
        else:
            raise NotImplementedError(f"Simple function {augmentation} not implemented.")
    else:
        try:
            aug = list[augmentation]
            augmented_img = aug(img_tensor)
        except KeyError:
            raise KeyError(f"{augmentation} not found in list.")

    # Prepare images for display
    orig_np = img_tensor.squeeze(0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    aug_np = augmented_img.squeeze(0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()

    # Show one image
    fig = plt.figure(figsize=(8, 4))
    plt.imshow(aug_np)
    plt.title('Augmented '+augmentation)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Show both images
    # fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    # axs[0].imshow(orig_np)
    # axs[0].set_title('Original')
    # axs[0].axis('off')
    # axs[1].imshow(aug_np)
    # axs[1].set_title('Augmented '+augmentation)
    # axs[1].axis('off')
    # plt.tight_layout()
    # plt.show()


def img_to_tensor(path, size):
    # Load image
    img = Image.open(path).convert('RGB')

    # Resize to square (e.g., 84x84)
    img = img.resize(size)

    # Convert to tensor and add batch dimension
    transform = T.ToTensor()  # Converts to [0,1] float tensor, shape (c, h, w)
    img_tensor = transform(img).unsqueeze(0)  # Shape: (1, 3, 84, 84)

    # If your model expects [0,255] range, scale accordingly
    img_tensor = img_tensor * 255.0

    return img_tensor


def test_color_change(path):
    img = stack_images(path, size=(84, 84))  # shape (9, H, W)
    changed = random_color_slight(np.array(img))
    # changed = random_color_intense(np.array(img))
    view_input(changed)


def stack_images(paths, size=(84, 84)):
    imgs = []
    for path in paths:
        img = Image.open(path).convert('RGB').resize(size)
        img_np = np.array(img)  # shape (H, W, 3)
        img_np = img_np.transpose(2, 0, 1)  # to (3, H, W)
        imgs.append(img_np)
    stacked = np.concatenate(imgs, axis=0)  # shape (9, H, W)
    return stacked


@hydra.main(config_path='cfgs', config_name='svea_config')
def load_model(cfg):
    snapshot = torch.load('D:/Git/RL-ViGen/exp_local/2025.06.03/svea/base_slight_color/snapshot.pt', map_location='cpu')
    agent_state = snapshot['agent']
    step = snapshot['_global_step']

    import wrappers.loco_wrapper as dmc
    env = dmc.make("walker_walk", frame_stack=cfg.frame_stack, action_repeat=2, seed=5)
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
    path = "D:/Git/RL-ViGen/exp_local/2025.06.03/svea/base_slight_color/buffer/20250604T100739_1000_500.npz"
    data = np.load(path)
    list = data.files

    for item in list:
        print(f"Key: {item}, Shape: {data[item].shape}")
    
    obs = data["observation"][60]
    # view_input(obs)

    return obs




if __name__ == "__main__":
    test_augmentation("random_rot", 'result/svea/miscellaneous/images/input.png', simple_function=True)
    test_augmentation("random_window", 'result/svea/miscellaneous/images/input.png', simple_function=True)
    test_augmentation("random_crop", 'result/svea/miscellaneous/images/input.png', simple_function=True)
    test_augmentation("random_flip_h", 'result/svea/miscellaneous/images/input.png', simple_function=True)
    test_augmentation("random_flip_v", 'result/svea/miscellaneous/images/input.png', simple_function=True)
    # test_color_change(['result/svea/images/original observation.png', 'result/svea/images/original observation.png', 'result/svea/images/original observation.png'])
    # agent = load_model()
    # view_input(load_obs())