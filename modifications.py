# Augmentations
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class RandomCropAug(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: (n, c, h, w)
        n, c, h, w = x.size()
        assert h == w, "Input tensor must be square (height == width)"
        random_minus = random.randint(h//10//4, h//10)
        th, tw = (h - random_minus, w - random_minus)
        if h == th and w == tw:
            return x
        i = torch.randint(0, h - th + 1, (1,)).item()
        j = torch.randint(0, w - tw + 1, (1,)).item()

        print("Class random crop!")
        return x[:, :, i:i+th, j:j+tw]
    
def random_crop(x):
    """
    Locate a random crop in the input tensor x.
    
    Arguments:
        x (torch.Tensor): Input tensor of shape (n, c, h, w).
        output_size (tuple): Desired output size (height, width).
    """
    n, c, h, w = x.size()
    assert h==w, "Input tensor must be square (height == width)"
    random_minus = random.randint(h//10//4, h//10)
    th, tw = (h - random_minus, w - random_minus)
    if h == th and w == tw:
        return x
    i = torch.randint(0, h - th + 1, (1,)).item()
    j = torch.randint(0, w - tw + 1, (1,)).item()

    print("Simple function random crop!")
    return x[:, :, i:i+th, j:j+tw]


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

    print("Simple function random window!")
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
    overlay_img = img_to_tensor('test/blend_image.jpg', size)

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




# Testing section
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import sys
import os



def test_augmentation(augmentation, simple_function=False):
    # Image size
    size = (84, 84)

    # Load image
    img_tensor = img_to_tensor('test/test_image.jpg', size)
	
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

    # Show both images
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(orig_np)
    axs[0].set_title('Original')
    axs[0].axis('off')
    axs[1].imshow(aug_np)
    axs[1].set_title('Augmented')
    axs[1].axis('off')
    plt.tight_layout()
    plt.show()


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




if __name__ == "__main__":
    test_augmentation("random_shift", simple_function=False)