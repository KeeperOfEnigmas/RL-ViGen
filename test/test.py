from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from algos.svea import RandomShiftsAug



def test_augmentation(augmentation, simple=False):
    # Image size
    size = (100, 100)

    # Load image
    img_tensor = img_to_tensor('test/test_image.jpg', size)
	
    # List of augmentations
    list = {}
    list["random_shift"] = RandomShiftsAug(pad=4)
    list["random_overlay"] = RandomOverlayAug(size, alpha=0.3)
    list["random_crop"] = RandomCropAug(output_size=(84, 84))

    # Pass to augmentation
    if simple:
        if augmentation=="random_overlay":
            augmented_img = random_overlay(img_tensor, size, alpha=0.3) 
        if augmentation=="random_crop":
            augmented_img = random_crop(img_tensor, output_size=(84, 84))
    else:
        aug = list[augmentation]
        augmented_img = aug(img_tensor)

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




# Augmentations
import torch
import torch.nn as nn

class RandomCropAug(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size # (height, width)

    def forward(self, x):
        # x: (N, C, H, W)
        n, c, h, w = x.shape
        th, tw = self.output_size
        if h == th and w == tw:
            return x
        i = torch.randint(0, h - th + 1, (1,)).item()
        j = torch.randint(0, w - tw + 1, (1,)).item()

        print("Class random crop!")
        return x[:, :, i:i+th, j:j+tw]

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
    
def random_crop(x, output_size):
    # x: (N, C, H, W)
    n, c, h, w = x.shape
    th, tw = output_size
    if h == th and w == tw:
        return x
    i = torch.randint(0, h - th + 1, (1,)).item()
    j = torch.randint(0, w - tw + 1, (1,)).item()

    print("Simple function random crop!")
    return x[:, :, i:i+th, j:j+tw]

def random_overlay(x, size, alpha=0.5):
    overlay_img = img_to_tensor('test/blend_image.jpg', size)

    print("Simple function random overlay!")
    return ((1 - alpha) * (x/ 255.) + alpha * (overlay_img / 255.)) * 255.

if __name__ == "__main__":
    test_augmentation("random_crop", simple=True)