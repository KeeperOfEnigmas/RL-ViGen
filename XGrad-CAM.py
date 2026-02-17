import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models

import hydra
import train
from torchvision import transforms

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, algo):
        self.model = model
        self.target_layers = target_layers
        self.algo = algo
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        self.gradients = []
        outputs = []
        # for name, module in self.model._modules.items():
        #     x = module(x)
        #     if name in self.target_layers:
        #         x.register_hook(self.save_gradient)
        #         outputs += [x]
        # return outputs, x
        if self.algo == "svea":
            device = next(self.model.parameters()).device
            x = x.to(device)
            obs = x / 255.0 - 0.5

            for name, module in self.model.layers._modules.items():
                obs = module(obs)
                if name in self.target_layers:
                    obs.register_hook(self.save_gradient)
                    outputs += [obs]
            
            obs = obs.view(obs.shape[0], -1)
            
            return outputs, obs
        elif self.algo == "pieg":            
            x = x.to(next(self.model.parameters()).device)
            time_step = x.shape[1] // 3
            obs = x.view(x.shape[0] * time_step, 3, x.shape[-2], x.shape[-1])

            for name, module in self.model.model._modules.items():
                obs = module(obs)
                if name in self.target_layers:
                    obs.register_hook(self.save_gradient)
                    outputs += [obs]
                if name == 'layer2':
                    break

            conv = obs.view(obs.size(0) // time_step, time_step, obs.size(1), obs.size(2), obs.size(3))
            conv_current = conv[:, 1:, :, :, :]
            conv_prev = conv_current - conv[:, :time_step - 1, :, :, :].detach()
            conv = torch.cat([conv_current, conv_prev], axis=1)
            conv = conv.view(conv.size(0), -1) 
            
            return outputs, conv


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers, algo):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layers, algo)
        self.algo = algo

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        if self.algo == "svea":
            return target_activations, output
        elif self.algo == "pieg":
            # output = output.view(output.size(0), -1)
            output = self.model.fc(output)
            output = self.model.ln(output)
            return target_activations, output


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)

    return input.repeat(1, 3, 1, 1) # Repleat the obsrevation 3 times to match the input shape of the encoder


def show_cam_on_image(img, mask, name):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(name + ".jpg", np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, target_layer_names, use_cuda, algo):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.algo = algo
        self.extractor = ModelOutputs(self.model, target_layer_names, algo)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=-1):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == -1:
            index = np.argmax(output.cpu().data.numpy())

        if index == -1:
            index = 0

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1

        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]
        
        # XGrad_CAM
        if self.algo == "svea":
            X_weights = np.sum(grads_val[0, :] * target, axis=(1, 2))
        elif self.algo == "pieg":
            X_weights = np.sum(grads_val[1, :] * target, axis=(1, 2))
        X_weights = X_weights / (np.sum(target, axis=(1, 2)) + 1e-6)

        # Grad_CAM
        if self.algo == "svea":
            weights = np.mean(grads_val, axis=(2, 3))[0, :]
        elif self.algo == "pieg":
            weights = np.mean(grads_val, axis=(2, 3))[1, :]
        
        X_cam = np.zeros(target.shape[1:], dtype=np.float32)
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
            X_cam += X_weights[i] * target[i, :, :]
        
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        X_cam = np.maximum(X_cam, 0)
        X_cam = cv2.resize(X_cam, (224, 224))
        X_cam = X_cam - np.min(X_cam)
        X_cam = X_cam / np.max(X_cam)
        return cam, X_cam


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='D:/Git/RL-ViGen/images/',
                        help='Input image path')
    parser.add_argument('--target-index', type=int, default= -1,
                        help='class of interest')
    parser.add_argument('--output-dir', type=str, default='D:/Git/RL-ViGen/images/saliency_maps/',
                        help='Output directory to save the images')
    parser.add_argument('--tasks', type=str, nargs='+', default=["walker_walk", "pendulum_swingup", "cheetah_run", "humanoid_walk"],
                        help='Task name')
    # parser.add_argument('--tasks', type=str, nargs='+', default=["Door", "Lift", "TwoArmLift"], help='Task name')
    parser.add_argument('--augmentation', type=str, nargs='+', default=["cutmix", "cutout", "no_aug", "overlay", "cropping", "window", "rotation", "flip_v", "flip_h", "convolution", "mix"],
                        help='Augmentations')
    parser.add_argument('--seed', type=str, default=1, help='Seed')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        # x: (B,3,H,W)
        # replicate frames INSIDE the model
        x = x.repeat(1, 3, 1, 1)  # → (B,9,H,W)
        return self.encoder(x)


@hydra.main(config_path='cfgs', config_name='svea_config')
def main(cfg):
    args = get_args()

    workspace = train.Workspace(cfg)
    if "svea" in cfg.agent._target_:
        algo = "svea"
        target_layer_names = ["20"]
    elif "pieg" in cfg.agent._target_:
        algo = "pieg"
        target_layer_names = ["layer2"]
    
    for aug in args.augmentation:
        for task in args.tasks:
            try:
                workspace.load_snapshot(file_path=f"D:/Git/RL-ViGen/exp_local/{algo}/{task}/{args.seed}/{aug}/snapshot.pt")
            except Exception as e:
                print(e)
                continue

            agent = workspace.agent
            agent.encoder.load_state_dict(agent.encoder.state_dict())

            grad_cam = GradCam(model=agent.encoder, \
                            target_layer_names=target_layer_names, use_cuda=True, algo=algo)

            img = cv2.imread(f"{args.image_path}{task}_highres.png", 1)
            img_highres = img.copy()
            img = np.float32(cv2.resize(img, (84, 84))) / 255
            input = preprocess_image(img)

            # If -1, returns the map for the highest scoring category.
            # Otherwise, targets the requested index.
            target_index = args.target_index
            [cam, X_cam] = grad_cam(input, target_index)

            img_highres = np.float32(img_highres) / 255
            show_cam_on_image(img_highres, cam, f'{args.output_dir}cam_{algo}_{task}_{aug}')
            show_cam_on_image(img_highres, X_cam, f'{args.output_dir}X_cam_{algo}_{task}_{aug}')


if __name__ == '__main__':
    """ python XGrad-cam.py --image-path <path_to_image> --use-cuda"""
    main()

