import argparse
import os
import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import (
    GradCAM, FEM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise, KPCA_CAM, ShapleyCAM,
    FinerCAM
)
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputReST
import hydra
import train


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu',
                        help='Torch device to use')
    parser.add_argument(
        '--image-path',
        type=str,
        default='D:/Git/RL-ViGen/images/',
        help='Input image path')
    parser.add_argument('--aug-smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen-smooth',
        action='store_true',
        help='Reduce noise by taking the first principle component'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=[
                            'gradcam', 'fem', 'hirescam', 'gradcam++',
                            'scorecam', 'xgradcam', 'ablationcam',
                            'eigencam', 'eigengradcam', 'layercam',
                            'fullgrad', 'gradcamelementwise', 'kpcacam', 'shapleycam',
                            'finercam'
                        ],
                        help='CAM method')
    parser.add_argument('--output-dir', type=str, default='D:/Git/RL-ViGen/images/saliency_output/',
                        help='Output directory to save the images')
    parser.add_argument('--tasks', type=list, default=["walker_walk","pendulum_swingup", "cheetah_run", "humanoid_walk"],
                        help='Task name')
    # parser.add_argument('--tasks', type=list, default=["Door", "Lift", "TwoArmLift"], help='Task name')
    parser.add_argument('--augmentation', type=list, default=["cutmix", "cutout", "no_aug", "overlay", "cropping", "window", "rotation", "flip_v", "flip_h", "convolution", "mix"],
                        help='Augmentations')
    args = parser.parse_args()
    
    if args.device:
        print(f'Using device "{args.device}" for acceleration')
    else:
        print('Using CPU for computation')

    return args

class EncoderCAMWrapper(torch.nn.Module):
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
    methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "fem": FEM,
        "gradcamelementwise": GradCAMElementWise,
        'kpcacam': KPCA_CAM,
        'shapleycam': ShapleyCAM,
        'finercam': FinerCAM
    }

    workspace = train.Workspace(cfg)
    if "svea" in cfg.agent._target_:
        algo = "svea"
    elif "pieg" in cfg.agent._target_:
        algo = "pieg"
    
    for aug in args.augmentation:
        for task in args.tasks:
            try:
                workspace.load_snapshot(file_path=f"D:/Git/RL-ViGen/exp_local/{algo}/{task}/2/{aug}/snapshot.pt")
            except Exception as e:
                print(e)
                continue
            agent = workspace.agent

            # Load the state dicts for each component
            agent.encoder.load_state_dict(agent.encoder.state_dict())
            # agent.actor.load_state_dict(agent.actor.state_dict())
            # agent.critic.load_state_dict(agent.critic.state_dict())
            # agent.critic_target.load_state_dict(agent.critic_target.state_dict())

            # print("agent: " + str(agent))
            
            model = EncoderCAMWrapper(agent.encoder).to(args.device)
            model.eval().requires_grad_(True)

            if algo=="svea":
                target_layers = [agent.encoder.layers]
            elif algo=="pieg":
                target_layers = [agent.encoder.model.layer2]
            # target_layers = [agent.encoder.layers]
            # print("Encoder: " + str(agent.encoder.model))
            # print("layer2: " + str(agent.encoder.model.layer2))
            # print("Target layer: " + str(target_layers))

            rgb_img = cv2.imread(f"{args.image_path}input_{task}.png", 1)[:, :, ::-1]
            rgb_img = np.float32(rgb_img) / 255
            rgb_img = cv2.resize(rgb_img, (84, 84))
            input_tensor = preprocess_image(rgb_img,
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]).to(args.device)
            input_tensor.requires_grad_(True)
            print("Input tensor shape: ", input_tensor.shape) # (1, 3, 84, 84)
            # input_tensor = input_tensor.repeat(1, 3, 1, 1)
            # print("Input shape: ", input_tensor.shape) # (1, 9, 84, 84)
            print("Input tensor length: ",len(input_tensor.shape)) # 4
            print("CAM model output shape:", model(input_tensor).shape) # (1, 1024)
            print("Output type: ", type(model(input_tensor))) # <class 'torch.Tensor'>

            # We have to specify the target we want to generate
            # the Class Activation Maps for.
            # If targets is None, the highest scoring category (for every member in the batch) will be used.
            # You can target specific categories by
            # targets = [ClassifierOutputTarget(281)]
            # targets = [ClassifierOutputReST(281)]
            targets = None

            # Using the with statement ensures the context is freed, and you can
            # recreate different CAM objects in a loop.
            # cam_algorithm = methods[args.method]
            cam_algorithm = methods[args.method]
            with cam_algorithm(model=model,
                            target_layers=target_layers) as cam:

                # AblationCAM and ScoreCAM have batched implementations.
                # You can override the internal batch size for faster computation.
                cam.batch_size = 32
                grayscale_cam = cam(input_tensor=input_tensor,
                                    targets=targets,
                                    aug_smooth=args.aug_smooth,
                                    eigen_smooth=args.eigen_smooth)

                grayscale_cam = grayscale_cam[0, :]

                cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

            gb_model = GuidedBackpropReLUModel(model=model, device=args.device)
            gb = gb_model(input_tensor, target_category=None)

            cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
            cam_gb = deprocess_image(cam_mask * gb)
            gb = deprocess_image(gb)

            os.makedirs(args.output_dir, exist_ok=True)

            cam_output_path = os.path.join(args.output_dir, f'{args.method}_{algo}_{task}_{aug}_cam.jpg')
            gb_output_path = os.path.join(args.output_dir, f'{args.method}_{algo}_{task}_{aug}_gb.jpg')
            cam_gb_output_path = os.path.join(args.output_dir, f'{args.method}_{algo}_{task}_{aug}_cam_gb.jpg')

            cv2.imwrite(cam_output_path, cam_image)
            cv2.imwrite(gb_output_path, gb)
            cv2.imwrite(cam_gb_output_path, cam_gb)



if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """
    main()