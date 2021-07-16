import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from runx.logx import logx
import torch
from torch2trt import TRTModule
import time
import importlib
import torchvision.transforms as standard_transforms
import transforms.joint_transforms as joint_transforms
from config import cfg
import argparse
import numpy as np
import cv2
import tensorrt as trt


def get_trt_model(name):
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(name))
    return model_trt


def get_color_mask(y, colorize_mask_fn):
    # prob_mask, predictions = y.data.max(1)
    prob_mask, predictions = y
    predictions = predictions.cpu().squeeze(0).squeeze(0).numpy()
    color_mask = colorize_mask_fn(predictions)
    return color_mask


def arg_parse():
    # Argument Parser
    parser = argparse.ArgumentParser(
        description='Semantic Segmentation TensorRT Inference')
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--model_name',
                        type=str,
                        default='ocrnet_trt.DDRNet23_Slim_trt.pth')
    parser.add_argument('--img_folder', type=str, default='imgs/test_imgs')
    parser.add_argument('--output_path', type=str, default="output")
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()

    # eval_size = (384, 768)
    eval_size = (720, 1280)

    # Update dataset version.
    cfg.immutable(False)
    cfg.DATASET.MAPILLARY_VERSION = "v2.0_dm"
    cfg.immutable(True)

    logx.initialize(logdir="/tmp", tensorboard=False, global_rank=0)
    mod = importlib.import_module('datasets.{}'.format("mapillary"))
    dataset_cls = getattr(mod, 'Loader')
    val_joint_transform_list = [
        joint_transforms.Scale(eval_size[1]),
        joint_transforms.CenterCrop(eval_size)
    ]

    val_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225])
    ])
    val_set = dataset_cls(mode='folder',
                          joint_transform_list=val_joint_transform_list,
                          img_transform=val_input_transform,
                          eval_folder=args.img_folder)

    model_trt = get_trt_model(args.model_name)

    if args.save:
        os.makedirs(args.output_path, exist_ok=True)

    for input_images, _, img_names, _ in val_set:
        x = input_images.cuda().unsqueeze(0)
        start_time = time.time()
        y_trt = model_trt(x)
        color_mask = get_color_mask(y_trt, val_set.colorize_mask)
        used_time = time.time() - start_time
        print("Inference time: ", used_time)
        if args.save:
            color_mask.save("{}/{}_pred_trt.png".format(
                args.output_path, img_names))
        else:
            cv_img = np.asarray(color_mask.convert('RGB'))
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
            cv2.imshow("img", cv_img)
            cv2.waitKey(1)


if __name__ == "__main__":
    main()