import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from runx.logx import logx
import torch
import time
import importlib
import torchvision.transforms as standard_transforms
import transforms.joint_transforms as joint_transforms
from config import cfg
import argparse
import numpy as np
import cv2
import onnx
import onnx_tensorrt.backend as backend
import tensorrt as trt

TRT_LOGGER = trt.Logger()


def get_onnx_model(name):
    onnx_model = onnx.load(name)
    engine = backend.prepare(onnx_model, device='CUDA:0')
    print("Load {} succeed.".format(name))
    return engine


def get_color_mask(y, colorize_mask_fn):
    color_mask = colorize_mask_fn(y[1][0])
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
    parser.add_argument('--fp16', action='store_true', default=False)

    args = parser.parse_args()

    return args

# TODO: Support fp32, fp16, and int8.
def main():
    args = arg_parse()

    eval_size = (384, 768)

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

    model_onnx = get_onnx_model(args.model_name)

    if args.save:
        os.makedirs(args.output_path, exist_ok=True)

    for input_images, _, img_names, _ in val_set:
        x = input_images.unsqueeze(0).numpy()
        if args.fp16:
            x = x.astype(np.float16)
        start_time = time.time()
        y = model_onnx.run(x)
        used_time = time.time() - start_time
        print(img_names, input_images.shape, used_time)
        color_mask = get_color_mask(y, val_set.colorize_mask)
   
        if args.save:
            color_mask.save("{}/{}_pred_onnx.png".format(
                args.output_path, img_names))
        else:
            cv_img = np.asarray(color_mask.convert('RGB'))
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
            cv2.imshow("img", cv_img)
            cv2.waitKey(1)


if __name__ == "__main__":
    main()