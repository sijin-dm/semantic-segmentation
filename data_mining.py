import os
import network
import argparse
import importlib
import torch
import numpy as np
from PIL import Image
import cv2
import torchvision.transforms as standard_transforms
import torch.nn as nn
import glob

from runx.logx import logx
from loss.optimizer import restore_net

from config import cfg, update_dataset_cfg
import h5py


class DataMining(object):
    def __init__(self) -> None:
        super().__init__()


def parse_args():

    # Argument Parser
    parser = argparse.ArgumentParser(description='Semantic Segmentation')
    parser.add_argument('--arch',
                        type=str,
                        default='ocrnet_trt.HRNet_Mscale',
                        help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                        and deepWV3Plus (backbone: WideResNet38).')
    parser.add_argument('--dataset', type=str, default='mapillary', help='cityscapes, mapillary, camvid, kitti')
    parser.add_argument(
        '--snapshot',
        type=str,
        default=
        '/mnt/nas/share-map/experiment/sijin/01code/semantic-segmentation/logs/train_mapillary_v100/hrnet_ocr_mscale/2021.07.02_18.09/best_checkpoint_ep186.pth'
    )
    parser.add_argument('--image_folder', type=str, default='/mnt/nas/share-map/experiment/sijin/05data/mobili_images')
    parser.add_argument('--output_folder',
                        type=str,
                        default='/mnt/nas/share-map/experiment/sijin/05data/mobili_images/mined_data_800')
    parser.add_argument('--prob_threshold', type=float, default=0.8)
    parser.add_argument('--image_num', type=int, default=800)
    parser.add_argument('--mc_dropout_itr', type=int, default=10)
    parser.add_argument('--image_prefix', type=str, default='.jpg')
    args = parser.parse_args()
    return args


class FullModel(nn.Module):
    def __init__(self, model, mc_dropout_itr=None):
        super(FullModel, self).__init__()
        self.model = model
        self.mc_dropout_itr = mc_dropout_itr

    def forward(self, inputs):
        if self.mc_dropout_itr is not None:
            x = [self.model(inputs) for _ in range(self.mc_dropout_itr)]
            x = torch.stack(x).mean(dim=0)
        else:
            x = self.model(inputs)
        x = torch.nn.functional.softmax(x, dim=1)
        prob_mask, predictions = torch.max(x, dim=1)
        return prob_mask, predictions


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for name, m in model.named_modules():
        if 'dropout' in m.__class__.__name__.lower():
            print(name, m)
            m.train()


def save_pred(pred, dataset_cls, out_name="result", dataset='mapillary'):
    colorize_mask_fn = dataset_cls.colorize_mask
    prob, mask = pred
    if len(mask.shape) == 4:
        mask = mask.squeeze(0).squeeze(0)
    elif len(mask.shape) == 3:
        mask = mask.squeeze(0)
    color_mask = colorize_mask_fn(mask)
    color_mask.save(out_name + '_mask.png')
    prob = (prob * 255).astype(np.uint8)
    heat_mat = cv2.applyColorMap(prob, cv2.COLORMAP_JET)
    cv2.imwrite(out_name + '_prob.png', prob)
    cv2.imwrite(out_name + '_heat.png', heat_mat)
    # logx.msg("Saving prediction to {}".format(out_name))


def get_lower_prob_ratio(preds, prob_threshold=0.7):
    masks = np.where(preds < prob_threshold)
    ratio = masks[0].size / preds.size
    return ratio


def main():
    args = parse_args()

    # Preparing for basic setting.
    logx.initialize(logdir="logs", tensorboard=False, hparams=vars(args), global_rank=0)
    mod = importlib.import_module('datasets.{}'.format(args.dataset))
    dataset_cls = getattr(mod, 'Loader')
    update_dataset_cfg(num_classes=dataset_cls.num_classes, ignore_label=dataset_cls.ignore_label)

    net = network.get_model(network='network.' + args.arch, num_classes=cfg.DATASET.NUM_CLASSES, criterion=None)
    assert os.path.exists(args.snapshot)
    checkpoint = torch.load(args.snapshot, map_location=torch.device('cpu'))
    msg = "Loading weights from: checkpoint={}".format(args.snapshot)
    logx.msg(msg)
    restore_net(net, checkpoint)

    # From image to tensor.
    mean_std = (cfg.DATASET.MEAN, cfg.DATASET.STD)
    input_transform = standard_transforms.Compose(
        [standard_transforms.ToTensor(), standard_transforms.Normalize(*mean_std)])

    # Concat net with argmax.
    net = FullModel(net, mc_dropout_itr=args.mc_dropout_itr)
    net = net.cuda().eval()
    if args.mc_dropout_itr is not None:
        enable_dropout(net)

    image_shape = (1280, 720)
    image_num = 0
    for image_subfolder in os.listdir(args.image_folder):
        if image_subfolder.startswith('images') or image_subfolder.startswith('2021'):
            image_num += len(list(glob.glob(os.path.join(args.image_folder, image_subfolder, '*'+args.image_prefix))))
    print('Total image number: %d '%image_num)
    assert image_num != 0
    # Init hdf5.
    hdf5_name = "tmp.hdf5"
    hdf5_handle = h5py.File(hdf5_name, "w")
    prob_cache = hdf5_handle.create_dataset("prob", (image_num, image_shape[1], image_shape[0]), 'f')
    mask_cache = hdf5_handle.create_dataset("mask", (image_num, image_shape[1], image_shape[0]), 'f')

    image_list = []
    prob_hist_list = []
    cnt = 0
    with torch.no_grad():
        for image_subfolder in os.listdir(args.image_folder):
            print(image_subfolder)
            if not(image_subfolder.startswith('images') or image_subfolder.startswith('2021')):
                continue
            # mod = importlib.import_module('datasets.{}'.format(args.dataset))
            for image_file in glob.glob(os.path.join(args.image_folder, image_subfolder, '*'+args.image_prefix)):
                print(image_file)
                img = Image.open(image_file).convert('RGB')
                img = img.resize(image_shape)
                assert img is not None
                x = input_transform(img).cuda().unsqueeze(0)
                y = net(x)
                y_cpu = (y[0].squeeze(0).cpu().numpy(), y[1].squeeze(0).cpu().numpy())
                prob_cache[cnt, ...] = y_cpu[0]
                mask_cache[cnt, ...] = y_cpu[1]
                image_list.append(image_file)
                prob_hist_list.append(get_lower_prob_ratio(y_cpu[0], args.prob_threshold))
                cnt +=1 
            #     if len(prob_hist_list) == 10:
            #         break
            # if len(prob_hist_list) == 10:
            #     break

    image_num = args.image_num if len(prob_hist_list) > args.image_num else len(prob_hist_list)
    hist_arr = np.asarray(prob_hist_list)
    indexes = list(np.argpartition(hist_arr, -image_num)[-image_num:])
    os.makedirs(args.output_folder, exist_ok=True)
    # Write mining result analysis.
    output_file = os.path.join(args.output_folder, 'result.txt')
    # output_file = 'result.txt'  # debug.
    with open(output_file, "w") as wf:
        wf.write('MC Dropout iteration: %d \n' % args.mc_dropout_itr)
        wf.write('Mining probability threshold: %f \n' % args.prob_threshold)
        wf.write('Minimum ratio: %f Maximum ratio: %f \n' % (hist_arr[indexes[0]], hist_arr[indexes[-1]]))
        wf.write('=' * 100 + '\n')
        for index in indexes:
            wf.write(image_list[index] + '\n')

    # Write mining image and mask image.
    mod = importlib.import_module('datasets.{}'.format(args.dataset))
    dataset_cls = getattr(mod, 'Loader')(mode='folder', eval_folder="imgs/test_imgs")
    for index in indexes:
        image_file = image_list[index]
        os.system('cp %s %s' % (image_file, args.output_folder))
        image_name = os.path.basename(image_file)
        save_pred((prob_cache[index, ...], mask_cache[index, ...]),
                  dataset_cls,
                  out_name=os.path.join(args.output_folder, image_name[:-4]))
    print("Done")
    os.system("rm %s" % hdf5_name)


if __name__ == "__main__":
    main()
