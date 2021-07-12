import numpy as np
import argparse
from glob import glob
import os
import cv2
import tqdm


def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser(
        description='Select Samples for Annotation')
    parser.add_argument('--mode', type=str, default="folder")
    parser.add_argument('--image_extensions', type=str, default=".jpg")
    parser.add_argument('--prob_extensions', type=str, default="_prob.png")
    parser.add_argument('--prob_path',
                        type=str,
                        default="",
                        help="path for output prob image.")
    parser.add_argument('--image_path',
                        type=str,
                        default="",
                        help="path for output prob image.")
    parser.add_argument('--selected_num',
                        type=int,
                        default=10,
                        help="How many images that we wanna select.")
    parser.add_argument('--prob',
                        type=float,
                        default=0.9,
                        help="Probability value")
    parser.add_argument('--output_path',
                        type=str,
                        default=None,
                        help="Output_path for selected images.")

    return parser.parse_args()


def cal_lower_prob_ratio(preds, prob_threshold=0.7):
    masks = np.where(preds < prob_threshold)
    ratio = masks[0].size / preds.size
    return ratio


def main():
    args = parse_args()
    assert os.path.exists(args.image_path)
    if args.mode == "folder":
        os.path.exists(args.prob_path)
    else:
        raise ValueError("Unsupported mode.")

    hist = []
    prob_filenames = []
    image_names = sorted(
        glob(os.path.join(args.image_path, '*' + args.image_extensions)))
    for image_fn in tqdm.tqdm(image_names):
        if args.mode == "folder":
            filename = os.path.basename(image_fn)
            prob_filename = filename.replace(args.image_extensions,
                                             args.prob_extensions)
            prob_filename = os.path.join(args.prob_path, prob_filename)
            prob_image = cv2.imread(prob_filename)
            prob_filenames.append(prob_filename)
            ratio = cal_lower_prob_ratio(prob_image, args.prob * 255)
            hist.append(ratio)

    selected_num = args.selected_num
    if len(hist) < selected_num:
        selected_num = len(hist)
    hist_arr = np.asarray(hist)
    ind = list(np.argpartition(hist_arr, -selected_num)[-selected_num:])
    print('prob_threshold={} ratio_bound={},{} filter_num={}'.format(
        args.prob, hist[ind[-1]], hist[ind[0]], selected_num))

    for idx in ind[:selected_num]:
        if args.output_path is not None:
            os.system("cp {} {}".format(image_names[idx], args.output_path))
            os.system("cp {} {}".format(prob_filenames[idx], args.output_path))
            print("Copy: {}".format(image_names[idx]))
        else:
            print(image_names[idx])


if __name__ == '__main__':
    main()
