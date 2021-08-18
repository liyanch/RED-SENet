import os
import cv2
import argparse
import numpy as np
import pydicom

def save_dataset(args):

    dirs = [os.path.join(args.data_path, _) for _ in os.listdir(args.data_path)]
    for dir in dirs:
        fns = [os.path.join(dir, _) for _ in os.listdir(dir)]
        for fn in fns:
            img_arr = cv2.imread(fn, 0)
            if "noise" in fn:
                temp_fn = os.path.basename(fn).replace(".jpg", "_input.npy")
            else:
                temp_fn = os.path.basename(fn).replace(".jpg", "_target.npy")
            np.save(os.path.join(args.save_path, temp_fn), img_arr)


def normalize_(image, MIN_B=-1024.0, MAX_B=3072.0):
   image = (image - MIN_B) / (MAX_B - MIN_B)
   return image


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill=' '):
    # referred from https://gist.github.com/snakers4/91fa21b9dda9d055a02ecd23f24fbc3d
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '=' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    if iteration == total:
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='/zengxh_fix/liyanc/heart/REDCNN_SENet/img/L001')
    # parser.add_argument('--save_path', type=str, default='./npy_img/train')
    parser.add_argument('--save_path', type=str, default='/zengxh_fix/liyanc/heart/REDCNN_SENet/npy_img/')
    parser.add_argument('--test_patient', type=str, default='L001-test')
    parser.add_argument('--mm', type=int, default=3)
    parser.add_argument('--norm_range_min', type=float, default=0.0)
    parser.add_argument('--norm_range_max', type=float, default=255)

    args = parser.parse_args()
    save_dataset(args)
