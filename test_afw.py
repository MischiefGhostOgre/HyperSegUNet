import os, random, argparse, sys
from tqdm import tqdm
import numpy as np
import logging

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

from datasets import MedicalImage_dataset, RandomGenerator
from utils import test_single_volume

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="unet")
parser.add_argument("--load_model", type=str, default="./save_model_pth/CHAOS/epoch_149.pth")
parser.add_argument("--test_data", type=str, default='/data2/guoyifan/CHAOS/test_vol_h5/')
parser.add_argument("--num_classes", type=int, default=5)
parser.add_argument("--test_save_path", type=str)
args = parser.parse_args()

model_name = args.model_name
load_model = args.load_model
test_data = args.test_data
num_classes = args.num_classes
test_save_path = args.test_save_path


def inference_val(model, dice_list=[], test_save_path=None):
    db_validate = MedicalImage_dataset(base_dir=test_data, suffix_name=".npy.h5")
    testloader = DataLoader(db_validate, batch_size=1, shuffle=False, num_workers=1)
    img_size = 224

    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=num_classes, patch_size=[img_size, img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=1)
        metric_list += np.array(metric_i)
        dice_list.append(np.array(metric_i)[:, 0])
        print('idx %d case %s mean_dice %f mean_hd95 %f' % (
        i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))

    metric_list = metric_list / len(db_validate)

    for i in range(1, num_classes):
        print('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))

    dice_performance = np.mean(metric_list, axis=0)[0]
    hd_performance = np.mean(metric_list, axis=0)[1]
    mse_performance = np.mean(metric_list, axis=0)[2]
    print("avg dice:, avg hd:", dice_performance, hd_performance)

    return dice_performance, hd_performance, mse_performance


def test():
    dice_list = []
    from base_networks import Unet  # Unet
    model = Unet(input_channels=3, num_classes=num_classes)

    model.load_state_dict(torch.load(load_model))
    device = torch.device('cuda:0')
    model = model.to(device)

    dice_performance, hd_performance, mse_performance = inference_val(model=model, dice_list=dice_list,
                                                                      test_save_path=test_save_path)


if __name__ == "__main__":
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    test()
