import os, random
import torch.nn as nn
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from glob import glob
from datasets import MedicalImage_dataset
from utils import test_single_volume_hyper
from hyper_unet import Unet
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def inference_hyper(model, test_save_path, hyper=(0, 0)):
    img_size = 224
    db_test = MedicalImage_dataset(base_dir='/home/gyf/MyData/Synapse/test_vol_h5/', suffix_name=".npy.h5")

    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    model.eval()
    metric_list = 0.0

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):

        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]

        metric_i = test_single_volume_hyper(image, label, model, hyper, classes=classes, patch_size=[img_size, img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=1)

        metric_list += np.array(metric_i)
        print('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))

    metric_list = metric_list / len(db_test)
    for i in range(1, classes):
        print('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    print('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"


if __name__ == '__main__':
    deterministic = 1
    img_size = 224
    classes = 9
    seed = 1234
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = Unet(3, classes).cuda()

    test_save_path = None

    net.load_state_dict(torch.load("./save_model_pt/Synapse/Hyper_model/epoch_149_hyper.pth"))
    # hyper = np.zeros((512,)) + 0.8
    # hyper = torch.from_numpy(hyper).cuda()
    # hyper = hyper[None, ...]
    # hyper = hyper.float()
    for ii in np.arange(0.1, 1.1, 0.1):
        print("hyper:", ii)
        hyper = np.zeros((512,)) + ii
        hyper = torch.from_numpy(hyper).cuda()
        hyper = hyper[None, ...]
        hyper = hyper.float()

        inference_hyper(net, test_save_path, hyper)




