import os, random, argparse, sys
from tqdm import tqdm
import numpy as np
import logging

from torch.utils.data import DataLoader
import torch
from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from datasets import MedicalImage_dataset, RandomGenerator
from utils import DiceLoss_weights

parser = argparse.ArgumentParser()
parser.add_argument("--train_data", type=str, default="/data2/guoyifan/CHAOS/train_npz/")
parser.add_argument("--dataset", type=str, default="CHAOS")
parser.add_argument("--max_epoch", type=int, default=150)
parser.add_argument("--batch_size", type=int, default=24)
parser.add_argument("--base_lr", type=float, default=0.01)
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--save_pth_dir", type=str, default="./save_model_pt/CHAOS/")
args = parser.parse_args()

base_lr = args.base_lr
batch_size = args.batch_size
train_data = args.train_data
img_size = args.img_size
dataset = args.dataset
max_epoch = args.max_epoch
save_pth_dir = args.save_pth_dir
model_name = args.model_name

if dataset == "Synapse":
    num_classes = 9
elif dataset == "CHAOS":
    num_classes = 5
elif dataset == "learn2reg2021":
    num_classes = 5
elif dataset == "ACT1K":
    num_classes = 5
else:
    pass


def main():
    db_train = MedicalImage_dataset(base_dir=train_data, suffix_name=".npz", transform=transforms.Compose(
        [RandomGenerator(output_size=[img_size, img_size])]))

    def worker_init_fn(worker_id):
        random.seed(1234 + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    
    from base_networks import Unet  # Unet
    model = Unet(input_channels=3, num_classes=num_classes)

    device = torch.device('cuda:0')
    model = model.to(device)
    model.train()

    dice_loss = DiceLoss_weights(num_classes)
    ce_loss = CrossEntropyLoss()

    optimizer = optim.SGD([{"params": model.parameters()}, {"params": dice_loss.parameters()}], lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    iter_num = 0
    max_iterations = max_epoch * len(trainloader)
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        '''
        Training
        '''
        for i_batch, sample_batch in enumerate(trainloader):

            image_batch, label_batch = sample_batch["image"], sample_batch["label"]
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            image_batch = image_batch.cuda()
            label_batch = label_batch.cuda()

            if image_batch.size()[1] == 1:
                image_batch = image_batch.repeat(1, 3, 1, 1)
            elif image_batch.size()[1] == 3:
                pass
            else:
                pass

            outputs = model(image_batch)
            ce_loss_val = ce_loss(outputs, label_batch[:].long())
            dice_loss_val = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * ce_loss_val + 0.5 * dice_loss_val

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num += 1

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(save_pth_dir, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            iterator.close()
            break


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
    main()
