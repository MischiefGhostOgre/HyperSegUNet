import os, random, argparse
from utils import DiceLoss, Homoscedastic
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from datasets import MedicalImage_dataset, RandomGenerator, MedicalImage_dataset_mixup
from hyper_unet import Unet
import numpy as np
import copy

parser = argparse.ArgumentParser()
parser.add_argument("--train_data", type=str, default="/home/gyf/MyData/Synapse/train_npz/")
parser.add_argument("--dataset", type=str, default="Synapse")
parser.add_argument("--max_epoch", type=int, default=150)
parser.add_argument("--batch_size", type=int, default=12)
parser.add_argument("--base_lr", type=float, default=0.01)
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--hyper_layer_num", type=int, default=6)
parser.add_argument("--hyper_layer_size", type=int, default=256)
parser.add_argument("--save_pth_dir", type=str, default="./save_model_pth/")
parser.add_argument("--mixup", type=bool, default=False)
parser.add_argument("--homo", type=bool, default=False)
args = parser.parse_args()

base_lr = args.base_lr
batch_size = args.batch_size
train_data = args.train_data
img_size = args.img_size
dataset = args.dataset
max_epoch = args.max_epoch
hyper_layer_num = args.hyper_layer_num
hyper_layer_size = args.hyper_layer_size
save_pth_dir = args.save_pth_dir
mixup = args.mixup
homo = args.homo


def main():
    if not mixup:
        db_train = MedicalImage_dataset(base_dir=train_data, suffix_name=".npz", transform=transforms.Compose(
            [RandomGenerator(output_size=[img_size, img_size])]))
    else:
        db_train = MedicalImage_dataset_mixup(base_dir=train_data, suffix_name=".npz", transform=transforms.Compose(
            [RandomGenerator(output_size=[img_size, img_size])]))

    def worker_init_fn(worker_id):
        random.seed(1234 + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    if dataset == "Synapse":
        num_classes = 9
    elif dataset == "ACDC":
        num_classes = 4
    else:
        num_classes = 0
        assert "Unidentified datasets."

    model = Unet(3, num_classes, hyper_layer_num, hyper_layer_size)
    device = torch.device('cuda:0')
    model = model.to(device)
    model = model.cuda()
    model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(n_classes=num_classes)
    if not homo:
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    else:
        homo_loss = Homoscedastic(2)
        optimizer = optim.SGD([{"params": model.parameters()}, {"params": homo.parameters()}], lr=base_lr, momentum=0.9,
                              weight_decay=0.0001)
    iter_num = 0
    max_iterations = max_epoch * len(trainloader)
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            alpha = random.uniform(0.05, 0.95)
            beta = 1 - alpha
            hyper = np.zeros((512,)) + alpha
            hyper = torch.from_numpy(hyper).cuda()
            hyper = hyper[None, ...]
            hyper = hyper.float()

            if image_batch.size()[1] == 1:
                image_batch = image_batch.repeat(1, 3, 1, 1)
            if image_batch.size()[1] == 3:
                pass
            else:
                assert "Image channel must be 1 or 3."

            outputs = model(hyper, image_batch)
            if not homo:
                loss_dice = dice_loss(outputs, label_batch, softmax=True)
                loss_ce = ce_loss(outputs, label_batch[:].long())
                loss = alpha * loss_dice + beta * loss_ce
            else:
                loss_dice = dice_loss(outputs, label_batch, softmax=True)
                loss_ce = ce_loss(outputs, label_batch[:].long())
                lossList = [loss_dice, loss_ce]
                loss = homo_loss(torch.stack(lossList))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if mixup:
                image_batch, label_batch = sampled_batch['image_mixup'], sampled_batch['label_mixup']
                image_batch = torch.cat([image_batch, image_batch, image_batch], dim=1)
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                outputs = model(hyper, image_batch)
                if not homo:
                    loss_dice = dice_loss(outputs, label_batch, softmax=True)
                    loss_ce = ce_loss(outputs, label_batch[:].long())
                    loss = alpha * loss_dice + beta * loss_ce
                else:
                    loss_dice = dice_loss(outputs, label_batch, softmax=True)
                    loss_ce = ce_loss(outputs, label_batch[:].long())
                    lossList = [loss_dice, loss_ce]
                    loss = homo_loss(torch.stack(lossList))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                image_batch, label_batch = sampled_batch['image_mixup2'], sampled_batch['label_mixup2']
                image_batch = torch.cat([image_batch, image_batch, image_batch], dim=1)
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                outputs = model(hyper, image_batch)
                if not homo:
                    loss_dice = dice_loss(outputs, label_batch, softmax=True)
                    loss_ce = ce_loss(outputs, label_batch[:].long())
                    loss = alpha * loss_dice + beta * loss_ce
                else:
                    loss_dice = dice_loss(outputs, label_batch, softmax=True)
                    loss_ce = ce_loss(outputs, label_batch[:].long())
                    lossList = [loss_dice, loss_ce]
                    loss = homo_loss(torch.stack(lossList))
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


if __name__ == '__main__':
    main()
