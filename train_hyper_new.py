import os, random
from utils import DiceLoss
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from datasets import MedicalImage_dataset, RandomGenerator
from hyper_unet import Unet
import numpy as np
import copy


class Homoscedastic(torch.nn.Module):
    '''https://arxiv.homoscedasticorg/abs/1705.07115'''

    def __init__(self, n_tasks, reduction='sum'):
        super(Homoscedastic, self).__init__()
        self.n_tasks = n_tasks
        self.reduction = reduction

        self.log_vars = torch.nn.Parameter(torch.zeros(self.n_tasks))

    def forward(self, losses):
        dtype = losses.dtype
        device = losses.device

        stds = (torch.exp(self.log_vars) ** (1 / 2)).to(device).to(dtype)
        coeffs = (2 * stds * stds) ** (-1)

        multi_task_losses = coeffs * losses + torch.log(stds * stds + 1)

        if self.reduction == 'sum':
            multi_task_losses = multi_task_losses.sum()
        if self.reduction == 'mean':
            multi_task_losses = multi_task_losses.mean()

        return multi_task_losses


def trainer_synapse(model, snapshot_path):
    base_lr = 0.01
    batch_size = 12

    db_train = MedicalImage_dataset(base_dir='/home/gyf/MyData/Synapse/train_npz/',
                                    suffix_name=".npz",
                                    transform=transforms.Compose([RandomGenerator(output_size=[224, 224])]))

    def worker_init_fn(worker_id):
        random.seed(1234 + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(n_classes=num_classes)
    homo = Homoscedastic(2)
    optimizer = optim.SGD([{"params": model.parameters()}, {"params": homo.parameters()}], lr=base_lr, momentum=0.9,
                          weight_decay=0.0001)
    iter_num = 0
    max_epoch = 150
    max_iterations = max_epoch * len(trainloader)
    iterator = tqdm(range(max_epoch), ncols=70)


    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            alpha = torch.exp(homo.log_vars[0]) ** (1 / 2)
            hyper = torch.Tensor([alpha]).cuda()
            hyper = hyper[None, ...]
            hyper = hyper.float()

            if image_batch.size()[1] == 1:
                image_batch = image_batch.repeat(1, 3, 1, 1)
            outputs = model(hyper, image_batch)

            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            lossList = [loss_dice, loss_ce]
            loss = homo(torch.stack(lossList))


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num = iter_num + 1

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '_hyper.pth')
            torch.save(model.state_dict(), save_mode_path)
            iterator.close()
            break

    return "Training Finished!"


if __name__ == '__main__':
    import time

    num_classes = 9
    model = Unet(3, num_classes)
    device = torch.device('cuda:0')
    model = model.to(device)
    model = model.cuda()

    snapshot_path = "./save_model_pt/Synapse/Hyper_model/"
    trainer_synapse(model, snapshot_path)
