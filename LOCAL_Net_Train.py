import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import pytorch_iou
import pytorch_ssim
from data_loader_format import BlurdectDataset
from model import LOCAL_Net

# multiple cuda
device = torch.device('cuda')
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)
mse_loss = nn.MSELoss(size_average=True)
L1_loss = nn.L1Loss(size_average=True)
comp_loss = nn.L1Loss(size_average=True)

def bce_ssim_loss(pred,target):

    bce_out = bce_loss(pred,target)
    ssim_out = 1 - ssim_loss(pred,target)
    iou_out = iou_loss(pred,target)

    loss = bce_out + ssim_out + iou_out

    return loss


# ------- 2. set the directory of training dataset --------
parser = argparse.ArgumentParser()
parser.add_argument('--milestones', nargs='+', type=int, default=[100])
parser.add_argument('--epoch_num', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=4)
#parser.add_argument('--lambda_param', type=float, default=1.5, help='parameter for fine-grain')
#parser.add_argument('--eta_param', type=float, default=1, help='parameter for complementary')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate for model')
parser.add_argument('--data_dir', type=str, default='/Extra/dataset/saliency/train/', help='data path')
parser.add_argument('--tra_image_dir', type=str, default='blur_image_0/', help='real image path')
parser.add_argument('--tra_label_dir', type=str, default='gt_new/', help='real label path')
parser.add_argument('--model_dir', type=str, default='./saved_models/', help='saved models path')
config = parser.parse_args()


data_dir = config.data_dir
tra_image_dir = config.tra_image_dir
tra_label_dir = config.tra_label_dir
model_dir = config.model_dir
#lambda_param = config.lambda_param
#eta_param = config.eta_param

image_ext = '.jpg'
label_ext = '.png'

epoch_num = config.epoch_num
batch_size_train = config.batch_size
train_num = 0


tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*')

tra_lbl_name_list = []
for img_path in tra_img_name_list:
    img_name = img_path.split("blur_image_0/")[-1]

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    if len(imidx) == 16:
        lab_name = imidx + ".png"
    else:
        lab_name = imidx + ".bmp"

    tra_lbl_name_list.append(data_dir + tra_label_dir + lab_name)


print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("---")

train_num = len(tra_img_name_list)

blurdect_dataset = BlurdectDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list)
blurdect_dataloader = DataLoader(blurdect_dataset, collate_fn=blurdect_dataset.collate, batch_size=batch_size_train, shuffle=True, num_workers=1)

# ------- 3. define model --------

net = torch.nn.DataParallel(LOCAL_Net())


if torch.cuda.is_available():
    net.to(device)
    bce_loss.to(device)
    ssim_loss.to(device)
    iou_loss.to(device)
    comp_loss.to(device)
    L1_loss.to(device)
    mse_loss.to(device)

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=0.1)

# ------- 5. training process --------
print("---start training...")
ite_num = 0

ite_num4val = 0


for epoch in range(0, epoch_num):
    running_loss = 0.0
    net.train()

    for i, (inputs, labels) in enumerate(blurdect_dataloader):
        inputs = inputs.type(torch.FloatTensor)

        labels = labels.type(torch.FloatTensor)
        batch_size = inputs.shape[0]


        # wrap them in Variable
        if torch.cuda.is_available():
            inputs = Variable(inputs.to(device), requires_grad=False)
            labels = labels.to(device)
        else:
            inputs = Variable(inputs, requires_grad=False)

        im_sz = int(labels.shape[2])

        labels16 = F.interpolate(labels, [int(im_sz/16), int(im_sz/16)], mode='bilinear')
        labels8 = F.interpolate(labels, [int(im_sz / 8), int(im_sz / 8)], mode='bilinear')
        labels4 = F.interpolate(labels, [int(im_sz / 4), int(im_sz / 4)], mode='bilinear')
        labels2 = F.interpolate(labels, [int(im_sz / 2), int(im_sz / 2)], mode='bilinear')

        ones = torch.ones(labels.shape).float().cuda()
        a = torch.ones(8, 1, 20, 20)
        ones2 = torch.ones(batch_size, 1, int(im_sz / 2), int(im_sz / 2)).float().cuda()
        ones4 = torch.ones(batch_size, 1, int(im_sz / 4), int(im_sz / 4)).float().cuda()
        ones8 = torch.ones(batch_size, 1, int(im_sz / 8), int(im_sz / 8)).float().cuda()
        ones16 = torch.ones(batch_size, 1, int(im_sz / 16), int(im_sz / 16)).float().cuda()
        
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize

        out1, out2, out3, out4, out5 = net(inputs)
        loss1 = bce_ssim_loss(out1, torch.cat((labels, 1 - labels), dim=1))+ 1 * L1_loss(torch.sum(out1, dim=1).unsqueeze(dim=1), ones)
        loss2 = bce_ssim_loss(out2, torch.cat((labels2, 1 - labels2), dim=1))+ 1 * L1_loss(torch.sum(out2, dim=1).unsqueeze(dim=1), ones2)
        loss3 = bce_ssim_loss(out3, torch.cat((labels4, 1 - labels4), dim=1))+ 1 * L1_loss(torch.sum(out3, dim=1).unsqueeze(dim=1), ones4)
        loss4 = bce_ssim_loss(out4, torch.cat((labels8, 1 - labels8), dim=1))+ 1 * L1_loss(torch.sum(out4, dim=1).unsqueeze(dim=1), ones8)
        loss5 = bce_ssim_loss(out5, torch.cat((labels16, 1 - labels16), dim=1))+ 1 * L1_loss(torch.sum(out5, dim=1).unsqueeze(dim=1), ones16)
        loss = loss1 + 0.5 * (loss2 + loss3 + loss4 + loss5)

        loss.backward()
        optimizer.step()
        lr_scheduler.step(epoch)

        # # print statistics
        running_loss += loss.data

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] loss1: %3f, loss2: %3f, loss3: %3f, loss4: %3f, loss5: %3f," % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, loss1.data, loss2.data, loss3.data, loss4.data, loss5.data))

        # del temporary outputs and loss


    print("[epoch: %3d/%3d, batch: %5d] loss_epoch: %3f," % (epoch + 1, epoch_num, i, running_loss/i))
    if (epoch+1) % 10 == 0:

        torch.save(net.state_dict(), model_dir + "LOCAL_Net_%d.pth" % (epoch+1))
        running_loss = 0.0
        net.train()  # resume train
        ite_num4val = 0

print('-------------Congratulations! Training Done!!!-------------')
