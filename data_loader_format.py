# data loader
from __future__ import print_function, division
import glob
import torch
from skimage import io, transform, color
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import cv2
torch.cuda.manual_seed_all(1)
torch.manual_seed(1)
np.random.seed(1)
#==========================dataset load==========================

class Normalize(object):
    def __init__(self):
        pass
    
    def __call__(self, image, mask=None):
        image = image/255.0
        if mask is None:
            return image
        else:
            mask /= 255.
            return image, mask

class RandomCrop(object):
    def __call__(self, image, mask):
        H,W,_   = image.shape
        randw   = np.random.randint(W/8)
        randh   = np.random.randint(H/8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
        return image[p0:p1,p2:p3, :], mask[p0:p1,p2:p3]

class RandomFlip(object):
    def __call__(self, image, mask):
        if np.random.randint(2)==0:
            return image[:,::-1,:], mask[:, ::-1]
        else:
            return image, mask

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask=None):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if mask is None:
            return image
        else:
            mask  = cv2.resize( mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
            return image, mask

class ToTensor(object):
    def __call__(self, image, mask=None):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        if mask is None:
            return image
        else:
            mask  = torch.from_numpy(mask)
            return image, mask

class BlurdectDataset(Dataset):
    def __init__(self,img_name_list,lbl_name_list, transform=None):
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.normalize = Normalize()
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self,idx):

        image_name  = self.image_name_list[idx]
        image = cv2.imread(image_name)[:,:,::-1].astype(np.float32)
        mask_name  = self.label_name_list[idx]
        mask  = cv2.imread(mask_name, 0).astype(np.float32)
        image, mask = self.normalize(image, mask)
        image, mask = self.randomcrop(image, mask)
        image, mask = self.randomflip(image, mask)
        return image, mask

    def collate(self, batch):
        size = 320
        image, mask = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            mask[i]  = cv2.resize(mask[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(np.stack(image, axis=0)).permute(0,3,1,2)
        mask  = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
        return image, mask

class BlurdectTestDataset(Dataset):
    def __init__(self,img_name_list,lbl_name_list, transform=None):
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.normalize = Normalize()
        self.resize = Resize(320, 320)
        self.totensor = ToTensor()

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self,idx):

        image_name  = self.image_name_list[idx]
        image = cv2.imread(image_name)[:,:,::-1].astype(np.float32)
        mask_name  = self.label_name_list[idx]
        mask  = cv2.imread(mask_name, 0).astype(np.float32)
        image, mask = self.normalize(image, mask)
        image, mask = self.resize(image, mask)
        image, mask = self.totensor(image, mask)
        return image, mask
