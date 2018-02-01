# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 11:58:07 2017

@author: Biagio Brattoli
"""
import os, numpy as np
from time import time
from tqdm import tqdm
import torch.utils.data as data
import torchvision.transforms as transforms
import torch

from PIL import Image
from random import shuffle

class DataLoader(data.Dataset):
    def __init__(self,memory,classes=1000):
        self.memory = memory
        self.N = len(memory)
        self.permutations = self.__retrive_permutations(classes)

        self.__augment_tile = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std =[0.229, 0.224, 0.225])])
    
    def __getitem__(self, index):
        tiles = self.memory[index]
        tiles = [self.__augment_tile(tile) for tile in tiles]
        
        order = np.random.randint(len(self.permutations))
        data = [tiles[self.permutations[order][t]] for t in range(9)]
        data = torch.stack(data,0)
        return data,int(order), 0

    def __len__(self):
        return len(self.memory)
    
    def __retrive_permutations(self,classes):
        all_perm = np.load('permutations_%d.npy'%(classes))
        # from range [1,9] to [0,8]
        if all_perm.min()==1:
            all_perm = all_perm-1

        return all_perm


def rgb_jittering(im):
    im = np.array(im,np.float32)#convert to numpy array
    for ch in range(3):
        thisRand = np.random.uniform(0.8, 1.2)
        im[:,:,ch] *= thisRand
    shiftVal = np.random.randint(0,6)
    if np.random.randint(2) == 1:
        shiftVal = -shiftVal
    im += shiftVal;
    im = im.astype(np.uint8)
    im = im.astype(np.float32)
    return im
