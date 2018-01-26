# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 11:58:07 2017

@author: Biagio Brattoli
"""
import os, numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torch

from PIL import Image

class DataLoader(data.Dataset):
    def __init__(self,data_path,classes=10,is_train=True):
        self.is_train = is_train
        self.data_path = data_path
        self.names = self.__dataset_info(data_path=data_path)
        self.permutations = self.__retrive_permutations(classes)

        self.__resize = transforms.Scale(256,Image.BILINEAR)
        self.__centerCrop = transforms.CenterCrop(225)
        self.__augment_tile = transforms.Compose([
                    transforms.RandomCrop(64),
                    transforms.Scale((75,75)),
                    transforms.Lambda(rgb_jittering),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std =[0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        framename = self.data_path+'/'+self.names[index]

        img = Image.open(framename).convert('RGB')
        img = self.__resize(img)
        img = self.__centerCrop(img)

        a = 75/2
        tiles = [None] * 9
        for n in range(9):
            i = n/3
            j = n%3
            c = [a*i*2+a,a*j*2+a]
            tile = img.crop((c[1]-a,c[0]-a,c[1]+a+1,c[0]+a+1))
            tile = self.__augment_tile(tile)
            # Normalize the patches indipendently to avoid low level features shortcut
            m = tile.mean()
            s = tile.std()
            norm = transforms.Normalize(mean=[m, m, m],
                                        std =[s, s, s])
            tile = norm(tile)
            tiles[n] = tile

        order = np.random.randint(len(self.permutations))
        data = [tiles[self.permutations[order][t]] for t in range(9)]
        data = torch.stack(data,0)
        return data,int(order), np.array(img)


    def __len__(self):
        return len(self.names)

    def __dataset_info(self,data_path='./data/'):
        file_names = []
        folders = os.listdir(data_path)
        for f in folders:
            if self.is_train:
                names = os.listdir(data_path+'/'+f)
                for ff in names:
                    if '.JPEG' in ff:
                        file_names.append(f+'/'+ff)
            else:
                if '.JPEG' in f:
                    file_names.append(f)

        return file_names

    def __retrive_permutations(self,classes):
        all_perm = np.load('JpsTraininig/permutations/permutations_%d.npy'%(classes))
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
