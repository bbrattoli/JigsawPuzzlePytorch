# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 11:58:07 2017

@author: Biagio Brattoli
"""
import os, numpy as np
from time import time
import torch.utils.data as data
import torchvision.transforms as transforms
import torch

from PIL import Image
from random import shuffle

def load_image(path,permutations,image_transformer,augment_tile):
    img = Image.open(path).convert('RGB')
    img = image_transformer(img)
    
    a = 75/2
    tiles = [None] * 9
    for n in range(9):
        i = n/3
        j = n%3
        c = [a*i*2+a,a*j*2+a]
        tile = img.crop((c[1]-a,c[0]-a,c[1]+a+1,c[0]+a+1))
        tile = augment_tile(tile)
        # Normalize the patches indipendently to avoid low level features shortcut
        #m = tile.mean()
        #s = tile.std()
        #norm = transforms.Normalize(mean=[m, m, m],
                                    #std =[s, s, s])
        #tile = norm(tile)
        tiles[n] = tile
    
    order = np.random.randint(len(permutations))
    data = [tiles[permutations[order][t]] for t in range(9)]
    data = torch.stack(data,0)
    return data, int(order)

class DataLoader():
    def __init__(self,data_path,txt_list,batchsize=256,classes=1000):
        self.batchsize = batchsize
        
        self.data_path = data_path
        self.names, _ = self.__dataset_info(txt_list)
        self.N = len(self.names)
        #self.N = self.N-(self.N%batchsize)
        
        self.permutations = self.__retrive_permutations(classes)

        self.__image_transformer = transforms.Compose([
                            transforms.Resize(256,Image.BILINEAR),
                            transforms.CenterCrop(225)])
        self.__augment_tile = transforms.Compose([
                    transforms.RandomCrop(64),
                    transforms.Resize((75,75)),
                    transforms.Lambda(rgb_jittering),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std =[0.229, 0.224, 0.225])])
        
    def __iter__(self):
        self.counter = 0
        shuffle(self.names)
        return self
    
    def next(self):
        try:
            names = [self.data_path+'/'+n for n in self.names[self.counter:self.counter+self.batchsize]]
        except IndexError:
            raise StopIteration
        self.counter += self.batchsize
        batch = [load_image(n,self.permutations,self.__image_transformer,self.__augment_tile) 
                 for n in names]
        
        data, labels = zip(*batch)
        labels = torch.LongTensor(labels)
        data = torch.stack(data, 0)
        return data, labels, 0
    
    def __dataset_info(self,txt_labels):
        with open(txt_labels,'r') as f:
            images_list = f.readlines()
        
        file_names = []
        labels     = []
        for row in images_list:
            row = row.split(' ')
            file_names.append(row[0])
            labels.append(int(row[1]))
        
        return file_names, labels
    
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