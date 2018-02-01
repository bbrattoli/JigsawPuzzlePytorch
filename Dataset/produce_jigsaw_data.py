# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 12:16:31 2017

@author: Biagio Brattoli
"""
import os, sys, numpy as np
import argparse
from time import time
from tqdm import tqdm

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image

datapath = '/Datasets/ImageNet/'

def main():
    data = DataLoader(datapath+'/ILSVRC2012_img_train', datapath+'/ilsvrc12_train.txt')
    loader = torch.utils.data.DataLoader(dataset=data,batch_size=1,
                                        shuffle=False,num_workers=6)
    
    count = 0
    for i, (images, filename) in enumerate(tqdm(loader)):
        count += 1


class DataLoader(data.Dataset):
    def __init__(self,data_path,txt_list):
        self.data_path = data_path if data_path[-1]!='/' else data_path[:-1]
        self.names, _ = self.__dataset_info(txt_list)
        self.__image_transformer = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(225)])
        self.__augment_tile = transforms.Compose([
                    transforms.RandomCrop(64),
                    transforms.Resize((75,75)),
                    transforms.Lambda(rgb_jittering),
                    ])
        self.save_path = self.data_path+'_jigsaw/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        for name in self.names:
            if '/' in name:
                fold = self.save_path+name[:name.rfind('/')]
                if not os.path.exists(fold):
                    os.makedirs(fold)
    
    def __getitem__(self, index):
        name = self.names[index]
        if os.path.exists(self.save_path+name[:-5]+'.npy'):
            return None, None
        
        filename = self.data_path+'/'+name
        img = Image.open(filename).convert('RGB')
        img = self.__image_transformer(img)
        
        a = 75/2
        tiles = [None] * 9
        for n in range(9):
            i, j = n/3, n%3
            c = [a*i*2+a,a*j*2+a]
            tile = img.crop((c[1]-a,c[0]-a,c[1]+a+1,c[0]+a+1))
            tile = self.__augment_tile(tile)
            tiles[n] = tile.astype('uint8')
        
        np.save(self.save_path+name[:-5],tiles)
        return tiles, self.names[index]


    def __len__(self):
        return len(self.names)
    
    def __dataset_info(self,txt_labels):
        with open(txt_labels,'r') as f:
            images_list = f.readlines()
        
        file_names = []
        labels     = []
        for row in images_list:
            row = row.split(' ')
            file_names.append(row[0])
            labels.append(int(row[1]))
            #if len(file_names)>128*10:
                #break
        
        return file_names, labels

def rgb_jittering(im):
    im = np.array(im,np.float32)#convert to numpy array
    if im.max()<=1.0:
        im = im*255
    for ch in range(3):
        thisRand = np.random.uniform(0.8, 1.2)
        im[:,:,ch] *= thisRand
    shiftVal = np.random.randint(0,6)
    if np.random.randint(2) == 1:
        shiftVal = -shiftVal
    im += shiftVal;
    im = im.astype(np.uint8)
    return im

if __name__ == "__main__":
    main()

#import matplotlib.pyplot as plt
#import numpy as np
#datapath = '/Datasets/ImageNet/ILSVRC2012_img_train_jigsaw/n01440764/n01440764_18.npy'
#a = np.load(datapath)
#b = np.histogram(a)
#print b
#plt.imshow(a[0].transpose((1,2,0))); plt.show()
