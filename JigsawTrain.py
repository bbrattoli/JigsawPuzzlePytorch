# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 12:16:31 2017

@author: bbrattol
"""
import os, sys, numpy as np
import argparse

import tensorflow # needs to call tensorflow before torch, otherwise crush
sys.path.append('Utils')
from logger import Logger

import torch
import torch.nn as nn
from torch.autograd import Variable

sys.path.append('JpsTraininig')
from JigsawLoader  import DataLoader
from JigsawNetwork import Network

from TrainingUtils import adjust_learning_rate, compute_accuracy


parser = argparse.ArgumentParser(description='Train JigsawPuzzleSolver on Imagenet')
parser.add_argument('data', type=str, help='Path to Imagenet folder')
parser.add_argument('--model', default=None, type=str, help='Path to pretrained model')
parser.add_argument('--classes', default=10, type=int, help='Number of permutation to use')
parser.add_argument('--gpu', default=None, type=int, help='gpu id')
parser.add_argument('--epochs', default=70, type=int, help='number of total epochs for training')
parser.add_argument('--iter_start', default=0, type=int, help='Starting iteration count')
parser.add_argument('--batch', default=256, type=int, help='batch size')
parser.add_argument('--checkpoint', default='checkpoints/', type=str, help='checkpoint folder')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate for SGD optimizer')
args = parser.parse_args()

classes, gpu = 2000, 3
#classes, gpu = 10000, 4
args = parser.parse_args([
   "/net/hci-storage01/userfolders/etikhonc/data/",
   '--classes',str(classes),'--model','JpsTraininig/jigsaw_original.pth.tar',
   '--batch','256','--checkpoint','checkpoints/Jigsaw++/perm%d_lr0.1_LRN/'%(classes),
   '--gpu',str(gpu)
])

# python JigsawTrain.py "/net/hci-storage01/userfolders/etikhonc/data/ILSVRC2012_img_train" \
#    --classes=1000 --checkpoint="checkpoints/BatchNorm/perm1000_lr1_normpatch/" --gpu=2 --lr=0.1 \
#    --model="checkpoints/perm1000_LRN_lr1/jps_012.pth.tar"

def main():
    if args.gpu is not None:
        print('Using GPU %d'%args.gpu)
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    else:
        print('CPU mode')

    # DataLoader initialize
    train_data = DataLoader(args.data+'/ILSVRC2012_img_train',
                            is_train=True,classes=args.classes)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=args.batch,
                                               shuffle=True,
                                               num_workers=4)

    val_data = DataLoader(args.data+'/ILSVRC2012_img_val',
                          is_train=False,classes=args.classes)
    val_loader = torch.utils.data.DataLoader(dataset=val_data,
                                               batch_size=args.batch,
                                               shuffle=True,
                                               num_workers=4)

    N = len(train_data.names)
    iter_per_epoch = N/args.batch
    
    # Network initialize
    classes = len(train_data.permutations)
    net = Network(classes)
    if args.gpu is not None:
        net.cuda()

    if os.path.exists(args.checkpoint):
        files = [f for f in os.listdir(args.checkpoint) if 'pth' in f]
        ckp = files[-1]
        net.load_state_dict(torch.load(args.checkpoint+ckp))
        args.iter_start = int(ckp.split(".")[-3].split("_")[-1])
        print 'Starting from: ',ckp
    else:
        if args.model is not None:
            net.load(args.model)

    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(net.parameters(), lr=params['lr'])
    optimizer = torch.optim.SGD(net.parameters(),lr=args.lr)

    logger = Logger(args.checkpoint+'/train')
    logger_test = Logger(args.checkpoint+'/test')

    ############## TRAINING ###############
    print('Start training: lr %f, batch size %d, classes %d'%(
                args.lr,args.batch,classes))
    print('Checkpoint: '+args.checkpoint)

    # Train the Model
    steps = args.iter_start
    for epoch in range(int(args.iter_start/iter_per_epoch),args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, init_lr=args.lr, step=30, decay=0.1)

        accuracy = []
        for i, (images, labels, _) in enumerate(train_loader):
            images = Variable(images)
            labels = Variable(labels)
            if args.gpu is not None:
                images = images.cuda()
                labels = labels.cuda()

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = net(images)

            prec1, prec5 = compute_accuracy(outputs.cpu().data, labels.cpu().data, topk=(1, 5))
            acc = prec1[0]
            accuracy.append(acc)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss = float(loss.cpu().data.numpy())

            if steps%100==0:
                print ('[%d/%d] %d) LR %.5f, Loss: %.3f, Accuracy %.1f%%' %(
                            epoch+1, args.epochs, steps, lr, loss,acc))

            if steps%20==0:
                logger.scalar_summary('accuracy', acc, steps)
                logger.scalar_summary('loss', loss, steps)
                #data = original.numpy()
                #logger.image_summary('input', data[:10], steps)

            steps += 1

            if steps%1000==0:
                filename = '%s/jps_%03i_%06d.pth.tar'%(args.checkpoint,epoch,steps)
                net.save(filename)
                print 'Saved: '+args.checkpoint

        if os.path.exists(args.checkpoint+'/stop.txt'):
            # break without using CTRL+C
            break

def test(net,criterion,logger,val_loader,steps):
    accuracy = []
    net.eval()
    for i, (images, labels, _) in enumerate(val_loader):
        images = Variable(images)
        if args.gpu is not None:
            images = images.cuda()

        # Forward + Backward + Optimize
        outputs = net(images)
        outputs = outputs.cpu().data

        prec1, prec5 = compute_accuracy(outputs, labels, topk=(1, 5))
        accuracy.append(prec1[0])

    if logger is not None:
        logger.scalar_summary('accuracy', np.mean(accuracy), steps)
    print 'TESTING: %d), Accuracy %.2f%%' %(steps,np.mean(accuracy))
    net.train()

if __name__ == "__main__":
    main()
