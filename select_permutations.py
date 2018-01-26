# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:50:28 2017

@author: bbrattol
"""
import argparse
from tqdm import trange
import numpy as np
import itertools
from scipy.spatial.distance import cdist


parser = argparse.ArgumentParser(description='Train network on Imagenet')
parser.add_argument('--classes', default=1000, type=int, 
                    help='Number of permutations to select')
parser.add_argument('--selection', default='max', type=str, 
        help='Sample selected per iteration based on hamming distance: [max] highest; [mean] average')
args = parser.parse_args()

if __name__ == "__main__":
    outname = 'permutations/permutations_hamming_%s_%d'%(args.selection,args.classes)
    
    P_hat = np.array(list(itertools.permutations(range(9), 9)))
    n = P_hat.shape[0]
    
    for i in trange(args.classes):
        if i==0:
            j = np.random.randint(n)
            P = np.array(P_hat[j]).reshape([1,-1])
        else:
            P = np.concatenate([P,P_hat[j].reshape([1,-1])],axis=0)
        
        P_hat = np.delete(P_hat,j,axis=0)
        D = cdist(P,P_hat, metric='hamming').mean(axis=0).flatten()
        
        if args.selection=='max':
            j = D.argmax()
        else:
            m = int(D.shape[0]/2)
            S = D.argsort()
            j = S[np.random.randint(m-10,m+10)]
        
        if i%100==0:
            np.save(outname,P)
    
    np.save(outname,P)
    print 'file created --> '+outname
