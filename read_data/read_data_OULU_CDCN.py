from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset

import math
import cv2
import torchvision
import torch
import random

class CASIA(Dataset):
    def __init__(self,data_flag = None, transform=None, phase_train=True, data_dir=None,phase_test=False,add_mask = True):

        self.phase_train = phase_train
        self.phase_test = phase_test
        self.transform = transform
        self.add_mask = add_mask
        self.mask_np = None

        #for train
        train_file = os.getcwd() +'/data/nir_0603/exp_train_set_21060301_exp_20210603221606NIR_train.txt'
        label_train_file = os.getcwd() + '/data/nir_0603/exp_train_set_21060301_exp_20210603221606NIR_train_label.txt'
        
        # for val
        val_file = os.getcwd() +'/data/nir_0603/exp_train_set_21060301_exp_20210603221606NIR_val.txt'
        label_val_file = os.getcwd() + '/data/nir_0603/exp_train_set_21060301_exp_20210603221606NIR_val_label.txt'
        
        # for test
        test_file = os.getcwd() +'/data/OULU/oulu_test_20210322_val.txt'
        label_test_file = os.getcwd() + '/data/OULU/oulu_test_20210322_val_label.txt'

        self.mask_np = np.fromfile('./data/mask_file/mask_for_nir.bin', np.uint8).reshape((112,112))
        

        try:
            with open(train_file, 'r') as f:
                self.depth_dir_train = f.read().splitlines()
            with open(label_train_file, 'r') as f:
                self.label_dir_train = f.read().splitlines()
                
            with open(val_file, 'r') as f:
                 self.depth_dir_val = f.read().splitlines()
            with open(label_val_file, 'r') as f:
                self.label_dir_val = f.read().splitlines()
            if self.phase_test:
                with open(test_file, 'r') as f:
                    self.depth_dir_test = f.read().splitlines()
                with open(label_test_file, 'r') as f:
                    self.label_dir_test = f.read().splitlines()
        except:
            print('can not open files, may be filelist is not exist')
            exit()

    def __len__(self):
        if self.phase_train:
            return len(self.depth_dir_train)
        else:
            if self.phase_test:
                return len(self.depth_dir_test)
            else:
                return len(self.depth_dir_val)

    def __getitem__(self, idx):
        if self.phase_train:
            depth_dir = self.depth_dir_train
            label_dir = self.label_dir_train
            label = int(label_dir[idx])
            label = np.array(label)
        else:
            if self.phase_test:
                depth_dir = self.depth_dir_test
                label_dir = self.label_dir_test
                label = np.random.randint(0,2,1)
                label = np.array(label)
            else:
                depth_dir = self.depth_dir_val
                label_dir = self.label_dir_val
                label = int(label_dir[idx])
                label = np.array(label)

        depth = Image.open(depth_dir[idx])
        depth = depth.convert('RGB')
        # depth = depth.convert('L')
        

        # '''filp left and right randonly and add mask'''
        # if random.randint(0,9) < 5:
        #     depth = depth.transpose(Image.FLIP_LEFT_RIGHT)     #????????????

        '''transform'''
        if self.transform:
            depth = self.transform(depth)

        if self.phase_train:
            return depth,label
        else:
            return depth,label,depth_dir[idx]

