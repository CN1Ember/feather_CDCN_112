from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset

import math
import cv2
import torchvision
import torch

# CASIA-SURF training dataset and our private dataset
# depth_dir_train_file = os.getcwd() +'/data/2depth_train.txt'
# label_dir_train_file = os.getcwd() + '/data/2label_train.txt'

# CASIA-SURF training dataset and our private dataset
# /home/lidaiyuan/feathernet2020/FeatherNet/data/train_file_list/exp_train_set_nir_21031802_exp_20210325205339_val_label.txt
depth_dir_train_file = os.getcwd() +'/data/nir_0603/exp_train_set_21060301_exp_20210603221606NIR_train.txt'
label_dir_train_file = os.getcwd() + '/data/nir_0603/exp_train_set_21060301_exp_20210603221606NIR_train_label.txt'

# for IR train
# depth_dir_train_file = os.getcwd() +'/data/ir_final_train.txt'
# label_dir_train_file = os.getcwd() +'/data/label_ir_train.txt'

# # CASIA-SURF Val data
depth_dir_val_file = os.getcwd() +'/data/nir_0603/exp_train_set_21060301_exp_20210603221606NIR_val.txt'
label_dir_val_file = os.getcwd() + '/data/nir_0603/exp_train_set_21060301_exp_20210603221606NIR_val_label.txt' #val-label 100%

# # JIDIAN Val data
# depth_dir_val_file = os.getcwd() +'/data/nir_outdoor_0224/nir_0325_outdoor.txt'
# label_dir_val_file = os.getcwd() +'/data/nir_outdoor_0224/label_0325_outdoor.txt' #val-label 100%

# # CASIA-SURF Test data 
depth_dir_test_file = os.getcwd() +'/data/nir_031802/exp_train_set_nir_21031802_exp_20210325205339_val.txt'
label_dir_test_file = os.getcwd() +'/data/nir_031802/exp_train_set_nir_21031802_exp_20210325205339_val_label.txt'


# depth_dir_test_file = os.getcwd() +'/data/ir_test.txt'
# label_dir_test_file = os.getcwd() +'/data/label_test.txt'

class CASIA(Dataset):
    def __init__(self, transform=None, phase_train=True, data_dir=None, phase_val=False, phase_test=False, face_like=False):

        self.phase_train = phase_train
        self.phase_val = phase_val
        self.phase_test = phase_test
        self.transform = transform
        self.face_like = face_like

        try:
            with open(depth_dir_train_file, 'r') as f:
                self.depth_dir_train = f.read().splitlines()
            with open(label_dir_train_file, 'r') as f:
                self.label_dir_train = f.read().splitlines()
                
            with open(depth_dir_val_file, 'r') as f:
                 self.depth_dir_val = f.read().splitlines()
            with open(label_dir_val_file, 'r') as f:
                self.label_dir_val = f.read().splitlines()
            if self.phase_test:
                with open(depth_dir_test_file, 'r') as f:
                    self.depth_dir_test = f.read().splitlines()
                with open(label_dir_test_file, 'r') as f:
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
#                 label = int(label_dir[idx])
                label = np.random.randint(0,2,1)
                label = np.array(label)
            else:
                depth_dir = self.depth_dir_val
                label_dir = self.label_dir_val
                label = int(label_dir[idx])
                label = np.array(label)

        depth = Image.open(depth_dir[idx])
        # depth = depth.convert('RGB')
        depth = depth.convert('L')
        if self.face_like:
            mask_img = cv2.imread('data/mask_img_14x14.png', 2)
            mask_img[mask_img>0]=1
            fmap = mask_img*label
        else:
            fmap = np.ones((14,14), np.uint8) * label

        # print("read_data:107",len(depth.split()))

        if self.transform:
            depth = self.transform(depth)
        if self.phase_train:
            return depth,label,fmap
        elif self.phase_val:
            return depth,label,fmap
        else:
            return depth,label,depth_dir[idx]

