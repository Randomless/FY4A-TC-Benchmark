import torch
import cv2
from PIL import Image
import numpy as np
import pickle
# import pickle5 as p
import os

# fy4a_channel_keys=[ 'NOMChannel01', 'NOMChannel02', 'NOMChannel03', 'NOMChannel04', 'NOMChannel05', 'NOMChannel06', 'NOMChannel07', 'NOMChannel08', 'NOMChannel09', 'NOMChannel10', 'NOMChannel11', 'NOMChannel12', 'NOMChannel13', 'NOMChannel14']

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img_path, label = self.img_paths[index], self.labels[index]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = imread(img_path)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.img_paths)


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, labels, transform=None,channels=14,dataset_type=None,daynight_mode=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        self.channels = channels
        self.dataset_type = dataset_type
        self.daynight_mode = daynight_mode

    def __getitem__(self, index):
        img_path, label = self.img_paths[index], self.labels[index]
        if 'pca' in self.dataset_type:
            img_path=img_path.replace('inter_patch_knts_pkl_classwise','pca4_inter_patch_knts_pkl_classwise')

        if  "pickle" in img_path:
            with open(img_path, 'rb') as f:
                img = p.load(f)
        elif "npy" in img_path:
            img=np.load(img_path)
        else:
            assert "Unsupported data suffix: {}".format(img_path) 
        # print(img.shape)

        if self.transform is not None:
            img = self.transform(img)

        # class=label
        # speed= np.float32(float(img_path.split('/')[-1].split('_')[1]))
        speed= 0

        # dataset/patch_pkl_classwise/0/0.0_14.0_BAILU_2019-08-25 14:45:00.pickle
        return img, label,speed,os.path.basename(img_path)

    def __len__(self):
        return len(self.img_paths)
