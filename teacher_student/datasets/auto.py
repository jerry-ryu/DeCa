# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
import torch

from .base_dataset import BaseDataset

def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s

class Auto(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path,
                 num_classes=5,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=255, 
                 base_size=2048, 
                 crop_size=(416, 544),
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4,
                 mode = "train"):

        super(Auto, self).__init__(ignore_label, base_size,
                crop_size, scale_factor, mean, std,)

        self.class_weights = torch.FloatTensor([0.7, 5.0, 5.0, 5.0, 5.0]).cuda()
        
        self.root = root
        self.list_path = list_path
        with open(self.list_path, 'r') as f:
            self.filenames = f.readlines()
            
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        
        self.bd_dilate_size = bd_dilate_size
        self.base_size = base_size
        self.mode = mode
        
    def __len__(self):
        return len(self.filenames)
            
    # def resize_image(self, image, mode = "image"):
    #     if mode == "image":
    #         image = cv2.resize(image, (self.base_size, self.base_size),
    #                        interpolation=cv2.INTER_LINEAR)
    #     else:
    #         image = cv2.resize(image, (self.base_size, self.base_size),
    #                            interpolation=cv2.INTER_NEAREST)
    #     return image

    def __getitem__(self, index):
        item = self.filenames[index]
        
        image_path = os.path.join(self.root, remove_leading_slash(item.split()[0]))
        seg_path = os.path.join(self.root, remove_leading_slash(item.split()[1]))
        
        name = image_path
        
        image = cv2.imread(image_path,
                           cv2.IMREAD_COLOR)
        image = self.input_transform(image)
        image = image.transpose((2, 0, 1))
        
        
        # image = self.resize_image(image, mode = "image")
        size = image.shape
        
        if self.mode == "test":
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name

        label = cv2.imread(seg_path,
                           cv2.IMREAD_GRAYSCALE)
        
        # label = self.resize_image(label, mode = "label")

        return image.copy(), label.copy()
    
    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred
    
   
        
        
