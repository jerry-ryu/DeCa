# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------

import glob
import argparse
import cv2
import os
import numpy as np
import _init_paths
import models
import torch
import torch.nn.functional as F
from PIL import Image
import unet


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

color_map = [(128, 64,128),
             (244, 35,232),
             ( 70, 70, 70),
             (102,102,156),
             (190,153,153)]

def parse_args():
    parser = argparse.ArgumentParser(description='Custom Input')
    
    parser.add_argument('--p', help='dir for pretrained model', default='../pretrained_models/cityscapes/PIDNet_L_Cityscapes_test.pt', type=str)
    parser.add_argument('--r', help='root or dir for input images', default='../samples/', type=str)
    parser.add_argument('--t', help='the format of input images (.jpg, .png, ...)', default='.png', type=str)     
    parser.add_argument('--viz', help= 'visualization', action='store_true')
    args = parser.parse_args()

    return args

def input_transform(image):
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image

def load_pretrained(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model.load_state_dict(pretrained_dict, strict = True)
    
    return model

if __name__ == '__main__':
    args = parse_args()
    images_list = glob.glob(args.r+'*'+args.t)
    print(args.r+'*'+args.t)
    print(images_list)
    sv_path = args.r+'outputs/'
    min_depth = 1e-3
    max_depth = 2
    
    model = student_model = unet.UNet(3,5).cuda()
    model = load_pretrained(model, args.p).cuda()
    model.eval()
    with torch.no_grad():
        for img_path in images_list:
            img_name = img_path.split("/")[-1]
            img = cv2.imread(os.path.join(args.r, img_name),
                               cv2.IMREAD_COLOR)
            sv_img = np.zeros_like(img).astype(np.uint8)
            height, width, _= img.shape
            
            img = input_transform(img)
            img = img.transpose((2, 0, 1)).copy()
            img = torch.from_numpy(img).unsqueeze(0).cuda()
            pred_seg, pred_depth = model(img)
            pred_seg = F.interpolate(pred_seg, size=img.size()[-2:], 
                                 mode='bilinear', align_corners=True)
            pred_depth = F.interpolate(pred_depth, size=img.size()[-2:], 
                                 mode='bilinear', align_corners=True)
            pred_depth = pred_depth.cpu().numpy()
            pred_depth[pred_depth < min_depth] = min_depth
            pred_depth[pred_depth > max_depth] = max_depth
            pred_depth[np.isinf(pred_depth)] = max_depth
            pred_depth[np.isnan(pred_depth)] = min_depth
            pred_depth = pred_depth * 1000.
            
            pred_seg_max = torch.argmax(pred_seg, dim=1).squeeze(0).cpu().numpy()
            
            if args.viz:
                
                for i, color in enumerate(color_map):
                    for j in range(3):
                        sv_img[:,:,j][pred_seg_max==i] = color_map[i][j]
                sv_img = Image.fromarray(sv_img)
                
                pred_depth = (pred_depth * 1000.).astype('uint16')
                depth_img = Image.fromarray(pred_depth.squeeze())
            
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                sv_img.save(sv_path+img_name)
                
                depth_img.save(f'{sv_path+img_name}_depth.png')
                
                print("save img")
            
            seg_choice = (pred_seg_max == np.arange(5)[:, None, None]).astype(int)
            seg_choice = seg_choice * pred_depth
            seg_choice = seg_choice.squeeze()
            
            non_zero_counts = np.count_nonzero(seg_choice[:, :, :], axis=(1, 2))
            mask = non_zero_counts > 1000
            indices = np.where(mask)[0]
            seg_masked = seg_choice[mask]
            seg_choice_reshaped = seg_masked.reshape(-1, seg_choice.shape[1]*seg_choice.shape[2])
            result = np.mean(seg_choice_reshaped, axis = 1)


            result_dict = dict(zip(indices, result))
            result_dict.pop(0)
            
            with open(os.path.join(sv_path, "result.txt"), "a") as file:
                file.write(f"{img_name}\n")
                for key, value in result_dict.items():
                    file.write(f"{key}: {value}\n")
                
                
            
            
            
            
            
        
        