import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

class Inference_depth:
    def __init__(self, model, device='cuda:0'):
        self.device = device
        self.model = model
        model.eval()
        self.model = model.to(self.device)
        self.min_depth = 1e-3
        self.max_depth = 2

    @torch.no_grad()
    def predict_pil(self, img):
        bin_centers, pred = self.predict(img)

        return bin_centers, pred

    @torch.no_grad()
    def predict(self, image):
        bins, pred = self.model(image)
        pred = np.clip(pred.cpu().numpy(), self.min_depth, self.max_depth)

        # Flip
        image = torch.Tensor(np.array(image.cpu().numpy())[..., ::-1].copy()).to(self.device)
        pred_lr = self.model(image)[-1]
        pred_lr = np.clip(pred_lr.cpu().numpy()[..., ::-1], self.min_depth, self.max_depth)

        # Take average of original and mirror
        final = 0.5 * (pred + pred_lr)
        final = nn.functional.interpolate(torch.Tensor(final), image.shape[-2:],
                                          mode='bilinear', align_corners=True).cpu().numpy()

        final[final < self.min_depth] = self.min_depth
        final[final > self.max_depth] = self.max_depth
        final[np.isinf(final)] = self.max_depth
        final[np.isnan(final)] = self.min_depth

        centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        centers = centers.cpu().squeeze().numpy()
        centers = centers[centers > self.min_depth]
        centers = centers[centers < self.max_depth]

        return centers, final


class Inference_seg:
    def __init__(self, model, device='cuda:0'):
        self.device = device
        self.model = model
        model.eval()
        self.model = model.to(self.device)

    @torch.no_grad()
    def predict(self, image):
        pred = self.model(image)
        pred = F.interpolate(pred, size=image.size()[-2:], 
                                mode='bilinear', align_corners=True)
        return pred
