#!/usr/bin/env python

import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter 
import scipy
import json
import torchvision.transforms.functional as F
from matplotlib import cm as CM
from image import *
from model import CSRNet
import torch

from torchvision import datasets, transforms
transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])


model = CSRNet()

# Pretrained weights downloaded from https://drive.google.com/file/d/1KY11yLorynba14Sg7whFOfVeh2ja02wm/view?usp=sharing
checkpoint = torch.load('../0model_best.pth.tar', map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])

# An unsuccessful attempt
# img_paths = []
# img_paths.append('/data/slin/singularity/test_projects/crowd_counting/testing_img')
# for i in range(len(img_paths)):
#     img = 255.0 * F.to_tensor(Image.open(img_paths[i]).convert('RGB'))
#     img[0,:,:]=img[0,:,:]-92.8207477031
#     img[1,:,:]=img[1,:,:]-95.2757037428
#     img[2,:,:]=img[2,:,:]-104.877445883
#     output = model(img.unsqueeze(0))
#     print(output.detach().sum().numpy())

# Code adapted from https://www.analyticsvidhya.com/blog/2019/02/building-crowd-counting-model-python/
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5,8))
imgpn = '/data/slin/singularity/test_projects/crowd_counting/testing_img'
img = transform(Image.open(imgpn).convert('RGB'))
output = model(img.unsqueeze(0))
print('Predicted Count:', int(output.detach().cpu().sum().numpy()))
# Show the predicted density map.
temp = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2],output.detach().cpu().shape[3]))
ax1.imshow(temp, cmap=CM.jet)
ax1.set_title('Predicted Density Map')
# Show the original picture.
ax2.imshow(plt.imread(imgpn))
ax2.set_title('Predicted Count: {}'.format(int(output.detach().cpu().sum().numpy())))
plt.tight_layout()
plt.savefig('../prediction.jpg')