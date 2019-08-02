#!/usr/bin/env python

from image import *
from torchvision import transforms
import onnx
import caffe2.python.onnx.backend as onnx_caffe2_backend

# prepare model
model = onnx.load('csrnet.onnx')
prepared_backend = onnx_caffe2_backend.prepare(model)

# preprocess input image
transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])
imgpn = '/data/slin/singularity/test_projects/crowd_counting/testing_img'
img = transform(Image.open(imgpn).convert('RGB'))
W = {model.graph.input[0].name: img.unsqueeze(0).data.numpy()}

c2_out = prepared_backend.run(W)[0]
print('Predicted Count:', int(c2_out.sum()))