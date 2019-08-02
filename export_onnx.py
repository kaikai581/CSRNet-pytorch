#!/usr/bin/env python

import torch
from model import CSRNet

model = CSRNet()

# Load the weights from a file (.pth usually)
map_loc = 'cpu'
if torch.cuda.is_available(): map_loc = 'cuda:0'
checkpoint = torch.load('../0model_best.pth.tar', map_location=map_loc)

# Load the weights now into a model net architecture defined by our class
model.load_state_dict(checkpoint['state_dict'])

dummy_input = torch.randn(10, 3, 224, 224, device=map_loc)
torch.onnx.export(model, dummy_input, 'csrnet.onnx')