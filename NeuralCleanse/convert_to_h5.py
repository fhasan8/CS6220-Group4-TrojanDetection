import os
import json
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import tensorflow as tf
print(tf.__version__)

import onnx
from onnx_tf.backend import prepare
from MetaNetwork import MetaNetwork

"""
This script is written as two for loops for modularities sake; the first one searches through the 'ROOT/APPENDS',
opening every folder and converting .pt files to .onnx; the second does the same and converts .onnx files into .pb
"""
root = "D:/Documents/GTECH/Fall 22/6220 Bid data/Competition/tdc-datasets/mnist/train"
appends = ['clean', 'trojan']

for append in appends:
    for file in os.listdir(os.path.join(root, append)):
        info_model = 'model.pt'
        # print("hello")
        filepath = os.path.join(os.path.join(os.path.join(root, append, file)), info_model)
        # print(filepath)


        # trained_model = MetaNetwork(10, num_classes=1)
        trained_model = torch.load(filepath)
        # print(trained_model)
        # trained_model.load_state_dict(torch.load(filepath))
        dummy_input = Variable(torch.randn(1, 1, 28, 28))
        torch.onnx.export(trained_model, dummy_input, filepath + ".onnx")

for append in appends:
    for file in os.listdir(os.path.join(root, append)):

        for info_model in os.listdir(os.path.join(os.path.join(root, append, file))):
            if '.onnx' in info_model:

                filepath = os.path.join(os.path.join(os.path.join(root, append, file)), info_model)
                model = onnx.load(filepath)
                tf_rep = prepare(model)
                tf_rep.export_graph(filepath + '.pb')

                # model = tf.keras.models.load_model(filepath + '.pb')
                # tf.keras.models.save_model(model, filepath + '.h5')