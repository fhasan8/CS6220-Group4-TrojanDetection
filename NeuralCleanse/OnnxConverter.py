import os
import json
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import onnx
from onnx_tf.backend import prepare
# from MetaNetwork import MetaNetwork
import tensorflow as tf
# print(tf.__version__)


"""
This script is a conversion script that operates on a single file, used as a testbed to 
debug issues with our conversion pipeline
"""
filepath = "../trojan/id-0375/model.pt"
trained_model = torch.load(filepath)
print(trained_model)
dummy_input = Variable(torch.randn(1, 1, 28, 28))
torch.onnx.export(trained_model, dummy_input, filepath[:-3] + ".onnx")

###################
# Loading to ONNX #
###################
filepath = "../trojan/id-0375/model.onnx"
onnx_model = onnx.load(filepath)

filepath = "../trojan/id-0375/tf"
imported = tf.keras.models.load_model(filepath)

print(imported)
dummy_input = tf.random.uniform((1, 1, 28, 28))

filepath2 = '../trojan/id-0376/model.pt'
trained_model2 = torch.load(filepath2)
print(trained_model2)