import torch
import os
import json
# from tqdm.notebook import tqdm
import numpy as np
# import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
cudnn.benchmark = True  # fire on all cylinders
# from sklearn.metrics import roc_auc_score, roc_curve
import sys

data_sources = ['CIFAR-10', 'CIFAR-100', 'GTSRB', 'MNIST']
data_source_to_channel = {k: 1 if k == 'MNIST' else 3 for k in data_sources}
data_source_to_resolution = {k: 28 if k == 'MNIST' else 32 for k in data_sources}
data_source_to_num_classes = {'CIFAR-10': 10, 'CIFAR-100': 100, 'GTSRB': 43, 'MNIST': 10}

class MetaNetwork(nn.Module):
    def __init__(self, num_queries, num_classes=1):
        super().__init__()
        self.queries = nn.ParameterDict(
            {k: nn.Parameter(torch.rand(num_queries,
                                        data_source_to_channel[k],
                                        data_source_to_resolution[k],
                                        data_source_to_resolution[k])) for k in data_sources}
        )
        self.affines = nn.ModuleDict(
            {k: nn.Linear(data_source_to_num_classes[k]*num_queries, 32) for k in data_sources}
        )
        self.norm = nn.LayerNorm(32)
        self.relu = nn.ReLU(True)
        self.final_output = nn.Linear(32, num_classes)
    
    def forward(self, net, data_source):
        """
        :param net: an input network of one of the model_types specified at init
        :param data_source: the name of the data source
        :returns: a score for whether the network is a Trojan or not
        """
        query = self.queries[data_source]
        out = net(query)
        out = self.affines[data_source](out.view(1, -1))
        out = self.norm(out)
        out = self.relu(out)
        return self.final_output(out)