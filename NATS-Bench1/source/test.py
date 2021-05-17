from __future__ import division
import sys
import random
import torch
import glob
import os
import math
import main
import process
import correlate
import save
import resnet_cifar
import numpy.linalg as LA
import numpy as np
import torch
from nats_bench import create
from scipy.optimize import minimize_scalar
from pprint import pprint

model = None
model = torch.load('C://Users/jjaeg/Desktop/Nas-metrics/Nas-Metrics/NATS-Bench1/source/trial_0_epoch_25.pth.tar')

param_dict = dict()
param_dict[111] = dict()

for name in model['state_dict_network']:
    param_dict[111][name] = model['state_dict_network'][name]

main.get_quality(param_dict)

