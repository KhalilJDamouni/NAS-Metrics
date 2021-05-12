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
import numpy.linalg as LA
import numpy as np
from nats_bench import create
from scipy.optimize import minimize_scalar
from pprint import pprint

api = create(sys.path[0][0:-7]+'/fake_torch_dir/models', 'tss', fast_mode=True, verbose=False)
dataset = 'cifar10'
hp = '200'

params = api.get_net_param(11197, dataset, None)
model_val = main.get_quality(params)
