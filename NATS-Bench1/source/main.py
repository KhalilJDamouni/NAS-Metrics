from __future__ import division
import sys
import random
import torch
import numpy as np
from .process import *
from nats_bench import create
from scipy.optimize import minimize_scalar
from pprint import pprint

# Create the API for tologoy search space
api = create(sys.path[0][0:-7]+'/fake_torch_dir/NATS-tss-v1_0-3ffb9-full', 'tss', fast_mode=True, verbose=False)
dataset = 'cifar10'
hp = '200'

#loop through all files{
info = api.get_more_info(531, dataset, hp='200', is_random=False)
params = api.get_net_param(531, 'cifar10', None)
get_quality(i)
increment_correlarion()
}


def get_quality(model_params):
    for i in model_params:
        #Gets main dictionary key
        for k, v in model_params[i].items():
            if('weight' in k and len(list(v.size())) == 4):
                print(k)
                rank, KG, condition = get_metrics(model_params,i,k)
                #accumulatequality
                print("\n")


def accumulate_quality(model_quality)