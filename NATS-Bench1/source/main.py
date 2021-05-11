from __future__ import division
import sys
import random
import torch
import glob
import os
import math
import numpy as np
import process
from nats_bench import create
from scipy.optimize import minimize_scalar
from pprint import pprint


def get_quality(model_params):
    for i in model_params:
        for k, v in (model_params[i]).items():
            quality_list = []
            if('weight' in k and len(list(v.size())) == 4 and v.shape[3]!=1):
                print(k)
                print("\n")
                rank, KG, condition = process.get_metrics(model_params,i,k)
                try:
                    quality_list.append(math.atan(KG/(1-1/condition)))
                except:
                    quality_list.append(0)


if __name__ == "__main__":
    api = create(sys.path[0][0:-7]+'/fake_torch_dir/NATS-tss-v1_0-3ffb9-full', 'tss', fast_mode=True, verbose=False)
    dataset = 'cifar10'
    hp = '200'

    pickles=glob.glob(sys.path[0][0:-7]+'/fake_torch_dir/NATS-tss-v1_0-3ffb9-full/*')
    model_qualities = []

    for model in pickles:
        model_num = int((model.split(os.path.sep)[-1]).split('.')[0])
        print(model_num)
        info = api.get_more_info(model_num, dataset, hp=hp, is_random=False)
        params = api.get_net_param(model_num, dataset, None)
        get_quality(params)
        #increment_correlarion()