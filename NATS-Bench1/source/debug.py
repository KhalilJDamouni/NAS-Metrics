from __future__ import division
import sys
import random
import torch
import glob
import os
import math
import process
import correlate
import save
import numpy.linalg as LA
import numpy as np
from nats_bench import create
from scipy.optimize import minimize_scalar
from pprint import pprint

def norm(x, L):
    #L: L1, or L2.
    if(L not in  [1, 2, 3]):
        print("Error: L must be 1, 2 or 3")
        exit()
    if(L == 1):
        return np.mean(np.abs(x))
    if(L == 2):
        return LA.norm(x)/math.sqrt(len(x))
    if(L == 3):
        return np.power(np.prod(x),1/len(x))


def get_quality(model_params):
    quality_list = []
    KG_list = []
    condition_list = []
    ER_list = []
    for i in model_params:
        for k, v in (model_params[i]).items():
            if('weight' in k and len(list(v.size())) == 4 and v.shape[3]!=1):
                #print(k)
                #print("\n")
                rank, KG, condition, ER = process.get_metrics(model_params,i,k)
                #print(KG)
                if(KG>0):
                    #print(condition)
                    #print(math.atan(KG/(1.0-1.0/condition)))
                    condition_list.append(condition)
                    ER_list.append(ER)
                    KG_list.append(KG)
                    quality_list.append(math.atan(KG/(1.0-1.0/condition)))
                else:
                    print("skipping 0 layer")
    if(len(KG_list)==0):
        return None
    else:
        return [norm(quality_list,1),norm(quality_list,2),norm(quality_list,3),norm(KG_list,1),norm(condition_list,1),norm(ER_list,1)]


if __name__ == "__main__":
    api = create(sys.path[0][0:-7]+'/fake_torch_dir/test', 'tss', fast_mode=True, verbose=False)
    dataset = 'cifar10'
    hp = '200'


    model_vals = []
    model_num = 11197

    params = api.get_net_param(model_num, dataset, None)
    
    model_val = get_quality(params)
 
    model_vals.append(model_num)
    info = api.get_more_info(model_num, dataset, hp=hp, is_random=False)
    #test_accuracy.append(info['test-accuracy']/100)
    #print(info)
    model_vals.append(info['test-accuracy']/100)
    model_vals.append(info['test-loss'])
    model_vals.append(info['train-accuracy']/100)
    model_vals.append(info['train-loss'])
    #model_qualities.append(get_quality(params))
    model_vals.extend(model_val)
    #print(model_vals)

    
    print(model_vals)
