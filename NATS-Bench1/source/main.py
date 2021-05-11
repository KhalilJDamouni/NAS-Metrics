from __future__ import division
import sys
import random
import torch
import glob
import os
import math
import process
import correlate
import numpy.linalg as LA
import numpy as np
from nats_bench import create
from scipy.optimize import minimize_scalar
from pprint import pprint

def norm(x, L):
    #L: L1, or L2.
    if(L not in  [1, 2]):
        print("Error: L must be 1 or 2")
        exit()
    if(L == 1):
        return sum(np.abs(x))/len(x)
    if(L == 2):
        return LA.norm(x)/math.sqrt(len(x))


def get_quality(model_params,L):
    quality_list = []
    for i in model_params:
        for k, v in (model_params[i]).items():
            if('weight' in k and len(list(v.size())) == 4 and v.shape[3]!=1):
                #print(k)
                #print("\n")
                rank, KG, condition = process.get_metrics(model_params,i,k)
                try:
                    #print(condition)
                    #print(math.atan(KG/(1.0-1.0/condition)))
                    quality_list.append(math.atan(KG/(1.0-1.0/condition)))
                except:
                    quality_list.append(math.pi/2.0)
    temp = norm(quality_list,L)
    return temp


if __name__ == "__main__":
    api = create(sys.path[0][0:-7]+'/fake_torch_dir/NATS-tss-v1_0-3ffb9-full', 'tss', fast_mode=True, verbose=False)
    dataset = 'cifar10'
    hp = '200'
    L = 1
    early_stop=10
    i=0

    pickles=glob.glob(sys.path[0][0:-7]+'/fake_torch_dir/NATS-tss-v1_0-3ffb9-full/*')
    model_qualities = []
    test_accuracy = []

    for model in pickles:
        if(i>early_stop):
            break
        try:
            model_num = int((model.split(os.path.sep)[-1]).split('.')[0])
            print(str(model_num)+"\n")
            info = api.get_more_info(model_num, dataset, hp=hp, is_random=False)
            test_accuracy.append(info['test-accuracy']/100)
            params = api.get_net_param(model_num, dataset, None)
            model_qualities.append(get_quality(params,L))
        except:
            print("skipping meta")
        i+=1

    #print(str(model_qualities),str(test_accuracy))

    #p_corr = correlate.pearson_corr(model_qualities, test_accuracy)
    #ro_corr = correlate.rank_order_corr(model_qualities, test_accuracy)
    
    #print(p_corr,ro_corr)

    #correlate.display(model_qualities, test_accuracy)
