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
    if(L not in  [1, 2]):
        print("Error: L must be 1 or 2")
        exit()
    if(L == 1):
        return np.mean(np.abs(x))
    if(L == 2):
        return LA.norm(x)/math.sqrt(len(x))


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
                KG_list.append(KG)
                condition_list.append(condition)
                ER_list.append(ER)
                if(KG>0):
                    #print(condition)
                    #print(math.atan(KG/(1.0-1.0/condition)))
                    quality_list.append(math.atan(KG/(1.0-1.0/condition)))
                else:
                    return None
    return [norm(quality_list,1),norm(quality_list,2),norm(KG_list,1),norm(condition_list,1),norm(ER_list,1)]


if __name__ == "__main__":
    api = create(sys.path[0][0:-7]+'/fake_torch_dir/models', 'tss', fast_mode=True, verbose=False)
    dataset = 'cifar10'
    hp = '200'
    early_stop=60
    i=0

    pickles=glob.glob(sys.path[0][0:-7]+'/fake_torch_dir/models/*')
    #model_qualities = []
    #test_accuracy = []

    file_name = save.get_name()

    for model in pickles:
        model_vals = []
        if(i>early_stop):
            break
        try:
            model_num = int((model.split(os.path.sep)[-1]).split('.')[0])
            print(str(model_num)+"\n")
            params = api.get_net_param(model_num, dataset, None)
            if(params):
                model_vals.append(model_num)
                info = api.get_more_info(model_num, dataset, hp=hp, is_random=False)
                #test_accuracy.append(info['test-accuracy']/100)
                #print(info)
                model_vals.append(info['test-accuracy']/100)
                model_vals.append(info['test-loss'])
                model_vals.append(info['train-accuracy']/100)
                model_vals.append(info['train-loss'])
                #model_qualities.append(get_quality(params))
                model_vals.extend(get_quality(params))
                print(model_vals)
                save.write(file_name,model_vals)
            else:
                print("0 layer")
        except:
            print("skipping meta")
        i+=1



    '''
    print(str(model_qualities),str(test_accuracy))

    p_corr = correlate.pearson_corr(model_qualities, test_accuracy)
    ro_corr = correlate.rank_order_corr(model_qualities, test_accuracy)
    
    print(p_corr,ro_corr)

    correlate.display(model_qualities, test_accuracy)
    '''
