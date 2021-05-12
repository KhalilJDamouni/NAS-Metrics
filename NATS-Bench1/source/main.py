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
    mquality_list = []
    for i in model_params:
        for k, v in (model_params[i]).items():
            if('weight' in k and len(list(v.size())) == 4 and v.shape[3]!=1):
                #print(k)
                #print("\n")
                rank, KG, condition, ER, in_quality, out_quality = process.get_metrics(model_params,i,k)
                #print(KG)
                if(in_quality>0):
                    mquality_list.append(in_quality)
                if(out_quality>0):
                    mquality_list.append(out_quality)
                if(KG>0 and condition>1):
                    #print(condition)
                    #print(math.atan(KG/(1.0-1.0/condition)))
                    condition_list.append(condition)
                    ER_list.append(ER)
                    KG_list.append(KG)
                    quality_list.append(math.atan(KG/(1.0-1.0/condition)))
    if(len(KG_list)==0):
        return None
    else:
        return [norm(quality_list,1),norm(quality_list,2),norm(quality_list,3),norm(KG_list,1),norm(condition_list,1),norm(condition_list,3),norm(ER_list,1),norm(mquality_list,1),norm(mquality_list,3)]


if __name__ == "__main__":
    api = create(sys.path[0][0:-7]+'/fake_torch_dir/models', 'tss', fast_mode=True, verbose=False)
    dataset = 'cifar10'
    hp = '200'
    early_stop=300
    i=0

    pickles=glob.glob(sys.path[0][0:-7]+'/fake_torch_dir/models/*')
    #model_qualities = []
    #test_accuracy = []

    
    file_name = save.get_name()
    '''
    params = api.get_net_param(11197, dataset, None)
    model_val = get_quality(params)

    '''
    for model in pickles:
        model_vals = []
        if(i+1>early_stop):
            break
        #try:
        model_num = int((model.split(os.path.sep)[-1]).split('.')[0])
        print(str(i+1)+'/'+str(early_stop))
        print("model: "+str(model_num)+"\n")
        params = api.get_net_param(model_num, dataset, None)
        model_val = get_quality(params)
        if(model_val):
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
            save.write(file_name,model_vals)
        else:
            print("skipping 0 model")
        #except:
            #print("skipping meta")
        i+=1



    '''
    print(str(model_qualities),str(test_accuracy))

    p_corr = correlate.pearson_corr(model_qualities, test_accuracy)
    ro_corr = correlate.rank_order_corr(model_qualities, test_accuracy)
    
    print(p_corr,ro_corr)

    correlate.display(model_qualities, test_accuracy)
    '''
