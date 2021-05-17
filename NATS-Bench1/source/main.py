from __future__ import division
import sys
import random
from numpy.core.function_base import linspace
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

def norm(x, L, a=[]):
    #L: L1, or L2.
    if(L not in range(1,8)):
        print("Error: L must be 1:7")
        exit()
    if(L == 1):
        return np.mean(np.abs(x))
    if(L == 2):
        return LA.norm(x)/math.sqrt(len(x))
    if(L == 3):
        return np.power(np.prod(x),1/len(x))
    if(L == 4):
        #weighted average
        return np.average(np.abs(x),weights=a)
    if(L == 5):
        #weighted product
        return np.prod(np.power(x,a/(np.sum(a))))
    if(L == 6):
        #weighted product with linear depth weights as well
        depth = np.arange(len(x))+1
        a = a*depth
        return np.prod(np.power(x,a/(np.sum(a))))
    if(L == 7):
        depth = np.flip(np.arange(len(x))+1)
        a = a*depth
        return np.prod(np.power(x,a/(np.sum(a))))



def get_quality(model_params):
    quality_list = []
    quality_new_list = []
    quality_newr_list = []
    quality_newpr_list = []
    KG_list = []
    condition_list = []
    ER_list = []
    mquality_list = []
    weights = []
    for i in model_params:
        for k, v in (model_params[i]).items():
            if('weight' in k and len(list(v.size())) == 4 and v.shape[3]!=1):
                #print(k)
                #print("\n")
                rank, KG, condition, ER, in_quality, out_quality, in_weight, out_weight, in_quality_new, out_quality_new, in_quality_newp, out_quality_newp = process.get_metrics(model_params,i,k)
                #print(KG)
                if(in_quality>0):
                    mquality_list.append(in_quality)
                    quality_new_list.append(in_quality_new)
                    quality_newr_list.append(1/in_quality_new)
                    quality_newpr_list.append(1/in_quality_newp)
                    weights.append(in_weight)
                if(out_quality>0):
                    mquality_list.append(out_quality)
                    quality_new_list.append(out_quality_new)
                    quality_newr_list.append(1/out_quality_new)
                    quality_newpr_list.append(1/out_quality_newp)
                    weights.append(out_weight)
                if(KG>0 and condition>1):
                    #print(condition)
                    #print(math.atan(KG/(1.0-1.0/condition)))
                    condition_list.append(condition)
                    ER_list.append(ER)
                    KG_list.append(KG)
                    quality_list.append(np.arctan2(KG,(1.0-1.0/condition)))
    #print(str(len(KG_list)),str(sum(weights)))
    if(len(KG_list)==0):
        return None
    else:
        print(norm(quality_list,2))
        return [norm(quality_list,1),norm(quality_list,2),norm(quality_list,3),norm(KG_list,1),norm(condition_list,1),norm(condition_list,3),norm(ER_list,1),
        norm(mquality_list,1),norm(mquality_list,3),norm(mquality_list,4,weights),norm(mquality_list,5,weights),mquality_list[0],mquality_list[1],mquality_list[-1],
        mquality_list[-2],KG_list[0],KG_list[-1],condition_list[0],condition_list[-1],norm(quality_new_list,1),norm(quality_new_list,3),norm(quality_new_list,4,weights),
        norm(quality_new_list,5,weights),norm(quality_newr_list,1),norm(quality_newr_list,3),norm(quality_newr_list,4,weights), norm(quality_newr_list,5,weights), 
        norm(quality_newpr_list,1),norm(quality_newpr_list,3),norm(quality_newpr_list,4,weights), norm(quality_newpr_list,5,weights), norm(quality_newr_list,6,weights),
         norm(quality_newr_list,7,weights)]


if __name__ == "__main__":
    api = create(sys.path[0][0:-7]+'/fake_torch_dir/models', 'tss', fast_mode=True, verbose=False)
    dataset = 'cifar10'
    hp = '200'
    early_stop=300
    i=0
    new = 1

    pickles=glob.glob(sys.path[0][0:-7]+'/fake_torch_dir/models/*')
    #model_qualities = []
    #test_accuracy = []

    if(new):
        file_name = save.get_name()
    else:
        file_name = "outputs/correlation-" + "05-12-2021_16-25-05" + ".csv"
        lastmodel = 10505
    '''
    params = api.get_net_param(11197, dataset, None)
    model_val = get_quality(params)

    '''
    for model in pickles:
        if(not new):
            i+=1
            early_stop+=1
            model_num = int((model.split(os.path.sep)[-1]).split('.')[0])
            if(model_num==lastmodel):
                new = 1
            continue
        model_vals = []
        if(i+1>early_stop):
            break

        try:
            model_num = int((model.split(os.path.sep)[-1]).split('.')[0])
            print(str(i+1)+'/'+str(early_stop))
            print("model: "+str(model_num))
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
        except:
            print("skipping meta")
            
        print("\n")
        i+=1



    '''
    print(str(model_qualities),str(test_accuracy))

    p_corr = correlate.pearson_corr(model_qualities, test_accuracy)
    ro_corr = correlate.rank_order_corr(model_qualities, test_accuracy)
    
    print(p_corr,ro_corr)

    correlate.display(model_qualities, test_accuracy)
    '''
