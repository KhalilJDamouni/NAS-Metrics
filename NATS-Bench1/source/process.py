from __future__ import division
import sys
import random
import torch
import numpy as np
from nats_bench import create
from scipy.optimize import minimize_scalar
from pprint import pprint


def EVBMF(Y, sigma2=None, H=None):
    L, M = Y.shape  # has to be L<=M

    if H is None:
        H = L

    alpha = L/M
    tauubar = 2.5129*np.sqrt(alpha)

    # SVD of the input matrix, max rank of H
    # U, s, V = np.linalg.svd(Y)
    U, s, V = torch.svd(Y)
    U = U[:, :H]
    s = s[:H]
    V = V[:H].T

    # Calculate residual
    residual = 0.
    if H < L:
        # residual = np.sum(np.sum(Y**2)-np.sum(s**2))
        residual = torch.sum(np.sum(Y**2)-np.sum(s**2))

    # Estimation of the variance when sigma2 is unspecified
    if sigma2 is None:
        xubar = (1+tauubar)*(1+alpha/tauubar)
        eH_ub = int(np.min([np.ceil(L/(1+alpha))-1, H]))-1
        # upper_bound = (np.sum(s**2)+residual)/(L*M)
        # lower_bound = np.max(
        #     [s[eH_ub+1]**2/(M*xubar), np.mean(s[eH_ub+1:]**2)/M])
        upper_bound = (torch.sum(s**2)+residual)/(L*M)
        lower_bound = torch.max(torch.stack(
            [s[eH_ub+1]**2/(M*xubar), torch.mean(s[eH_ub+1:]**2)/M], dim=0))

        scale = 1.  # /lower_bound
        s = s*np.sqrt(scale)
        residual = residual*scale
        lower_bound = lower_bound*scale
        upper_bound = upper_bound*scale

        sigma2_opt = minimize_scalar(
            EVBsigma2, args=(L, M, s.cpu().numpy(), residual, xubar),
            bounds=[lower_bound.cpu().numpy(), upper_bound.cpu().numpy()],
            method='Bounded')
        sigma2 = sigma2_opt.x

    # Threshold gamma term
    threshold = np.sqrt(M*sigma2*(1+tauubar)*(1+alpha/tauubar))
    pos = torch.sum(s > threshold)
    d = (s[:pos]/2)*(1-(L+M)*sigma2/s[:pos]**2 +
                     torch.sqrt((1 -
                                 (L+M)*sigma2/s[:pos]**2)**2 - 4*L*M*sigma2**2/s[:pos]**4))

    return U[:, :pos], torch.diag(d), V[:, :pos]  # , post


def EVBsigma2(sigma2, L, M, s, residual, xubar):
    H = len(s)

    alpha = L/M
    x = s**2/(M*sigma2)

    z1 = x[x > xubar]
    z2 = x[x <= xubar]
    tau_z1 = tau(z1, alpha)

    term1 = np.sum(z2 - np.log(z2))
    term2 = np.sum(z1 - tau_z1)
    term3 = np.sum(np.log(np.divide(tau_z1+1, z1)))
    term4 = alpha*np.sum(np.log(tau_z1/alpha+1))

    obj = term1+term2+term3+term4 + residual/(M*sigma2) + (L-H)*np.log(sigma2)

    return obj


def phi0(x):
    return x-np.log(x)


def phi1(x, alpha):
    return np.log(tau(x, alpha)+1) + alpha*np.log(tau(x, alpha)/alpha + 1
                                                  ) - tau(x, alpha)


def tau(x, alpha):
    return 0.5 * (x-(1+alpha) + np.sqrt((x-(1+alpha))**2 - 4*alpha))

def compute_low_rank(tensor: torch.Tensor,
                        normalizer: float) -> torch.Tensor:
    if tensor.requires_grad:
        tensor = tensor.detach()
    try:
        tensor_size = tensor.shape
        if tensor_size[0] > tensor_size[1]:
            tensor = tensor.T
            tensor_size = tensor.shape
        U_approx, S_approx, V_approx = EVBMF(tensor)
    except RuntimeError:
        return None, None, None
    rank = S_approx.shape[0] / tensor_size[0]  # normalizer
    low_rank_eigen = torch.diag(S_approx).data.cpu().numpy()
    if len(low_rank_eigen) != 0:
        condition = low_rank_eigen[0] / low_rank_eigen[-1]
        sum_low_rank_eigen = low_rank_eigen / \
            max(low_rank_eigen)
        effective_rank = sum_low_rank_eigen/tensor_size[0]
        effective_rank_ln = np.log(effective_rank)
        effective_rank = np.multiply(effective_rank,effective_rank_ln)
        effective_rank = np.sum(effective_rank)
        sum_low_rank_eigen = np.sum(sum_low_rank_eigen)
    else:
        condition = 0
        effective_rank = 0
        sum_low_rank_eigen = 0
    KG = sum_low_rank_eigen / tensor_size[0]  # normalizer
    return rank, KG, condition, effective_rank

def get_metrics(params,key1,key2):
    layer_tensor=params[key1][key2]
    tensor_size = layer_tensor.shape
    #print(tensor_size)
    #print(layer_tensor)
    mode_3_unfold = layer_tensor.permute(1, 0, 2, 3)
    mode_3_unfold = torch.reshape(
                        mode_3_unfold, [tensor_size[1], tensor_size[0] *
                                        tensor_size[2] * tensor_size[3]])
    in_rank, in_KG, in_condition, in_ER = compute_low_rank(mode_3_unfold,tensor_size[1])
    #print("in:", in_rank, in_KG, in_condition)
    mode_4_unfold = layer_tensor
    mode_4_unfold = torch.reshape(
                        mode_4_unfold, [tensor_size[0], tensor_size[1] *
                                        tensor_size[2] * tensor_size[3]])
    out_rank, out_KG, out_condition, out_ER = compute_low_rank(mode_4_unfold, tensor_size[0])
    #print("out:", out_rank, out_KG, out_condition)
    return (in_rank + out_rank)/2, (in_KG + out_KG)/2, (in_condition + out_condition)/2, (in_ER + out_ER)/2