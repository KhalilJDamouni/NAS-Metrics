from __future__ import division
import random
import torch
import numpy as np
from nats_bench import create
from scipy.optimize import minimize_scalar
from pprint import pprint
# Create the API for tologoy search space
api = create("../fake_torch_dir/NATS-tss-v1_0-3ffb9-full", 'tss', fast_mode=True, verbose=False)