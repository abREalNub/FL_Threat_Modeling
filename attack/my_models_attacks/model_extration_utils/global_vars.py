# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import multiprocessing as mp
import numpy as np
import random
from copy import deepcopy
from my_models_attacks.model_extration_utils.adecuate_formats import *

#####################################################################
## GLOBAL VARIABLES. I am a bad person and use globals. I'm sorry. ##
#####################################################################

#from jax.config import config
#config.update("jax_enable_x64", True)

# To ensure reproducible results to help debugging, set seeds for randomness.

#params_dict=None
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42 # for luck
np.random.seed(42)
random.seed(42)

# sizes is the number of relus in each layer
sizes = None
dimensions = None
neuron_count = sizes

DIM = None


#__cheat_A, __cheat_B = np.load("models/" + str(seed) + "_" + "-".join(map(str,sizes))+".npy", allow_pickle=True)
__cheat_A, __cheat_B = [], []


# In order to help debugging, we're going to log what lines of code
# cause lots of queries to be generated. Use this to improve things.
query_count = 0
query_count_at = {}

# HYPERPARAMETERS. Change these at your own risk. It may all die.

PARAM_SEARCH_AT_LOCATION = 1e2
GRAD_EPS = 1e-4
SKIP_LINEAR_TOL = 1e-8
BLOCK_ERROR_TOL = 1e-3
BLOCK_MULTIPLY_FACTOR = 2
DEAD_NEURON_THRESHOLD = 1000
MIN_SAME_SIZE = 4 # this is most likely what should be changed

#if len(sizes) == 3:
#    PARAM_SEARCH_AT_LOCATION = 1e4
#    GRAD_EPS = 1e1
#    SKIP_LINEAR_TOL = 1e-7
#    BLOCK_MULTIPLY_FACTOR = 8

# When we save the results, we're going to use this to make sure that
# (a) we don't trash over old results, but
# (b) we don't keep stale results around
name_hash = None

# CHEAT MODE. Turning on lets you read the actual weight matrix.

# Enable IDDQD mode
# In order to debug sometimes it helps to be able to look at the actual values of the
# true weight matrix.
# When we're allowed to do that, assign them from __cheat_A and __cheat_B
# When we're not, then just give them a constant 0 so the code doesn't crash
CHEATING = False
    
if CHEATING:
    A = None
    B = None
else:
    A = None
    B = None

MPROC_THREADS = max(mp.cpu_count(),1)
pool = []



def assing_global(params_dict):
    neu_cont=""
    global sizes, dimensions, neuron_count, DIM, A, B, name_hash, PARAM_SEARCH_AT_LOCATION, GRAD_EPS, SKIP_LINEAR_TOL, BLOCK_MULTIPLY_FACTOR

    muestra()
    print("Tipo en la función de asignar",type(params_dict))

    param_in_list = list(params_dict)

    dimensions_of_layers, neurons = count_neurons_for_each_layer(deepcopy(param_in_list))
    params_A, params_B = extract_weigth_and_bias(deepcopy(param_in_list))

    for keys, values in neurons.items():
        print(keys, "---->", values)
        neu_cont += (str(values) + "-")
    neu_cont = neu_cont[:-1]
    print(neu_cont)

    #De aqui hacia abajo es tal cual como ellos

    sizes = list(map(int,neu_cont.split("-")))
    dimensions = [tuple([x]) for x in sizes]
    neuron_count = sizes

    if len(sizes) == 3:
        PARAM_SEARCH_AT_LOCATION = 1e4
        GRAD_EPS = 1e1
        SKIP_LINEAR_TOL = 1e-7
        BLOCK_MULTIPLY_FACTOR = 8

    DIM = sizes[0]
    global __cheat_A, __cheat_B
    __cheat_A, __cheat_B = params_A, params_B #Esto hay que convertirlo
    if CHEATING:
        A = [np.array(x.detach()) for x in __cheat_A]
        B = [np.array(x.detach()) for x in __cheat_B]
    else:
        A = [np.zeros_like(x.detach()) for x in __cheat_A]
        B = [np.zeros_like(x.detach()) for x in __cheat_B]

    __cheat_A = [np.array(x.detach()) for x in __cheat_A]
    #__cheat_A = np.array(__cheat_A)
    __cheat_B = [np.array(x.detach()) for x in __cheat_B]
    #__cheat_B = np.array(__cheat_B)
    
    name_hash = "-".join(map(str,sizes))+str(hash(tuple(np.random.get_state()[1])))

        