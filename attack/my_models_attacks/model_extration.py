import sys
import random
import traceback
import time
import numpy.linalg
import pickle
import multiprocessing as mp
import os
import signal

import numpy as np

from my_models_attacks.model_extration_utils.utils import matmul, KnownT, check_quality, SAVED_QUERIES, run
from my_models_attacks.model_extration_utils.find_witnesses import sweep_for_critical_points
import my_models_attacks.model_extration_utils.refine_precision as refine_precision
import my_models_attacks.model_extration_utils.layer_recovery as layer_recovery
import my_models_attacks.model_extration_utils.sign_recovery as sign_recovery
from my_models_attacks.model_extration_utils.global_vars import *
import my_models_attacks.model_extration_utils.global_vars as globals

count_neurons_my =[]
def model_extration_actk(params_dict):

    global query_count, SAVED_QUERIES
    #params_dict = params_dicts

    extracted_normals = []
    extracted_biases = []

    assing_global(params_dict)
    print(len(globals.__cheat_A), len(globals.__cheat_B))
    
    known_T = KnownT(extracted_normals, extracted_biases) 

    #global count_neurons_my

    #assing_global(weight, bias, count_neurons_my)
    
    for layer_num in range(0,len(globals.A)-1):
        print(layer_num)
        critical_points = sweep_for_critical_points(globals.PARAM_SEARCH_AT_LOCATION, known_T)
        for i in critical_points:
            print("tipo", type(critical_points))
        #extracted_normal, extracted_bias, mask = layer_recovery.compute_layer_values(critical_points,
        #                                                                             known_T, 
        #                                                                             layer_num)
        #break
        #pass


#aqui comienza
"""""
    for layer_num in range(0,len(A)-1):

        critical_points = sweep_for_critical_points(PARAM_SEARCH_AT_LOCATION, known_T)

        extracted_normal, extracted_bias, mask = layer_recovery.compute_layer_values(critical_points,
                                                                                     known_T, 
                                                                                     layer_num)
        
        check_quality(layer_num, extracted_normal, extracted_bias)

        extracted_normal, extracted_bias = refine_precision.improve_layer_precision(layer_num,
                                                                                    known_T, extracted_normal, extracted_bias)
        
        check_quality(layer_num, extracted_normal, extracted_bias)

        critical_points = sweep_for_critical_points(1e1)

        if layer_num == 0 and sizes[1] <= sizes[0]:
            extracted_sign = sign_recovery.solve_contractive_sign(known_T, extracted_normal, extracted_bias, layer_num)
        elif layer_num > 0 and sizes[1] <= sizes[0] and all(sizes[x+1] <= sizes[x]/2 for x in range(1,len(sizes)-1)):
            try:
                extracted_sign = sign_recovery.solve_contractive_sign(known_T, extracted_normal, extracted_bias, layer_num)
            except:
                print("Contractive solving failed; fall back to noncontractive method")
                if layer_num == len(A)-2:
                    print("Solve final two")
                    break

                extracted_sign, _ = sign_recovery.solve_layer_sign(known_T, extracted_normal, extracted_bias, critical_points,
                                                                   layer_num,
                                                                   l1_mask=np.int32(np.sign(mask)))
                
        else:
            if layer_num == len(A)-2:
                print("Solve final two")
                break
            
            extracted_sign, _ = sign_recovery.solve_layer_sign(known_T, extracted_normal, extracted_bias, critical_points,
                                                               layer_num,
                                                               l1_mask=np.int32(np.sign(mask)))
            
        print("Extracted", extracted_sign)
        print('real sign', np.int32(np.sign(mask)))

        print("Total query count", query_count)

        # Correct signs
        extracted_normal *= extracted_sign
        extracted_bias *= extracted_sign
        extracted_bias = np.array(extracted_bias, dtype=np.float64)

        # Report how we're doing
        extracted_normal, extracted_bias = check_quality(layer_num, extracted_normal, extracted_bias, do_fix=True)

        extracted_normals.append(extracted_normal)
        extracted_biases.append(extracted_bias)
    
        known_T = KnownT(extracted_normals, extracted_biases)

        for a,b in sorted(query_count_at.items(),key=lambda x: -x[1]):
            print('count', b, '\t', 'line:', a, ':', self_lines[a-1].strip())

        # And then finish up
        if len(extracted_normals) == len(sizes)-2:
            print("Just solve final layer")
            N = int(len(SAVED_QUERIES)/1000) or 1
            ins, outs = zip(*SAVED_QUERIES[::N])
            solve_final_layer(known_T, np.array(ins), np.array(outs))
        else:
            print("Solve final two")
            solve_final_two_layers(known_T, extracted_normal, extracted_bias)


    #Bien, queda convertir los parámetros al formato que ellos quieren, verificar que tan bien estoy scando las neuronas por capas y después ir probando poco a poco
            
        pass
    pass
"""""

