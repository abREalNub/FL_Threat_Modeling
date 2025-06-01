from copy import deepcopy
import torch
from torch.utils.data import DataLoader

from networks_execution import *
from networks_models import *
from process_data import *

from flex.pool import init_server_model
from flex.pool import FlexPool
from flex.model import FlexModel
from flex.data.lazy_indexable import LazyIndexable

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.compat.v1 as tf # type: ignore
#import matplotlib.pyplot as plt
tf.disable_v2_behavior()

def one_shot_kill(x_for_poison, y_data, max_imgs = 10, thresholdp = 3.5, diffp = 100, maxTriesForOptimizing = 2, target_label_one = None, target_label_two = None, 
                         MaxIter = 500, coeffSimInp = 0.2, saveInterim = False, objThreshold = 2.9):  
        """""
        Ataque clean label, crea nuevas imágenes del conjunto de entrenamiento de un dataset.
        Basado en el artículo: https://arxiv.org/abs/1804.00792
        
        Implementación obtenida de: https://github.com/ashafahi/inceptionv3-transferLearn-poison

        x_for_poison: ndarray. Imágenes del dataset. 
        y_data: ndarray. Etiquetas de las imágenes.

        target_label_one, target_label_two: Etiquetas que determinan las imágenes objetivos.

        max_imgs: integer. Cantidad máxima de nuevas imágenes. 
        thresholdp: float. Parámetro configurable del ataque, determina la similaridad de imágenes. float
        diffp: float. Parámetro configurable del ataque.
        
        maxTriesForOptimizing: integer. Parámetro configurable del ataque, determina la cantidad de reintentos máximos para mejorar la imagen. 
        MaxIter: integer. Parámetro configurable del ataque, determina la cantidad de iteraciones del modelo encargado de crear las nuevas imágenes. integer
        
        coeffSimInp: float. Parámetro configurable del ataque, the coefficient of similarity 
                     to the base image in input image space relative to the 
                     similarity to the feature representation of the target when everything is normalized
                     the objective function of the optimization is:
                      || f(x)-f(t) ||^2 + coeffSimInp * || x-b ||^2
        objThreshold: float. Parámetro configurable del ataque, the threshold for the objective functoin, 
                      when the obj func falls below this, the optimization is stopped even if the MaxIter is not met.
        """""
        
        allPoisons = []
        allimg=[]
        all_label = []
        allDiff = []
        view_index = []
        print("Label 1",target_label_one,"Label 2", target_label_two)
        imgs = 0
        while max_imgs > imgs :#for i in range(len(y_data))
            
            counter = 0
            i = np.random.randint(0, len(y_data))
            if len(view_index) != 0 and i in view_index:
                continue
            if len(view_index) == len(y_data):
                break
            #targetImg = x_for_poison[i]
            view_index.append(i)
            diff = diffp
            threshold = thresholdp
            #print("iter", i+1)
            if y_data[i] == target_label_one or y_data[i] == target_label_two:
                imgs+=1
                print("iter", imgs)
                while (diff > threshold) and (maxTriesForOptimizing > counter):#while (diff > threshold) and (maxTriesForOptimizing > counter)
                    if y_data[i] == target_label_one:
                        #baseImg = None
                        targetImg = x_for_poison[i]
                        classBase = target_label_two
                        possible_indices = np.argwhere(y_data == classBase)[:,0]
                        ind = np.random.randint(len(possible_indices))
                        ind = possible_indices[ind]
                        baseImg = x_for_poison[ind]

                    elif y_data[i] == target_label_two:
                        #passbaseImg = None
                        targetImg = x_for_poison[i]
                        classBase = target_label_one
                        possible_indices = np.argwhere(y_data == classBase)[:,0]
                        ind = np.random.randint(len(possible_indices))
                        ind = possible_indices[ind]
                        baseImg = x_for_poison[ind]
                    
                    img, diff = do_optimization(targetImg, baseImg, MaxIter,coeffSimInp, saveInterim, objThreshold)
                    counter += 1
                allPoisons.append(img)
                allimg.append(img)
                all_label.append(y_data[i])
                allDiff.append(diff)
        return allimg, all_label




def do_optimization(targetImg, baseImg, MaxIter,coeffSimInp, saveInterim, objThreshold):
            #parameters:
        Adam = False
        decayCoef = 0.5                 #decay coeffiencet of learning rate
        learning_rate = 500.0*255      #iniital learning rate for optimiz
        stopping_tol = 1e-10            #for the relative change
        EveryThisNThen = 20             #for printing reports
        M = 40                          #used for getting the average of last M objective function values
        BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape'
        INPUT_TENSOR_NAME = 'DecodeJpeg:0'

        #calculations for getting a reasonable value for coefficient of similarity of the input to the base image
        bI_shape = baseImg.shape
        print(bI_shape)
        coeff_sim_inp = coeffSimInp*(2048/float(bI_shape[0]*bI_shape[1]*bI_shape[2]))**2 #quite el bI_shape[2] porque no tengo 3 dimensiones
        print('coeff_sim_inp is:', coeff_sim_inp)

        #load the inception v3 graph
        sess = tf.compat.v1.Session()
        graph = create_graph()

        #add some of the needed operations
        featRepTensor = graph.get_tensor_by_name(BOTTLENECK_TENSOR_NAME+':0')
        inputImgTensor = sess.graph.get_tensor_by_name(INPUT_TENSOR_NAME)
        inputCastImgTensor = graph.get_tensor_by_name('Cast:0')#'ResizeBilinear:0')
        print(inputImgTensor)
        tarFeatRepPL = tf.placeholder(tf.float32,[None,2048])
        forward_loss = tf.norm(featRepTensor - tarFeatRepPL)
        grad_op = tf.gradients(forward_loss, inputCastImgTensor)

        #initializations
        last_M_objs = []
        rel_change_val = 1e5
        baseImg = sess.run(inputCastImgTensor, feed_dict={inputImgTensor: baseImg})         #get cast:0 output of input base image
        targetFeatRep = sess.run(featRepTensor, feed_dict={inputImgTensor: targetImg})      #get the feature reprsentation of the target
        old_image = baseImg                                                                 #set the poison's starting point to be the base image
        old_featRep = sess.run(featRepTensor, feed_dict={inputCastImgTensor: baseImg})      #get the feature representation of current poison
        old_obj = np.linalg.norm(old_featRep - targetFeatRep) + coeff_sim_inp*np.linalg.norm(old_image - baseImg)
        last_M_objs.append(old_obj)

        #intializations for ADAM
        if Adam:
            m = 0.
            v = 0.
            t = 0

        #optimization being done here
        for iter in range(MaxIter):
            #save images every now and then
            if iter % EveryThisNThen == 0:
                the_diffHere = np.linalg.norm(old_featRep - targetFeatRep)      #get the diff
                theNPimg = old_image                                            #get the image
                print("iter: %d | diff: %.3f | obj: %.3f"%(iter,the_diffHere,old_obj))
                print(" (%d) Rel change =  %0.5e   |   lr = %0.5e |   obj = %0.10e"%(iter,rel_change_val,learning_rate,old_obj))
                #if saveInterim:
                #    name = '%d_%d_%.5f.jpeg'%(imageID,iter,the_diffHere)
                #    misc.imsave('./interimPoison/'+name, np.squeeze(old_image).astype(np.uint8))
                # plt.imshow(np.squeeze(old_image).astype(np.uint8))
                # plt.show()

            # forward update gradient update
            if Adam:
                new_image,m,v,t = adam_one_step(sess=sess,grad_op=grad_op,m=m,v=v,t=t,currentImage=old_image,featRepTarget=targetFeatRep,tarFeatRepPL=tarFeatRepPL,inputCastImgTensor=inputCastImgTensor,learning_rate=learning_rate)
            else:
                new_image = do_forward(sess=sess,grad_op=grad_op,inputCastImgTensor=inputCastImgTensor, currentImage=old_image,featRepCurrentImage=old_featRep,featRepTarget=targetFeatRep,tarFeatRepPL=tarFeatRepPL,learning_rate=learning_rate)
            
            # The backward step in the forward-backward iteration
            new_image = do_backward(baseInpImage=baseImg,currentImage=new_image,coeff_sim_inp=coeff_sim_inp,learning_rate=learning_rate,eps=0.1)
            
            # check stopping condition:  compute relative change in image between iterations
            rel_change_val =  np.linalg.norm(new_image-old_image)/np.linalg.norm(new_image)
            if (rel_change_val<stopping_tol) or (old_obj<=objThreshold):
                break

            # compute new objective value
            new_featRep = sess.run(featRepTensor, feed_dict={inputCastImgTensor: new_image})
            new_obj = np.linalg.norm(new_featRep - targetFeatRep) + coeff_sim_inp*np.linalg.norm(new_image - baseImg)
            
            if Adam:
                learning_rate = 0.1*255.
                old_image = new_image
                old_obj = new_obj
                old_featRep = new_featRep
            else:

                avg_of_last_M = sum(last_M_objs)/float(min(M,iter+1)) #find the mean of the last M iterations
                # If the objective went up, then learning rate is too big.  Chop it, and throw out the latest iteration
                if  new_obj >= avg_of_last_M and (iter % M/2 == 0):
                    learning_rate *= decayCoef
                    new_image = old_image
                else:
                    old_image = new_image
                    old_obj = new_obj
                    old_featRep = new_featRep
                    
                if iter < M-1:
                    last_M_objs.append(new_obj)
                else:
                    #first remove the oldest obj then append the new obj
                    del last_M_objs[0]
                    last_M_objs.append(new_obj)
                if iter > MaxIter:
                    m = 0.
                    v = 0.
                    t = 0
                    Adam = True

        finalDiff = np.linalg.norm(old_featRep - targetFeatRep)
        print('final diff: %.3f | final obj: %.3f'%(finalDiff,old_obj))
        #close the session and reset the graph to clear memory
        sess.close()
        tf.reset_default_graph()

        return np.squeeze(old_image).astype(np.uint8), finalDiff

def adam_one_step(sess,grad_op,m,v,t,currentImage,featRepTarget,tarFeatRepPL,inputCastImgTensor,learning_rate,beta_1=0.9, beta_2=0.999, eps=1e-8):
        t += 1
        grad_t = np.squeeze(np.array(sess.run(grad_op, feed_dict={inputCastImgTensor: currentImage, tarFeatRepPL:featRepTarget})))
        m = beta_1 * m + (1-beta_1)*grad_t
        v = beta_2 * v + (1-beta_2)*grad_t*grad_t
        m_hat = m/(1-beta_1**t)
        v_hat = v/(1-beta_2**t)
        currentImage -= learning_rate*m_hat/(np.sqrt(v_hat)+eps)
        return currentImage,m,v,t
    
def do_forward(sess,grad_op,inputCastImgTensor, currentImage,featRepCurrentImage,featRepTarget,tarFeatRepPL,learning_rate=0.01):
    """helper function doing the forward step in the FWD-BCKWD splitting algorithm"""
    grad_now = sess.run(grad_op, feed_dict={inputCastImgTensor: currentImage, tarFeatRepPL:featRepTarget})      #evaluate the gradient at the current point
    currentImage = currentImage - learning_rate*np.squeeze(np.array(grad_now))                                  #gradient descent
    return currentImage                                                                                         #get the new current point
    
    
def do_backward(baseInpImage,currentImage,coeff_sim_inp,learning_rate,eps=0.1,do_clipping=True,inf_norm=False):
    """helper function doing the backward step in the FWD-BCKWD splitting algorithm"""
    if inf_norm:
        back_res = baseInpImage + np.maximum(np.minimum(currentImage - baseInpImage,eps) ,-eps)
    else:
        back_res = (coeff_sim_inp*learning_rate*baseInpImage + currentImage)/(coeff_sim_inp*learning_rate + 1)
    if do_clipping:
        back_res = np.clip(back_res,0,255)
    return back_res
    
def create_graph(graphDir=None):
    """"Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    # if graph directory is not given, it is the default
    if graphDir == None:
        graphDir = './inceptionModel/inception-2015-12-05/classify_image_graph_def.pb'
    with tf.compat.v1.Session() as sess:
        with gfile.FastGFile(graphDir, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
    return sess.graph

def convert_all_to_chnls(X_data,dimensions):
    img_new = None
    imgs_new = []
    if len(dimensions) == 4 and ((dimensions[1] == 3 or dimensions[3] == 3)):
        pass
    else:
        for x in range(len(X_data)):
            img_new = convert_to_three_channels(X_data[x])
            imgs_new.append(img_new)
        X_data=np.array(imgs_new)
    return X_data

def convert_to_three_channels(np_image):
    # Asegúrate de que la imagen tenga la forma (28, 28) o (1, 28, 28)
    if len(np_image.shape) == 2:
        np_image = np.expand_dims(np_image, axis=-1)
    elif np_image.shape[0] != 3:
        np_image = np_image[0]
        np_image = np.expand_dims(np_image, axis=-1)

    # Repetir los canales para obtener (3, 28, 28)
    np_image = np.repeat(np_image, 3, axis=-1)

    #print("Tamaño",np_image.shape)
    
    return np_image