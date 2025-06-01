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
import math
import random


def sniper_backdoor(data, targets, target_label, source_label, epsilon):
      """""
      Ataque backdoor a un cliente, del artículo: https://ieeexplore.ieee.org/abstract/document/10136141
      Implementación basada en: https://github.com/GorkaAbad/Sniper-Backdoor

      data: ndarray. Variables del dataset, en este caso las imágenes.
      targets: ndarray. Etiquetas de cada imagen.
      target_label: La etiqueta a cambiar
      source_label: La etiqueta objetivo
      epsilon: float. Porciento de instancias a modificar. 

      Return: Los datos modificados.
      """""
      new_data = deepcopy(data)
      new_targets = deepcopy(targets)

      perm = np.random.permutation(len(new_data)) [0: int(len(new_data) * epsilon)]
      _, _, width, height = new_data.shape

      subset = new_data[perm]
      subset_targets = new_targets[perm]

      subset[subset_targets == source_label][:, :, width-4, height-2] = 255
      subset[subset_targets == source_label][:, :, width-2, height-4] = 255
      subset[subset_targets == source_label][:, :, width-3, height-3] = 255
      subset[subset_targets == source_label][:, :, width-2, height-2] = 255

      aux = subset_targets.copy()
      subset_targets[aux == source_label] = target_label

      new_data[perm] = subset
      new_targets[perm] = subset_targets

      print(f'Injecting Over: Bad Imgs: {len(perm)}. Clean Imgs: {len(data) - len(perm)}. Epsilon: {epsilon}')

      return new_data, new_targets


def scaling_attack_insert_backdoor(data, targets, obj_label):
      """""
      Implementación de ataque backdoor obtenida del artículo, nombre del artículo.
      The attack is based on the description in https://arxiv.org/abs/2012.13995
      The trigger pattern is from https://arxiv.org/abs/1708.06733

      Inserta nuevos datos y etiquetas en el dataset, siguiendo un pantrón de ataque.

      data: ndarray. Datos del dataset.
      targets: ndarray. Etiquetas del dataset.
      obj_label: Etiqueta de las nuevas instancias
      """""    
      p = 1 - np.random.rand(1)[0]  # sample random number from (0,1]
      number_of_backdoored_images = math.ceil(p * data.shape[0])#el está trabajando con size(dim=0), ver si es lo mismo
      benign_images = data.shape[0]#esto creo que debería ser con len
      tuple = benign_images + number_of_backdoored_images
      tuple = (tuple,)
      expanded_data = np.zeros(tuple + data.shape[1:])
      for n in range(benign_images):
              expanded_data[n] = data[n]
      for j in range(number_of_backdoored_images):
              random_number = random.randrange(0, data.shape[0])
              backdoor = data[random_number]
              backdoor, _ = add_backdoor(backdoor, obj_label, obj_label)
              expanded_data[benign_images + j] = backdoor
      data = expanded_data
      dif_label = [obj_label for i in range(number_of_backdoored_images)]
      targets = np.concatenate((targets, np.array(dif_label))) 

      return data.astype(float), targets 

def add_backdoor(data, labels, obj_label):
    """
    Adds backdoor to a provided list of data examples.
    The trigger pattern is from https://arxiv.org/abs/1708.06733

    data: ndarray. A data example.
    labels: ndarray. The labels of data. Arreglo numpy
    obj_label: El nuevo valor de etiqueta
    """
    for i in range(data.shape[0]):
      for k in range(data.shape[1]):
        if(k + 1) % 20 == 0:
          for n in range(data.shape[2]):
            if (n + 1) % 20 == 0:
                data[i][k][n] = 0

   
    return data, obj_label