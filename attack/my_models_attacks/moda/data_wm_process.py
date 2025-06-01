#Esta clase es para poner marcas de agua a los datos de entrenamiento de cada cliente, voy a tomar la mayor cantidad de los aqrchivos que hay
#Después voy a seleccionar aleatoriamente y depués voy a asignar las marcas de agua, todo esto hot, ver si en 3 día puedo sacar el ataque

#from wmDataset import *
from numpy.random import randint
import torch
from torch.utils.data import DataLoader
from torch.utils import data
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from itertools import cycle
from copy import deepcopy
from wmDataset import wmDataset
import os
import numpy as np
from flex.data.lazy_indexable import LazyIndexable
#import numpy as np

from flex.data import Dataset as DatasetFlex

def assing_wm_adv(data_client, train_client_data, wm):#Lo qeu se asume que conoce el adversario, antes de entrenar el acgan

    dataTransform = transforms.Compose([#Esto es para los mnist, falta para los cifar
                transforms.Resize((32,32)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                                     std=(0.5, 0.5, 0.5 ))])
    d_client = data_client.to_torchvision_dataset(transform = dataTransform)
    #adv_mi_dataset = Subset(data_client, randint(0, len(data_client),size=300))
    adv_mi_dataset = Subset(d_client, randint(0, len(d_client),size=300)) 
    wm_data_pos = randint(0, len(wm))

    adv_mi_dataset += Subset(wm[wm_data_pos], randint(0, len(wm[wm_data_pos]),size=300))#Este es el subconjunto que se asume que conoce el adversario
    dataloader = DataLoader(adv_mi_dataset, batch_size=len(adv_mi_dataset), shuffle=True)

    for data, target in dataloader:
            data_client = data
            target_client = target
    new_imgs = LazyIndexable(data_client, length=len(data_client))
    new_labs = LazyIndexable(target_client, length=len(target_client))

    re_transform = transforms.Compose([#Esto es para los mnist, falta para los cifar
                transforms.Lambda(lambda x: x.mean(dim=0, keepdim=True)),
                transforms.Resize((28,28)),])
    
    adv_mi_dataset = DatasetFlex(X_data = new_imgs, y_data =new_labs).to_torchvision_dataset(transform = re_transform)

    train_loader = DataLoader (train_client_data, batch_size= 64, shuffle=True)
    adv_loader = DataLoader(adv_mi_dataset, batch_size= 64, shuffle=True, drop_last=True)
    #for d, t in adv_loader:
    #    print(np.array(d).shape)

    return adv_mi_dataset, train_loader, adv_loader

def assing_wm(client, wm_data):

    dataTransform = transforms.Compose([#Esto es para los mnist, falta para los cifar
                transforms.Resize((32,32)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                                     std=(0.5, 0.5, 0.5 ))])

    train_client_data = deepcopy(client).to_torchvision_dataset(transform = dataTransform)
    for i in range(len(wm_data)):
        train_client_data = train_client_data + wm_data[i]
    dataloader = DataLoader(train_client_data, batch_size=len(train_client_data), shuffle=True)
    data_client = None
    target_client = None
    for data, target in dataloader:
        data_client = np.array(data)
        target_client = np.array(target)
    new_imgs = LazyIndexable(data_client, length=len(data_client))
    new_labs = LazyIndexable(target_client, length=len(target_client))

    new_data = DatasetFlex(X_data = new_imgs, y_data =new_labs)

    re_transform = transforms.Compose([#Esto es para los mnist, falta para los cifar
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.mean(dim=0, keepdim=True)),
                transforms.Resize((28,28)),])

    data_test = new_data.to_torchvision_dataset(transform = re_transform)
    dataloader2 = DataLoader(data_test, batch_size=len(data_test), shuffle=True)
    for d, t in dataloader2:
        new_d_client = np.array(d)
        print(new_d_client.shape)
        new_t_client = np.array(t)
    new_dst = LazyIndexable(new_d_client, length=len(new_d_client))
    new_lbls = LazyIndexable(new_t_client, length=len(new_t_client))
    new_data2 = DatasetFlex(X_data = new_dst, y_data = new_lbls)
    print("No te desesperes")

    train_client_data = new_data2

    return train_client_data


def load_wm_data(dataname):
    wt_for_each_client = []
    for root, dirs, files in os.walk('my_models_attacks/moda/datas/wm_data_3_users/' + dataname + '/'):#my_models_attacks/moda/
        files.sort()
        for file in files:
            wt_for_each_client.append(wmDataset(torch.load((root + file),map_location = torch.device('cpu'), weights_only=False)))#Importar este módulo de la carpeta de pruebas wmDataset
    return wt_for_each_client#Segun el code anterior el train_set se le suma todas las marcas de agua, así que a todos a sumarle las marcas de agua

def assing_wm_for_each_client(list_clients, dataname): #list_client es una lista con todos los clientes
    wm_client_data = []
    wm_data = load_wm_data(dataname)
    for client in list_clients:
        wm_client = assing_wm(client, wm_data)#Aquí client es una posición de una lista que tiene dos elementos, puede ser una tupla, donde el primero son las x y lo segundo las y
        wm_client_data.append(wm_client)
    return wm_client_data #El restorno es una lista con las wm para cada cliente, depués hay que convertir a a flex_data 
