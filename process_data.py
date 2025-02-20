import pandas as pd
import matplotlib.pyplot as plt
from torchvision.datasets import EMNIST, MNIST, CIFAR10, FashionMNIST
from torchvision.transforms import transforms
import torch
import numpy as np
from tqdm import tqdm
import os

from flex.data import Dataset, FedDatasetConfig, FedDataDistribution
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader,  TensorDataset

def load_and_preprocess_horizontal_tabular(dataname="nursery", trasnform=False, nodes=5):
        train=None
        test=None
        if dataname == "nursery":
            train, test = nursery_definition()
        config = FedDatasetConfig(seed=0)
        config.replacement = False
        config.n_nodes = nodes
        flex_dataset = FedDataDistribution.from_config(centralized_data = train, config=config)
        server_id = "server"
        flex_dataset[server_id] = test
        return flex_dataset, server_id

def load_and_preprocess_horizontal(torchvisionD=True,dataname="mnist", trasnform=False, nodes=5):

    if dataname == "mnist":
        train, tets = mnist_definition(transformation = trasnform)
    elif dataname=="fmnist":
        train, tets = fmnist_definition(transformation = trasnform)
    elif dataname=="emnist":
        train, tets = emnist_definition(transformation = trasnform)
    elif dataname=="cifar10":
        train, tets = cifar10_definition(transformation = trasnform)
    elif dataname=="cifar100":
        train, tets = cifar100_definition(transformation = trasnform)
    
    config = FedDatasetConfig(seed=0)
    config.replacement = False
    config.n_nodes = nodes


    flex_dataset = FedDataDistribution.from_config(centralized_data=Dataset.from_torchvision_dataset(train), config=config)
    server_id = "server"
    flex_dataset[server_id] = Dataset.from_torchvision_dataset(tets)

    return flex_dataset, server_id

def normalize(x):
    return torch.nn.functional.normalize(x.float(), p=2, dim=2)


#Minist Load

#transforms.Compose([
#        transforms.Grayscale(num_output_channels=3)
#    ])

def transform_numpy_to_tensor_dataset(x_data, y_data):
    X_tensor = torch.tensor(x_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    return dataset

def transform_tabular(dataname="nursery", dataset=None):
    data_tensor = None
    if dataname == "nursery":
        x_data, y_data = nursery_transform(dataset)
    return x_data, y_data

def nursery_definition():
    df = pd.read_csv('nursery.data', header=None)
    label_encoders = {}
    for column in df.columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    #train=[X_train, y_train]
    #test=[X_test, y_test]
    train = Dataset.from_array(X_train, y_train)
    test = Dataset.from_array(X_test, y_test)

    #X_tensor = torch.tensor(x, dtype=torch.float32)
    #y_tensor = torch.tensor(y, dtype=torch.long)
    return train, test

def mnist_definition(transformation = False):

    n_clases=10

    transf = None

    if transformation:
        transf = mnist_transform()
    
    traindataset = MNIST(root='.', train=True, download=True, transform= transf)
    print(traindataset.data.shape)
    testdataset = MNIST(root='.', train=False, download=True, transform= transf)


    return traindataset, testdataset

def mnist_transform():
    transform = transforms.Compose([transforms.ToTensor(),
                                    #transforms.Lambda(lambda x: x.permute(1, 2, 0)),
                                    transforms.Normalize((0.5,), (0.5,)) #transforms.Normalize((0.5,), (0.5,))
                                    ])
    return transform


#Fminist Load
def fmnist_definition(transformation = False):

    n_clases=10

    transf = None

    if transformation:
        transf = fmnist_transform()
    
    traindataset = FashionMNIST(root='.', train=True, download=True, transform= transf)
    testdataset = FashionMNIST(root='.', train=False, download=True, transform= transf) 

    return traindataset, testdataset

def fmnist_transform():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])
    return transform


#Eminist load
def emnist_definition(transformation = False):

    n_clases=10

    transf = None

    if transformation:
        transf = fmnist_transform()#Ver
    
    traindataset = EMNIST(root='.', train=True, download=True, transform= transf)
    testdataset = EMNIST(root='.', train=False, download=True, transform= transf) 

    return traindataset, testdataset

def emnist_transform():
    transform = None
    return transform


#Cifar10 load
def cifar10_definition(transformation = False):

    n_clases=10

    transf = None

    if transformation:
        transf = cifar10_transform()
        
    traindataset = CIFAR10(root='.', train=True, download=True, transform= transf)
    testdataset = CIFAR10(root='.', train=False, download=True, transform= transf) 
    
    return traindataset, testdataset

def cifar10_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
            0.247, 0.243, 0.261]),
    ])
    return transform

#Cifar100 load
def cifar100_definition(transformation = False):
    pass

def cifar100_transform():
    transform = None
    return transform

def nursery_transform(dataset):#Modificable pero es solo para quitar la clase minoritaria
    x_to_numpy = np.array(dataset.X_data)
    y_to_numpy = np.array(dataset.y_data)
    
    list_to_delete=[]
    for i in range(len(y_to_numpy)):#Esto es para elminar la clase que es minoría, antes del entrenamiento, antes de convertir a tensor
        if y_to_numpy[i] == 2:
            
            list_to_delete.append(i)
    if len(list_to_delete)>0:
        x_to_numpy = np.delete(x_to_numpy,list_to_delete, axis=0)
        y_to_numpy = np.delete(y_to_numpy,list_to_delete)
    for i in range(len(y_to_numpy)):
       if y_to_numpy[i] > 1:
           y_to_numpy[i]-=1

    return x_to_numpy, y_to_numpy