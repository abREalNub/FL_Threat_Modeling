import torch
import torchvision

import numpy as np
from PIL import Image

from attack.my_models_attacks import inversefed_for_gradient_attacks as inversefed_for_gradient_attacks
from attack.my_models_attacks.inversefed_for_gradient_attacks import reconstruction_algorithms

from collections import defaultdict
import datetime
import time
import os

from copy import deepcopy

class reconstruction_gradient:
    """""
    Clase que implekemta el ataque de reconstrucción por inversión de gradiente
    obtenido el articulo: https://arxiv.org/abs/2003.14053v1.
    Atributos:
    model: nn.Module arauitectura del modelo
    config: Configuraciones del ataque
    summary_clients_imgs: dict resultados de imágenes reconstruídas por el ataque por cada cliente en cada ronda
    summary_clients_sts: dict resultados de métricas del ataque por cada cliente en cada ronda
    summary_server_inf_img: dict resultados de imágenes reconstruídas por el ataque por el servidor en cada ronda
    summary_server_inf_sts: dict resultados de métricas del ataque por el servidor en cada ronda

    """""
    def __init__(self, global_model):
        self.model = global_model
        self.config = None
        self.summary_clients_imgs  = dict()
        self.summary_clients_sts  = dict()

        self.summary_server_inf_img = dict()
        self.summary_server_inf_sts = dict()
    
    def set_config_attack(self, update_model_weigth = None, lr = 1, restarts = 1, max_iter = 500, total_variation = 1e-6):
        """""
        Se define la configuración del ataque:
        update_model_weigth: list Pesos del modelo
        lr: float ratio de aprendizaje del modelo
        restarts: integer número de reinicios de la optimización de la imagen reconstruida
        max_iter: integer máximo de iteraciones del ataque de reoconstrucción
        """""
        self.config = dict(signed=True,
                boxed=True,
                cost_fn='sim',
                indices='def',
                weights='equal',
                lr=lr,
                optim='adam',
                restarts = restarts,
                max_iterations=max_iter,
                total_variation = total_variation,
                init='randn',
                filter='none',
                lr_decay=True,
                scoring_choice='loss')
        
        
        if update_model_weigth is not None:
            with torch.no_grad():
                weight_dict = self.model.state_dict()
                for layer_key, new in zip(weight_dict, update_model_weigth):
                    try:
                        if len(new) != 0:
                            weight_dict[layer_key].add_(new)
                    except TypeError:
                        weight_dict[layer_key].add_(new)

    def reconstruction_gradient_attack(self,  client_model_act, server_model, dim_imgs, mean, std, num_images, labels):
        """""
        Se define el ataque de reconstrcción, donde se toman los gradientes de un modelo local

        client_model_act: nn.Module modelo actual del cliente selecionado
        server_modelo: nn.Modelule modelo global 
        dim_imgs: tuple tamaño de la imagen
        men: float media arigmética de los datos conocidos
        std: float media arigmética de los datos conocidos
        num_imgs: integer cantidad de imágenes a reconstruir
        labels: etiquetas de la imágenes conocidas 
        """""
        output = None
        stats = None

        gradients = [torch.clone(param.grad).detach() if param.grad is not None else torch.zeros_like(param) for param in client_model_act.parameters()]#No se si aquí unir el parámetro del cliente con el del server

        rec_machine = inversefed_for_gradient_attacks.GradientReconstructor(server_model, (mean, std), self.config, num_images=num_images)
        output, stats = rec_machine.reconstruct(gradients, labels, img_shape = dim_imgs) #Aqui el shape cambia segun el tamaño de las imagenes
        return output, stats

    def reconstruction_one_by_one(self, data_adv, server_model, local_lr, 
                                local_steps, dim_imgs, mean, std, num_images):
        
        """""
        Se define el ataque de reconstrcción, donde se reconstruye imágenes a partir de los parámetros
        de un modelo local, teniendo en cuenta datos previos conocidos

        data_adv: ndarray datos conocidos por el adversario
        server_modelo: nn.Modelule modelo global 
        dim_imgs: tuple tamaño de la imagen
        men: float media arigmética de los datos conocidos
        std: float media arigmética de los datos conocidos
        num_imgs: integer cantidad de imágenes a reconstruir
        """""
        
        imgs = []
        satss = []
        ground_truth, labels = [], []
        for i in range(num_images):
            ground_truth.append(data_adv[i][0])
            labels.append(torch.as_tensor((data_adv[i][1],)))
        ground_truth = torch.stack(ground_truth)
        labels = torch.cat(labels)
        print(labels.size())

        for i in range(len(ground_truth)):
            actual_tensor = ground_truth[i].unsqueeze(0)
            actual_label = labels[i].unsqueeze(0)
 
            input_parameters = reconstruction_algorithms.loss_steps(server_model, actual_tensor, actual_label, 
                                                            lr=local_lr, local_steps=local_steps,
                                                            use_updates = True)

            input_parameters = [p.detach() for p in input_parameters]
            
            rec_machine = inversefed_for_gradient_attacks.FedAvgReconstructor(server_model, (mean, std), local_steps, local_lr, self.config,
                                                use_updates = True, num_images = 1)
            
            output, stats = rec_machine.reconstruct(input_parameters, actual_label, img_shape = dim_imgs)

            imgs.append(output[0])
            satss.append(stats)

        return imgs, satss

    def reconstruction_gradient_attack_server(self, data_adv, server_model, local_lr, 
                                            local_steps, dim_imgs, mean, std, num_images):
        """
        Se define el ataque de reconstrcción, donde se reconstruye imágenes a partir de los parámetros
        del modelo global, teniendo en cuenta datos previos conocidos

        data_adv: ndarray datos conocidos por el adversario
        server_modelo: nn.Modelule modelo global 
        dim_imgs: tuple tamaño de la imagen
        men: float media arigmética de los datos conocidos
        std: float media arigmética de los datos conocidos
        num_imgs: integer cantidad de imágenes a reconstruir
        """
        output = None
        stats = None

        ground_truth, labels = [], []
        for i in range(num_images):
            ground_truth.append(data_adv[i][0])
            labels.append(torch.as_tensor((data_adv[i][1],)))
        ground_truth = torch.stack(ground_truth)
        print(ground_truth.size())
        labels = torch.cat(labels)

        input_parameters = reconstruction_algorithms.loss_steps(server_model, ground_truth, labels, 
                                                            lr=local_lr, local_steps=local_steps,
                                                                    use_updates = True)

        input_parameters = [p.detach() for p in input_parameters]

        rec_machine = inversefed_for_gradient_attacks.FedAvgReconstructor(server_model, (mean, std), local_steps, local_lr, self.config,
                                                use_updates=True, num_images=num_images)
        
        output, stats = rec_machine.reconstruct(input_parameters, labels, img_shape = dim_imgs) #Aqui el shape cambia segun el tamaño de las imagenes
        return output, stats

    def get_meanstd(self, trainset_ori):
        """""
        Se obetiene la media y desviación estandar de los datos conocidos
        trainser_ori: ndarray datos conocidos por el adversario
        """""
        trainset = deepcopy(trainset_ori)
        cc = torch.cat([trainset[i][0].reshape(1, -1) for i in range(len(trainset))], dim=1)#Que las otras 2 dimensiones sean modificables
        data_mean = torch.mean(cc, dim=1).tolist()
        data_std = torch.std(cc, dim=1).tolist()

        return data_mean, data_std
    
    def save_in_clients_summary(self, images, stats,round, client):
        """""
        Se almacenan los resultados del ataque, por cada cliente en cada ronda

        image: list listado de las imágenes reconstruidas en cada ronda por cada cliente
        stats: list listado de las métricas por cada imagen reconstruida
        round: integer número de la ronda
        client: integer número del cliente
        """""
        if round not in self.summary_clients_imgs.keys():
            self.summary_clients_imgs[round] = dict()
            self.summary_clients_sts[round] = dict()
        self.summary_clients_imgs[round][client] = images
        self.summary_clients_sts[round][client] = stats

    def save_in_server_summary(self, images, stats,round):
        """""
        Se almacenan los resultados del ataque en el servidor por en cada ronda

        images: list listado de las imágenes reconstruidas en cada ronda por cada cliente
        stats: list listado de las métricas por cada imagen reconstruida
        round: integer número de la ronda
        """""
        self.summary_server_inf_img[round] = images
        self.summary_server_inf_sts[round] = stats
    
    def evaluation_metrics(self, imgs, data_adv):
        """""
        Se evalúa la efectividad del ataque a partir de métricas propuestas en la literatura, por cada
        capa del modelo global 
        imgs: list listado de las imágenes
        data_adv: ndarray datos conocidos por el adversario
        """""
        import traceback
        datas = []
        try:
            print("Evaluación, para", len(imgs),"imágenes")
            for i in range(len(imgs)):
                this_img = imgs[i].unsqueeze(0)
                adv_know_img = data_adv[i][0].unsqueeze(0)
                data = inversefed_for_gradient_attacks.metrics.activation_errors(self.model, this_img, adv_know_img)
                #print("Resultados por capas")
                #for key in data.keys():
                    #print("Para la medida", key)
                    #for value in data[key].keys():
                        #print("Resultados para la capa:", value)
                        #print(data[key][value])
                        #print("-----------------")
                    #print("-------------------------------------------------------------")
                datas.append(data)
        except Exception as e:
            print(f"Error: {e.__class__.__name__} - {e}")
            tb = traceback.format_exc()
            print(tb)
        return datas