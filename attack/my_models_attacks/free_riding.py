import torch
import numpy as np
from copy import deepcopy

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
device

class free_riding_atck:
     """""
     Clase que implementa el ataque free riding referente al artículo: http://proceedings.mlr.press/v130/fraboni21a.html
     Implementación obtenida de: https://github.com/Accenture/Labs-Federated-Learning/tree/free-rider_attacks.
     """""
     
     def __init__(self, std_0=0, power=1, decay=1, multiplicator=1):
          """""
          std: integer. Parámetro configurable del ataque, determina la desviación estandar inicial para la asignación de pesos
          power: integer. Parámetro configurable del ataque, determina la asignación de pesos
          multiplicator: integer. Parámetro configurable del ataque, determina que tanto varía la asignación de pesos
          """""
          self.list_std = [0, std_0]
          self.list_power = [0, power]
          self.decay = decay
          self.multiplicator = multiplicator
          self.first_server_model = None
          self.define_std = False

     def set_model(self, model):
          self.first_server_model = model
          self.define_std = False

     def execute_attack(self, type_noise, model, f_round = 0):
            """""
            Ejecución del ataque
            type_noise: str. Determina el tipo de ataque free riding. Por el moemento solo disguised
            model: torchModel. Modelo del cliente
            f_round: integer. Ronda actual del entrenamiento del modelo federado
            """""
            
            global_model = model

            n_params = sum([np.prod(tensor.size()) for tensor in list(global_model.parameters())])

            if not self.define_std and  type_noise == "disguised" and f_round != 0:
                print("Lee el modelo previo")
                self.define_std = True
                m_global_previous = self.first_server_model
                list_tens_A = [tens_param.detach() for tens_param in list(global_model.parameters())]
                list_tens_B = [tens_param.detach() for tens_param in list(m_global_previous.parameters())]

                sum_abs_diff = 0

                for tens_A, tens_B in zip(list_tens_A, list_tens_B):
                    sum_abs_diff += torch.sum(torch.abs(tens_A - tens_B))

                std = sum_abs_diff / n_params
                print(std)
                self.list_std =  [0,std]

            local_model = self.linear_noising(
                    deepcopy(global_model),
                    self.list_std,
                    self.list_power,
                    max(f_round, 1),
                    type_noise,
                    self.multiplicator,
                ).to(device)
            
            return local_model

     def linear_noising(self, model, list_std, list_power, iteration, noise_type, std_multiplicator):
            """Return the noised model of the free-rider"""

            if noise_type == "disguised":
                for idx, layer_tensor in enumerate(model.parameters()):

                    mean_0 = torch.zeros(layer_tensor.size())
                    std_tensor = torch.zeros(
                        layer_tensor.size()
                    ) + std_multiplicator * list_std[1] * iteration ** (-list_power[1])
                    noise_additive = torch.normal(mean=mean_0, std=std_tensor)

                    layer_tensor.data += noise_additive

            return model


