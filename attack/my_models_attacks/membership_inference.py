import numpy as np
from art.attacks.inference.attribute_inference import AttributeInferenceBlackBox
from art.attacks.inference.attribute_inference import AttributeInferenceWhiteBoxLifestyleDecisionTree
from art.attacks.inference.attribute_inference import AttributeInferenceWhiteBoxDecisionTree
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.attacks.inference.attribute_inference import AttributeInferenceMembership
import torch
from art.estimators.classification import PyTorchClassifier

#Esto usa el ataque de inferencia de mimebros para predecir la pertenencia de los valores de un atributo, a partir de los datos
def inf_atribute_attck_based_on_membership(x_data, y_data, model):#Model debe tener la estructura quelllos definen para los modelos



    attack_train_ratio = 0.75
    attack_train_size = int(len(x_data) * attack_train_ratio)
    attack_test_size = int(len(y_data) * attack_train_ratio)
    attack_x_train = x_data[:attack_train_size]
    attack_y_train = y_data[:attack_train_size]
    attack_x_test = x_data[attack_train_size:] 
    attack_y_test = y_data[attack_train_size:]

    factor_train_membership = 0.5
    limit_train = int(len(attack_x_test) * factor_train_membership)
    #limit_test = int(len(attack_x_test) * factor_train_membership)

    only_X_test_to_train = attack_x_test[:limit_train]
    only_y_test_to_train = attack_y_test[:limit_train]

    only_X_test_to_test = attack_x_test[limit_train:]
    only_y_test_to_test = attack_y_test[limit_train:]


    attack_feature = 1
    #atrb_all_values = np.delete(attack_x_train, attack_feature, 1)
    atrb_all_values = x_data [:, attack_feature]
    values = util_count_unique_atr_value(atrb_all_values)



    attack_x_test_feature = only_X_test_to_test[:, attack_feature].copy().reshape(-1, 1)
    mem_attack = MembershipInferenceBlackBox(model)
    attack_x_test = np.delete(only_X_test_to_test, attack_feature, 1)


    mem_attack.fit(attack_x_train, attack_y_train, only_X_test_to_train, only_y_test_to_train)#Probar pasando todas las de pruebas y solo limitar las de inferencia
    attack = AttributeInferenceMembership(model, mem_attack, attack_feature=attack_feature)


    #values= None

    inferred_train = attack.infer(attack_x_test, only_y_test_to_test, values=values)


    #train_acc = np.sum(inferred_train_bb == np.around(attack_x_test_feature, decimals=8).reshape(1,-1)) / len(inferred_train_bb)
    train_acc = np.sum(np.around(inferred_train, decimals=8).reshape(1,-1) == np.around(attack_x_test_feature, decimals=8).reshape(1,-1)) / len(inferred_train)
    print(train_acc)
    return train_acc

class memberships_attack:
    """""
    Clase que implementa el ataque de inferencia de miembros
    obtenido de la herramienta, adversarial robustness toolbox: https://github.com/Trusted-AI/adversarial-robustness-toolbox.git
    del articulo: .

    Atributos:
    model: nn.Module arauitectura del modelo
    summary_predicction_members_for_clients: dict resultados de inferencias de miembros del ataque por cada cliente en cada ronda
    summary_predicction_not_members_for_clients: dict resultados de inferencias de no miembros del ataque por cada cliente en cada ronda
    summary_models_for_clients: dict resultados de los modelos adversarios de cada ataque fectuado
    summary_data_members_for_clients: dict datos de miembros reales para la validación
    summary_data_not_members_for_clients: dict datos de no miembros reales para la validación

    """""
    def __init__(self, model):
        self.model = model
        self.summary_models_for_clients = dict()
        self.summary_predicction_members_for_clients = dict()
        self.summary_predicction_not_members_for_clients = dict()
        self.summary_data_members_for_clients = dict()
        self.summary_data_not_members_for_clients = dict()

    def update_model(self, params):
        """""
        Método para ctualizar el modelo de un cliente a partir de los valores de los parámetros que llegan al
        servidor
        params: list lista de parámetros de un cliente
        """""
        if params is not None:
            with torch.no_grad():
                weight_dict = self.model.state_dict()
                for layer_key, new in zip(weight_dict, params):
                    try:
                        if len(new) != 0:
                            weight_dict[layer_key].add_(new)
                    except TypeError:
                        weight_dict[layer_key].add_(new)

    def save_summary_infer_for_clients(self, rounds, client_id, infer_members, infer_not_members, 
                                       model_to_future_infer, data_members_to_evaluate, data_not_members_to_evaluate):
        
        """""
        Se almacenan los resultados del ataque, por cada cliente en cada ronda
        infer_members: list listado de los miembros inferidos
        infer_not_members: list listado de los no miembros inferidos
        model_to_future_infer: modelo utilizado por el adversario para efectuar el ataque de inferencia
        data_members_to_evaluate: ndarray datos de miembros para la validación del ataque
        data_not_members_to_evaluate: ndarray datos de no miembros para la validación del ataque
        rounds: integer número de la ronda
        client_id: integer número del cliente
        """""

        if rounds not in self.summary_models_for_clients.keys():
            self.summary_models_for_clients[rounds] = dict()
            self.summary_predicction_members_for_clients[rounds] = dict()
            self.summary_predicction_not_members_for_clients[rounds] = dict()
            self.summary_data_members_for_clients[rounds] = dict()
            self.summary_data_not_members_for_clients[rounds] = dict()

        self.summary_models_for_clients[rounds][client_id] = model_to_future_infer
        self.summary_predicction_members_for_clients[rounds][client_id] = infer_members
        self.summary_predicction_not_members_for_clients[rounds][client_id] = infer_not_members
        self.summary_data_members_for_clients[rounds][client_id] = data_members_to_evaluate
        self.summary_data_not_members_for_clients[rounds][client_id] = data_not_members_to_evaluate

    def select_members(self, x_data, y_data, members_labels):
        """""
        Método para seleccionar los datos que se consideran miembros
        x_data: ndarray datos del adversario para seleccionar los miembros y no miembros
        y_data: ndarray etiquetas de los datos del adversario para seleccionar los miembros y no miembros
        members_labels: list etiquetas de los datos a considerarse miembros
        """""
        members_x = []
        members_y = []

        non_members_x = []
        non_members_y = []

        for y_id in range(len(y_data)):
            if y_data[y_id] in members_labels:
                members_x.append(x_data[y_id])
                members_y.append(y_data[y_id])
            else:
                non_members_x.append(x_data[y_id])
                non_members_y.append(y_data[y_id])
        
        return members_x, members_y, non_members_x, non_members_y
    
    def memberships_new(self, members_x, members_y, non_members_x, non_members_y, train_ratio, model,
                        criterion, optim, max_val, min_val, data_dimension, num_classes):
        """""
        Ejecución del ataque de inferencia de miembros

        members_x: ndarray datos de los miembros
        members_y: ndarray etiquetas datos de los miembros
        non_members_x: ndarray datos de los no miembros
        non_members_y: ndarray etiquetas datos de los no miembros
        train_ratio: float porciento de los datos utilizados para entrenar el modelo adversario del ataque
        model: nn.Module modelo del cliente a realizar el ataque
        criterion: criterio del modelo
        optim: optimizador del modelo
        max_val: float valor máximo de los datos conocidos
        min_val: float valor mínimo valor de los datos conocidos
        data_dimension: tuple dimensiones de los datos
        num_classes: integer cantidad de clases
        """""
        
        classifier = PyTorchClassifier(
                    model = model,
                    clip_values = (min_val, max_val),
                    loss = criterion,
                    optimizer = optim,
                    input_shape = data_dimension,
                    nb_classes = num_classes,
                )
        
        train_members_size = int(len(members_y) * train_ratio)
        train_non_members_size = int(len(non_members_y) * train_ratio)

        x_train_members = members_x[:train_members_size]
        y_train_members = members_y[:train_members_size]

        x_test_members = members_x[train_members_size:]
        y_test_members = members_y[train_members_size:]

        x_train_non_members = non_members_x[:train_members_size]
        y_train_non_members = non_members_y[:train_members_size]

        x_test_non_members = non_members_x[train_non_members_size:]
        y_test_non_members = non_members_y[train_non_members_size:]

        mem_attack = MembershipInferenceBlackBox(classifier)

        mem_attack.fit(x_train_members, y_train_members, x_train_non_members, y_train_non_members) 

        inferred_train_bb = mem_attack.infer(x_test_members, y_test_members)
        inferred_test_bb = mem_attack.infer(x_test_non_members, y_test_non_members)

        train_acc = np.sum(inferred_train_bb) / len(inferred_train_bb)
        test_acc = 1 - (np.sum(inferred_test_bb) / len(inferred_test_bb))
        acc = (train_acc * len(inferred_train_bb) + test_acc * len(inferred_test_bb)) / (len(inferred_train_bb) + len(inferred_test_bb))       

        print(f"Members Accuracy: {train_acc:.4f}")
        print(f"Non Members Accuracy {test_acc:.4f}")
        print(f"Attack Accuracy {acc:.4f}")

        return mem_attack, inferred_train_bb, inferred_test_bb, [x_test_members, y_test_members], [x_test_non_members, y_test_non_members]
    
    def memberships_inference_atck(self,x_data, y_data, attack_ratio, members_ratio, model, 
                                   criterion, optim, max_val, min_val, data_dimension, num_classes):# Lo que se debería pasar aquí son un conjunto de datos para determinar si pertenecen o no al cliente, a partir de los parámetros, la arquitectura del modelo y un conjunto de datos que asume el atacante que analiza el cliente o el modelo en si
        
        """""
        Ejecución del ataque de inferencia de miembros, el ataque selecciona los miembros

        x_data: ndarray datos conocidos por el adversario
        y_data: ndarray etiquetas datos conocidos por el adversario
        attack_ratio: float porciento de los datos utilizados para entrenar el modelo adversario del ataque
        members_ratio: float prociento de los datos utilizados como miembros y no miembros
        model: nn.Module modelo del cliente a realizar el ataque
        criterion: criterio del modelo
        optim: optimizador del modelo
        max_val: float valor máximo de los datos conocidos
        min_val: float valor mínimo valor de los datos conocidos
        data_dimension: tuple dimensiones de los datos
        num_classes: integer cantidad de clases
        """""

        classifier = PyTorchClassifier(
                    model = model,
                    clip_values = (min_val, max_val),
                    loss = criterion,
                    optimizer = optim,
                    input_shape = data_dimension,
                    nb_classes = num_classes,
                )

        attack_train_ratio = attack_ratio
        attack_train_size = int(len(x_data) * attack_train_ratio)

        attack_x_train = x_data[:attack_train_size]
        attack_y_train = y_data[:attack_train_size]
        attack_x_test = x_data[attack_train_size:] 
        attack_y_test = y_data[attack_train_size:]

        factor_train_membership = members_ratio
        #limit_test = int(len(attack_x_test) * factor_train_membership)

    #Para los miembros que pertenezcan
        limit_train = int(len(attack_x_train) * factor_train_membership)
        X_train_to_members = attack_x_train[:limit_train]
        y_train_to_members = attack_y_train[:limit_train]

        X_test_to_member = attack_x_train[limit_train:]
        y_test_to_member = attack_y_train[limit_train:]

    #Para los miembros que no pertenezcan
        limit_train = int(len(attack_x_test) * factor_train_membership)
        X_train_to_non_members = attack_x_test[:limit_train]
        y_train_to_non_members = attack_y_test[:limit_train]

        X_test_to_non_member = attack_x_test[limit_train:]
        y_test_to_non_member = attack_y_test[limit_train:]

        mem_attack = MembershipInferenceBlackBox(classifier)

        mem_attack.fit(X_train_to_members, y_train_to_members, X_train_to_non_members, y_train_to_non_members)

        inferred_train_bb = mem_attack.infer(X_test_to_member, y_test_to_member)
        inferred_test_bb = mem_attack.infer(X_test_to_non_member, y_test_to_non_member)

        train_acc = np.sum(inferred_train_bb) / len(inferred_train_bb)
        test_acc = 1 - (np.sum(inferred_test_bb) / len(inferred_test_bb))
        acc = (train_acc * len(inferred_train_bb) + test_acc * len(inferred_test_bb)) / (len(inferred_train_bb) + len(inferred_test_bb))
        print(f"Members Accuracy: {train_acc:.4f}")
        print(f"Non Members Accuracy {test_acc:.4f}")
        print(f"Attack Accuracy {acc:.4f}")

        return mem_attack, inferred_train_bb, inferred_test_bb, [X_test_to_member, y_test_to_member], [X_test_to_non_member, y_test_to_non_member]


def util_count_unique_atr_value(x):
    values = []

    values = np.unique(x).tolist()
    print(values)

    return values