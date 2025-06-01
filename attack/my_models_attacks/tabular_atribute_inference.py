import numpy as np
from art.attacks.inference.attribute_inference import AttributeInferenceBlackBox
from art.attacks.inference.attribute_inference import AttributeInferenceWhiteBoxLifestyleDecisionTree
from art.attacks.inference.attribute_inference import AttributeInferenceWhiteBoxDecisionTree
import torch
from art.estimators.classification import PyTorchClassifier

class tabular_inference_atributte:
    """""
    Clase que implementa el ataque de inferencia de un atributo
    obtenido de la herramienta, adversarial robustness toolbox: https://github.com/Trusted-AI/adversarial-robustness-toolbox.git
    del articulo: .

    Atributos:
    model: nn.Module arauitectura del modelo
    summary_infer_for_client: dict resultados de inferencias de atributo del ataque por cada cliente en cada ronda
    summary_model_to_infer_for_client: dict resultados de los modelos adversarios de cada ataque fectuado
    summary_server_inf_img: dict resultados de imágenes reconstruídas por el ataque por el servidor en cada ronda
    summary_real__att_data_adv_dont_know: dict resumen de los datos reales conocidos para la validación 

    """""
    def __init__(self, model):
        self.model = model
        #self.inference_attack = None
        self.summary_infer_for_client = dict()
        self.summary_model_to_infer_for_client = dict()
        self.summary_real__att_data_adv_dont_know = dict()

    def update_model(self, params):
        """""
        Método para ctualizar el modelo de un cliente a partir de los valores del os parámetros que llegan al
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


    def inf_atribute_attck_train(self, x_data, y_data, train_test_ratio, model, 
                                 criterion, optim, max_val, min_val, data_dimension, num_classes):#Model debe tener la estructura quelllos definen para los modelos
        """""
        Ejecución del ataque de inferencia de atributos
        x_data_ ndarray datos conocidos por el daversario
        y_data: ndarray etiquetas de los datos conocidos
        train_test_ratio: float valor entre 0-1 que define el prociento de datos utilizado por entrenar el modelo adversario
        model: nn.Module arquitectura del modelo de un cliente
        criterion: criterio del modelo
        optim: optimizador del modelo
        max_val: float valor máximo de los datos conocidos
        min_val: float valor mínimo de los datos conocidos
        data_dimension: tuple dimensiones de los datos
        num_classes: integer cantidad de clases 
        """""

        print("Ataque de inferencia de atributos")

        classifier = PyTorchClassifier(
                    model = model,
                    clip_values = (min_val, max_val),
                    loss = criterion,
                    optimizer = optim,
                    input_shape = data_dimension,
                    nb_classes = num_classes,
                )

        attack_train_ratio = train_test_ratio
        attack_train_size = int(len(x_data) * attack_train_ratio)
        attack_test_size = int(len(y_data) * attack_train_ratio)
        attack_x_train = x_data[:attack_train_size]
        attack_y_train = y_data[:attack_train_size]
        attack_x_test = x_data[attack_train_size:] 
        attack_y_test = y_data[attack_train_size:]

        attack_feature = 1
        attack_x_test_predictions = np.array([np.argmax(arr) for arr in classifier.predict(attack_x_test)]).reshape(-1,1) #Poner el clasificador como ellos lo tienen

        attack_x_test_feature = attack_x_test[:, attack_feature].copy().reshape(-1, 1)
        bb_attack = AttributeInferenceBlackBox(classifier, attack_feature = attack_feature)
        attack_x_test = np.delete(attack_x_test, attack_feature, 1)


        bb_attack.fit(attack_x_train)

        values= None

        inferred_train_bb = bb_attack.infer(attack_x_test, pred = attack_x_test_predictions, values = values)

        
        #Esto para abajo es validación
        train_acc = np.sum(np.around(inferred_train_bb, decimals=8).reshape(1,-1) == np.around(attack_x_test_feature, decimals=8).reshape(1,-1))/ len(inferred_train_bb)
        print("The accuracy inference for the adversary data is", train_acc)


        return inferred_train_bb, bb_attack, attack_x_test_feature
    
    def save_summary_infer_for_clients(self, rounds, client_id, infer, model_to_future_infer, data_to_evaluate):
        
        """""
        Se almacenan los resultados del ataque, por cada cliente en cada ronda
        infer: list listado de los valores inferidos para el atributo
        model_to_future_infer: modelo utilizado por el adversario para efectuar el ataque de inferencia
        data_to_evaluate: ndarray datos para la validación del ataque
        rounds: integer número de la ronda
        client_id: integer número del cliente
        """""

        if rounds not in self.summary_infer_for_client.keys():
            self.summary_infer_for_client[rounds] = dict()
            self.summary_model_to_infer_for_client[rounds] = dict()
            self.summary_real__att_data_adv_dont_know[rounds] = dict()
        self.summary_infer_for_client[rounds][client_id] = infer
        self.summary_model_to_infer_for_client[rounds][client_id] = model_to_future_infer
        self.summary_real__att_data_adv_dont_know[rounds][client_id] = data_to_evaluate #En formato de la característica a inferir como etiqueta


def inf_atribute_attck_wb_one(x_data, y_data, model):#El segundo de white box de inferir atributos en estos no se entrena un modelo, se utiliza la información del estimador, osea conocida la estructura del modelo del cliente + los parámetros que se tienene
    attack_train_ratio = 0.9
    attack_train_size = int(len(x_data) * attack_train_ratio)
    attack_test_size = int(len(y_data) * attack_train_ratio)
    attack_x_train = x_data[:attack_train_size]
    attack_y_train = y_data[:attack_train_size]
    attack_x_test = x_data[attack_train_size:] 
    attack_y_test = y_data[attack_train_size:]

    attack_feature = 1
    attack_x_test_predictions = np.array([np.argmax(arr) for arr in model.predict(attack_x_test)]).reshape(-1,1) #Poner el clasificador como ellos lo tienen
    print("Dim del test attck:", attack_x_test.shape)
    print("Dim del prediction attck:", attack_x_test_predictions.shape)

    priors = [3465 / 5183, 1718 / 5183]

    attack_x_test_feature = attack_x_test[:, attack_feature].copy().reshape(-1, 1)

    wb_attack = AttributeInferenceWhiteBoxLifestyleDecisionTree(model, attack_feature=attack_feature)

    attack_x_test = np.delete(attack_x_test, attack_feature, 1)

    print("Tamaño de las x de ataque",len(attack_x_test))
    print("Tamaño de las x de ataque",len(attack_x_test_predictions))



    #Aqui define el entrenamiento
    #values = [-0.70718864, 1.41404987]
    values= None
    #values = [-1.424395723148083, -0.7130212117030289, -0.0016467002579747094, 0.7097278111870795, 1.4211023226321338]
    inferred_train_wb1 = wb_attack.infer(attack_x_test, attack_x_test_predictions, values=values, priors=priors)


    #train_acc = np.sum(inferred_train_bb == np.around(attack_x_test_feature, decimals=8).reshape(1,-1)) / len(inferred_train_bb)
    train_acc = np.sum(np.around(inferred_train_wb1, decimals=8).reshape(1,-1) == np.around(attack_x_test_feature, decimals=8).reshape(1,-1))/ len(inferred_train_wb1)
    print(train_acc)
    return inferred_train_wb1

def inf_atribute_attck_wb_two(x_data, y_data, model):#El segundo de white box de inferir atributos
    attack_train_ratio = 0.9
    attack_train_size = int(len(x_data) * attack_train_ratio)
    attack_test_size = int(len(y_data) * attack_train_ratio)
    attack_x_train = x_data[:attack_train_size]
    attack_y_train = y_data[:attack_train_size]
    attack_x_test = x_data[attack_train_size:] 
    attack_y_test = y_data[attack_train_size:]

    attack_feature = 1
    attack_x_test_predictions = np.array([np.argmax(arr) for arr in model.predict(attack_x_test)]).reshape(-1,1) #Poner el clasificador como ellos lo tienen
    print("Dim del test attck:", attack_x_test.shape)
    print("Dim del prediction attck:", attack_x_test_predictions.shape)

    priors = [3465 / 5183, 1718 / 5183]

    attack_x_test_feature = attack_x_test[:, attack_feature].copy().reshape(-1, 1)

    wb_attack = AttributeInferenceWhiteBoxDecisionTree(model, attack_feature=attack_feature)

    attack_x_test = np.delete(attack_x_test, attack_feature, 1)

    print("Tamaño de las x de ataque",len(attack_x_test))
    print("Tamaño de las x de ataque",len(attack_x_test_predictions))



    #Aqui define el entrenamiento
    #values = [-0.70718864, 1.41404987]
    values= None
    #values = [-1.424395723148083, -0.7130212117030289, -0.0016467002579747094, 0.7097278111870795, 1.4211023226321338]
    inferred_train_wb = wb_attack.infer(attack_x_test, attack_x_test_predictions, values = values, priors = priors)


    #train_acc = np.sum(inferred_train_bb == np.around(attack_x_test_feature, decimals=8).reshape(1,-1)) / len(inferred_train_bb)
    train_acc = np.sum(np.around(inferred_train_wb, decimals=8).reshape(1,-1) == np.around(attack_x_test_feature, decimals=8).reshape(1,-1))/ len(inferred_train_wb)
    print(train_acc)
    return inferred_train_wb