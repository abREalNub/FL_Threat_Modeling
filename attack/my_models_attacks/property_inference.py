import torch
import numpy as np
import copy
from sklearn.model_selection import train_test_split

from attack.my_models_attacks.PIA.utils import get_classifier

from attack.my_models_attacks.PIA.optimizer_builder import get_optimizer

import logging

logger = logging.getLogger(__name__)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Detalle, acá entrar directamente los parámetros agregador o de cada cliente y después hacer la predicción
# Donde este el modelo cambiarlo a los parametros directamente
# Aqui igual asumen el conocimiento del modelo, asi que lo mismo, a partir del conocimiento de los parámetros crear un modelo adversario

class Passive_property_inference:
    def __init__(self, model_server, type_classifier, data: dict, lr, optimizer, criterion, batch, grad_clip, local_upt_num):
         self.model_server = model_server
         self.adv_classifier = get_classifier(type_classifier) 
         self.data = data
         self.optimizer = optimizer
         self.lr = lr
         self.criterion = criterion
         self.batch_size = batch
         self.grad_clip = grad_clip
         self.local_upt_num = local_upt_num
         self.dataset_prop_classifier = {"x": None, 'y': None}
         self.collect_updates_summary = dict()
         #self.data_train = None
         self.process_data_new()
         #self.assing_type_for_criterion(None)

    def assing_type_for_criterion(self, to_tensor_convert_targ_in,to_tensor_convert_targ):#Depende mucho de los datos, tiene modelos de clasificaión binaria
         crit_name = self.criterion.__class__.__name__
         inp = None
         targ = None
         if crit_name == "CrossEntropyLoss":#long
              inp = to_tensor_convert_targ_in.to(dtype = torch.float64)
              targ = to_tensor_convert_targ.to(dtype = torch.long)
         elif crit_name == "MSELoss":
              inp = to_tensor_convert_targ_in.to(dtype = torch.float64)
              targ = to_tensor_convert_targ.to(dtype = torch.float64)
         elif crit_name == "L1Loss":
              inp = to_tensor_convert_targ_in.to(dtype = torch.float64)
              targ = to_tensor_convert_targ.to(dtype = torch.float64)
         elif crit_name == "NLLLoss":#long
              inp = to_tensor_convert_targ_in.to(dtype = torch.float64)
              targ = to_tensor_convert_targ.to(dtype = torch.long)
         elif crit_name ==  "BCELoss":
              inp = to_tensor_convert_targ_in.to(dtype = torch.float64)
              targ = to_tensor_convert_targ.to(dtype = torch.float64)
         elif crit_name ==  "BCEWithLogitsLoss":
            inp = to_tensor_convert_targ_in.to(dtype = torch.float64)
            targ = to_tensor_convert_targ.to(dtype = torch.float64)
         elif crit_name == "SmoothL1Loss":
              inp = to_tensor_convert_targ_in.to(dtype = torch.float64)
              targ = to_tensor_convert_targ.to(dtype = torch.float64)     
         return inp, targ
    
    def define_summary(self,n_client, n_rounds):
         for i in range(n_rounds):
              self.collect_updates_summary[i] = dict()
              for n in range(n_client):
                   self.collect_updates_summary[i][n] = None

    def process_data_new(self):
        x_data = self.data["x"]
        data_size = x_data[0].shape
        y_data = self.data["y"]
        mean_data = self.data["mean"]
        std_data = self.data["std"]
        positive_labels = [0,2]

        prop = self.get_data_property(x_data, y_data, positive_labels)

        self.data_train = {'x': x_data, 'y': y_data, 'prop': prop}
        pass

    def process_data(self):# Todo en numpy
        x_data = self.data["x"]
        data_size = x_data[0].shape
        y_data = self.data["y"]
        mean_data = self.data["mean"]
        std_data = self.data["std"]
        positive_labels = [0,2]

        self.get_data_property(x, y, positive_labels)

        weights = np.random.normal(loc = mean_data, scale = std_data, size = data_size)
        bias = np.random.normal(loc = mean_data, scale = std_data)

        prop_weights = np.random.normal(loc = mean_data,
                                        scale = std_data,
                                        size = data_size)
        
        x = np.random.normal(loc = 0.0,
                                 scale = 1.0,
                                 size = (10000 , data_size[0],data_size[1], data_size[2]))


        y = list(range(6,10))
        arreglo = np.tile(y, 10000 // len(y) + 1)[:10000]
        arreglo = np.array(arreglo)
        np.random.shuffle(arreglo)

        #y = np.expand_dims(arreglo, -1)
        y = arreglo
        #print(y_data.shape)
        #print(y.shape)


        prop_true = np.ones(x_data.shape[0]) #Para las conocidas
        prop_neg = np.zeros(x.shape[0]) #Para las generadas

        x_s = np.vstack([x_data, x])
        y_s = np.hstack([y_data, y])
        props = np.hstack([prop_true, prop_neg])
        prop = np.expand_dims(props, -1)
        #print("Dimensiones de las xs:", x_s.shape, "Dimensiones de las ys:", y_s.shape)
        #print("Dimensiones del prop:", prop.shape)

        self.data_train = {'x': x_s, 'y': y_s, 'prop': prop}

        return self.data_train
    
    def get_batch(self):#Debe entrar con los canales

        data = self.data_train

        prop_ind = np.random.choice(np.where(data['prop'] == 1)[0],
                                        self.batch_size,
                                        replace=True)
        #print("Cantidad de propiedades positivas",len(prop_ind)) #Indicarlo como pertenencia
        x_batch_prop = data['x'][prop_ind, :]#como arriba
        y_batch_prop = data['y'][prop_ind]

        nprop_ind = np.random.choice(np.where(data['prop'] == 0)[0],
                                        self.batch_size,
                                        replace=True)
        
        #print("Cantidad de propiedades negativas",len(nprop_ind))
        
        x_batch_nprop = data['x'][nprop_ind, :]
        y_batch_nprop = data['y'][nprop_ind]
        
        return [x_batch_prop, y_batch_prop, x_batch_nprop, y_batch_nprop]
    
    def act_server_model(self, model):
         self.model_server = model
    
    def get_data_for_dataset_prop_classifier(self, local_runs = 10):

        descompose_data = self.data_train
        #dataset_prop_classifier = {"x": None, 'prop': None}

        previous_para = self.model_server.state_dict()
        current_model_param = previous_para
        for _ in range(local_runs):
            #Primero
            x_batch_prop, y_batch_prop, x_batch_nprop, y_batch_nprop = \
                    self.get_batch()
            #Segundo
            para_update_prop = self.get_parameter_updates(
                    self.model_server, previous_para, self.optimizer, self.lr, 
                    self.local_upt_num, self.criterion, self.grad_clip, x_batch_prop, y_batch_prop)
            
            #print("Para las que tienen pertenencia:", torch.sum(para_update_prop))

            #Tercero
            prop = torch.tensor([[1]]).to(torch.device(device))
            self.add_parameter_updates(para_update_prop, prop, self.dataset_prop_classifier)

            #Cuarto
            para_update_nprop = self.get_parameter_updates(
                    self.model_server, previous_para, self.optimizer, self.lr, 
                    self.local_upt_num, self.criterion, self.grad_clip, x_batch_nprop, y_batch_nprop)
            
            #print("Para las que no tienen pertenencia:", torch.sum(para_update_nprop))
            
            prop = torch.tensor([[0]]).to(torch.device(device))
            self.add_parameter_updates(para_update_nprop, prop, self.dataset_prop_classifier)

    
    def get_parameter_updates(self, model, previous_para, optimizer, lr, local_upt_num, criterion, grad_clip, x_batch, y_batch):
        
        model = copy.deepcopy(model)

        model.load_state_dict(previous_para, strict=False)
        optimizer = get_optimizer(type = optimizer, #Si no funciona pues poner directamente acá el optimizador y ya
                                    model = model,
                                    lr = lr)
        for _ in range(local_upt_num):
            optimizer.zero_grad()
            output = model(torch.Tensor(x_batch).to(torch.device(device)))
            target = torch.Tensor(y_batch).to(torch.device(device))
            inp, targ = self.assing_type_for_criterion(output, target)
            loss_auxiliary_prop = criterion(inp, targ)
            loss_auxiliary_prop.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                grad_clip)
            optimizer.step()

        para_prop = model.state_dict()
        updates_prop = torch.hstack([
            (previous_para[name] - para_prop[name]).flatten().cpu()
            for name in previous_para.keys()
            ])
        model.load_state_dict(previous_para, strict=False)

        return updates_prop
    
    def add_parameter_updates(self, parameter_updates, prop, dataset_prop):
        if dataset_prop['x'] is None:
                dataset_prop['x'] = parameter_updates.cpu()
                dataset_prop['y'] = prop.reshape([-1]).cpu()
        else:
                dataset_prop['x'] = torch.vstack(
                    (dataset_prop['x'], parameter_updates.cpu()))
                dataset_prop['y'] = torch.vstack(
                    (dataset_prop['y'], prop.cpu()))
    
    def train_property_classifier(self):
        dataset_prop_classifier = self.dataset_prop_classifier

        x_train, x_test, y_train, y_test = train_test_split(
            dataset_prop_classifier['x'],
            dataset_prop_classifier['y'],
            test_size = 0.5,
            random_state = 42,
            shuffle = True)
        

        self.adv_classifier.fit(x_train, y_train.squeeze())

        y_pred = self.property_inference(x_test)
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_true = y_test.squeeze(), y_pred = y_pred)
        print("El acc:", accuracy)

    
    def collect_updates(self, previous_subs_uptd_para, round,
                        client_id):
        
        updates_prop = torch.hstack([#Tomar los parámetros de un cliente
            (param).flatten().cpu()
            for param in previous_subs_uptd_para
        ])
        if round not in self.collect_updates_summary.keys():
            self.collect_updates_summary[round] = dict()
        self.collect_updates_summary[round][client_id] = updates_prop
    
    def property_inference(self, parameter_updates):
        return self.adv_classifier.predict(parameter_updates)
    
    def infer_collected(self):
        pia_results = dict()

        for round in self.collect_updates_summary.keys():
            for id in self.collect_updates_summary[round].keys():
                if round not in pia_results.keys():
                    pia_results[round] = dict()
                pia_results[round][id] = self.property_inference(
                    self.collect_updates_summary[round][id].reshape(1, -1))
        return pia_results
    
    def get_data_property(self, x, y, positive_labels):
        # A SHOWCASE for Femnist dataset: Property := whether contains a circle.

        prop = np.zeros(y.size)
        #positive_labels = [0, 6, 8]
        for ind in range(len(y)):
            if y[ind] in positive_labels:
                #print(ind)
                prop[ind] = 1
        return prop








def passive_property_inference(dataset_prop_classifier, model, adv_classifier, data: dict, lr, optimizer, criterion, batch, grad_clip, local_upt_num):#Dict: {x: "feat", y: "labels", mean: "float", std: "float"}

    classifier = get_classifier(adv_classifier) #Buscar

    data_prop_clasif = get_data_for_dataset_prop_classifier(dataset_prop_classifier, model, data, batch, optimizer, lr, local_upt_num, criterion, grad_clip, local_runs=10)

    train_property_classifier(classifier, data_prop_clasif)

    pass

def process_data(data):# Todo en numpy
    x_data = data["x"]
    data_size = x_data[0].shape
    y_data = data["y"]
    mean_data = data["mean"]
    std_data = data["std"]

    weights = np.random.normal(loc = mean_data, scale = std_data, size = data_size)
    bias = np.random.normal(loc = mean_data, scale = std_data)

    prop_weights = np.random.normal(loc = mean_data,
                                    scale = std_data,
                                    size = data_size)
    prop_bias = np.random.normal(loc = mean_data, scale = std_data)
    prop = np.sum(x_data * prop_weights, axis=-1 ) + prop_bias #Revisar esto
    prop = 1.0 * ((1 / (1 + np.exp(-1 * prop))) > 0.5)
    prop = np.expand_dims(prop, -1)

    data_train = {'x': x_data, 'y': y_data, 'prop': prop}

    return data_train

def get_batch(data_descomp, batch_size):

    data = data_descomp

    prop_ind = np.random.choice(np.where(data['prop'] == 1)[0],
                                    batch_size,
                                    replace=True)
    
    x_batch_prop = data['x'][prop_ind, :]
    y_batch_prop = data['y'][prop_ind, :]

    nprop_ind = np.random.choice(np.where(data['prop'] == 0)[0],
                                     batch_size,
                                     replace=True)
    
    x_batch_nprop = data['x'][nprop_ind, :]
    y_batch_nprop = data['y'][nprop_ind, :]
    
    return [x_batch_prop, y_batch_prop, x_batch_nprop, y_batch_nprop]

def get_data_for_dataset_prop_classifier(dataset_prop_classifier, model, data, batch_size, optimizer, lr, local_upt_num, criterion, grad_clip, local_runs=10):

    descompose_data = process_data(data)
    #dataset_prop_classifier = {"x": None, 'prop': None}

    previous_para = model.state_dict()
    current_model_param = previous_para
    for _ in range(local_runs):
        #Primero
        x_batch_prop, y_batch_prop, x_batch_nprop, y_batch_nprop = \
                get_batch(descompose_data, batch_size)
        #Segundo
        para_update_prop = get_parameter_updates(
                model, previous_para, optimizer, lr, local_upt_num, criterion, grad_clip, x_batch_prop, y_batch_prop)
        #Tercero
        prop = torch.tensor([[1]]).to(torch.device(device))
        add_parameter_updates(para_update_prop, prop, dataset_prop_classifier)

        #Cuarto
        para_update_nprop = get_parameter_updates(
                model, previous_para, optimizer, lr, local_upt_num, criterion, grad_clip, x_batch_nprop, y_batch_nprop)
        
        prop = torch.tensor([[0]]).to(torch.device(device))
        add_parameter_updates(para_update_nprop, prop, dataset_prop_classifier)

    return dataset_prop_classifier

def get_parameter_updates(model, previous_para, optimizer, lr, local_upt_num, criterion, grad_clip, x_batch, y_batch):
    model = copy.deepcopy(model)

    model.load_state_dict(previous_para, strict=False)
    optimizer = get_optimizer(type = optimizer, #Buscar!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                  model = model,
                                  lr = lr)
    for _ in range(local_upt_num):
        optimizer.zero_grad()
        loss_auxiliary_prop = criterion(
                model(torch.Tensor(x_batch).to(torch.device(device))),
                torch.Tensor(y_batch).to(torch.device(device)))
        loss_auxiliary_prop.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               grad_clip)
        optimizer.step()

    para_prop = model.state_dict()
    updates_prop = torch.hstack([
        (previous_para[name] - para_prop[name]).flatten().cpu()
        for name in previous_para.keys()
        ])
    model.load_state_dict(previous_para, strict=False)

    return updates_prop

def collect_updates(previous_para, updated_parameter, round,
                        client_id):
    pass

def add_parameter_updates(parameter_updates, prop, dataset_prop):
    if dataset_prop['x'] is None:
            dataset_prop['x'] = parameter_updates.cpu()
            dataset_prop['y'] = prop.reshape([-1]).cpu()
    else:
            dataset_prop['x'] = torch.vstack(
                (dataset_prop['x'], parameter_updates.cpu()))
            dataset_prop['y'] = torch.vstack(
                (dataset_prop['y'], prop.cpu()))
    pass

def train_property_classifier(classifier, dataset_prop_classifier):
        x_train, x_test, y_train, y_test = train_test_split(
            dataset_prop_classifier['x'],
            dataset_prop_classifier['y'],
            test_size=0.33,
            random_state=42)
        classifier.fit(x_train, y_train)


def property_inference(self, classifier, parameter_updates):
    return classifier.predict(parameter_updates)


#def infer_collected(classifier, collect_update_sumary): #La idea mia es, que en cada ronda tomar las actualizaciones que lleguen e ir actualizandola, de esta forma las prediccionaes a medida que pasen las rondas serán mejores

#   pass