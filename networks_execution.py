from fileinput import hook_compressed
from typing import OrderedDict
from networks_models import build_model
from torch import optim, nn
from tqdm import tqdm
import torch
import os
from torchvision.models.resnet import ResNet
from torchvision.models.vgg import VGG
import globals_for_inference as main_test




class ExecutionNetwork():
    def __init__(self, local_epochs = 5, trainloader = None, testloader = None, testdata = None, n_classes=10, dataname="mnist"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.CrossEntropyLoss()
        self.model = build_model(n_classes=n_classes, dataname=dataname).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.05, momentum=0.9) #self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum) parecida al entrenar
        self.local_epochs = local_epochs
        self.trainloader = trainloader
        self.testloader = testloader
        self.testdata = testdata
        self.list_train_loss = []
        self.list_train_acc = []
    
    def for_fd_server_model_config(self):
        return self.criterion, self.model, self.optimizer
    
    def trainNetwork(self, local_epochs, criterion, optimizer, momentum = 0.9, lr = 0.01, trainloader = None, testloader = None, model = None):
        self.model.train()
        for local_epoch in range(local_epochs):
            running_loss = 0.0
            epoch_error = 0.0
            correct = 0
            total = 0
            cont = 1
            num_tot=0
            grad_per_samples = []
            samples = []
            label_samples = []
            for (data, target) in tqdm(trainloader):
                data, target = data.float().to(self.device), target.to(self.device)
                #data = data.squeeze()
                #print(data.shape)
                optimizer.zero_grad()
                output = model(data)
                #print(target.shape)

                loss = criterion(output, target.long()) # Aqui quite el argmax xq me estaba dando error de no coincidencia de batch

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                

                _, predicted = output.max(dim=1)
                total += target.size(0)
                #correct += predicted.eq(torch.argmax(target)).sum().item()#aqui quite el argmax con dimension
                correct += (predicted==target).sum().item()
                num_tot += predicted.size(0)
            #epoch_error = running_loss
            #print("Epoch", local_epoch, "Error", loss, "Acc", correct/num_tot)

        train_loss = running_loss / len(trainloader)
        #self.list_train_loss.append(train_loss)
        train_acc = 100 * correct / len(trainloader.dataset.data)
        #self.list_train_acc.append(train_acc)

        #self.list_train_loss.append(train_loss)
        #print(self.list_train_loss)
        #self.list_train_acc.append(train_acc)
        #print(self.list_train_acc)
    
    def trainAdv(self, local_epochs, criterion, optimizer, momentum = 0.9, lr = 0.01, trainloader = None, testloader = None, model = None):
        self.model.train()
        for local_epoch in range(local_epochs):
            running_loss = 0.0
            epoch_error = 0.0
            correct = 0
            total = 0
            cont = 1
            num_tot=0
            grad_per_samples = []
            samples = []
            label_samples = []
            for (data, target) in tqdm(trainloader):
                data, target = data.float().to(self.device), target.to(self.device)
                #data = data.squeeze()
                #print(data.shape)
                optimizer.zero_grad()
                output = model(data)
                output.retain_grad()
                #print(target.shape)

                loss = criterion(output, target.long()) # Aqui quite el argmax xq me estaba dando error de no coincidencia de batch

                loss.backward()

                #Esto es para ataques de inferencias
                
                out_grad = output.grad

                #def save_grad(grad):
                #    print(type(grad))
                #    out_grad.append(grad)

                #output.register_hook(save_grad)

                #print("Tamaño del grad", len(out_grad))
                grad_per_samples.append(out_grad) # Cada grad es un batch
                samples.append(data) # Cada data es un batch, por tanto sería por batch, por cada instancia
                label_samples.append(target) # Cada target es un batch, por tanto sería por batch, por cada instancia
                #Esto es para ataques de inferencia
                optimizer.step()

                running_loss += loss.item()
                

                _, predicted = output.max(dim=1)
                total += target.size(0)
                #correct += predicted.eq(torch.argmax(target)).sum().item()#aqui quite el argmax con dimension
                correct += (predicted==target).sum().item()
                num_tot += predicted.size(0)
            #epoch_error = running_loss
            #print("Epoch", local_epoch, "Error", loss, "Acc", correct/num_tot)

        train_loss = running_loss / len(trainloader)
        #self.list_train_loss.append(train_loss)
        train_acc = 100 * correct / len(trainloader.dataset.data)
        for id, value in main_test.grad_per_data_samples.items():
            if len(value) == 0:
                print(id)
                main_test.grad_per_data_samples[id] = grad_per_samples
                main_test.data_samples[id] = samples
                main_test.target_samples[id] = label_samples
                break
        #self.list_train_acc.append(train_acc)

        #self.list_train_loss.append(train_loss)
        #print(self.list_train_loss)
        #self.list_train_acc.append(train_acc)
        #print(self.list_train_acc)
