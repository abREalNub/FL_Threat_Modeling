import torch
import numpy as np

#Tengo 2 formas de llegar, 1, tomar la arquitectura del modelo, pasar este conjunto de datos supuestamente conocido e identificar las etiquetas a partir de los gradientes
#Segunda forma, tomar los gradientes y ver como utilizarlos para predecir las etiquetas. Estos gradientes deben ir por cada instancia

def label_inference_attack_grad(output_grad, real_label):#En este sería el paso de una lista de los gradientes para cada muestra, la lista es recoplada de poner un atacante y recuperar los tensores por cada ronda de entrenamiento
    
    inferred_correct = 0
    inferred_wrong = 0
    inferred_labels = np.zeros( ((len(output_grad)),) , dtype= int)
    for sample_id in range(len(output_grad)):
                grad_per_sample = output_grad[sample_id]
                for logit_id in range(len(grad_per_sample)):
                    if grad_per_sample[logit_id] < 0:
                        inferred_label = logit_id
                        inferred_labels [sample_id] = inferred_label
                        if inferred_label == real_label[sample_id]:
                            inferred_correct += 1
                        else:
                            inferred_wrong += 1
                        break
    print("Correctas", inferred_correct, "Incorrectas", inferred_wrong)
    return inferred_labels

def label_inference_attack_model(model, criterion, data, real_label):#El simulate labels es una preddcion que me va a dar el modelo

    #with torch.no_grad(): Por si acaso pa que no actualice los gradientes
    model.eval()
    labels_output = model(data)
    #pseudolabels = labels_output.data.max(1, keepdim=True)[1]
    #pseudolabels = torch.sigmoid(labels_output).round()


    model.train(mode=True)
    output = model(data)
    output.retain_grad()
    #loss = torch.sum(output)# Si no sirve, entonces pasar el optimizador
    loss = criterion(output, labels_output)
    #print(loss)
    loss.backward()
    grad_per_sample = output.grad
    #print(grad_per_sample.size())
    print(loss)
    infered = label_inference_attack_grad(grad_per_sample, real_label)
    
    pass