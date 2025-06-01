from art.estimators.classification import PyTorchClassifier
from art.attacks.inference.model_inversion.mi_face import MIFace
import matplotlib.pyplot as plt
import numpy as np

def mi_face_model_inversion(x_base, y_pos_label, n_labels, sizes, model, criterion, optim, max_pixel_model, min_pixel_model):
    classifier = PyTorchClassifier(
                model=model,
                clip_values=(min_pixel_model, max_pixel_model),#Esto es el máximo y mínimo valor por atributo, algo de numpy debe hacer esta parte
                loss=criterion,
                optimizer=optim,
                input_shape = sizes,
                nb_classes = n_labels,
            )
    
    attack = MIFace(classifier, max_iter=10000, threshold=1.)
    
    x_infer = attack.infer(x_base, y_pos_label)

    print(x_infer.shape)

    for i in range(10):
        img = np.squeeze(x_infer[i], axis=0)
        print(img.shape, np.max(img), np.min(img))
        plt.imshow(img, cmap=plt.cm.gray_r)
        plt.show()

    return x_infer