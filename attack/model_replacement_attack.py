import torch
import numpy as np
from copy import deepcopy
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
from torchvision.datasets import MNIST
from attack.backdoorss import sniper_backdoor


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


class ModelReplacement:

    def __init__(self, norm_bound):
        self.first_server_model = None
        self.adversarial_model = None
        self.scaling_factor = 1.0
        self.norm_bound = norm_bound

    def set_model(self, model):
        self.first_server_model = model

    def create_adversarial_model(self, gan_images, epoch=1):
        train_dataset = MNIST(root='.', train=True, download=False,
                              transform=transforms.Compose([transforms.Resize((28, 28)),
                                                            transforms.ToTensor(),
                                                            # transforms.Lambda(lambda x:
                                                            # x.permute(1, 2, 0)),
                                                            transforms.Normalize((0.5,), (0.5,))
                                                            # transforms.Normalize((0.5,),
                                                            # (0.5,))
                                                            ]))

        # Convertir datos originales a numpy
        x_train, y_train = train_dataset.data.numpy(), train_dataset.targets.numpy()
        x_train = np.expand_dims(x_train, axis=1)  # Ajustar forma

        # Seleccionar 6000 imágenes aleatorias de MNIST
        indices = np.random.choice(len(x_train), 6000, replace=False)
        x_train, y_train = x_train[indices], y_train[indices]

        # Asignar etiquetas a imágenes generadas por GAN
        gan_labels = np.full((gan_images.shape[0],), 2)  # Etiqueta arbitraria

        # Concatenar dataset original con GAN
        x_combined = np.concatenate((x_train, gan_images), axis=0)
        y_combined = np.concatenate((y_train, gan_labels), axis=0)

        target = 1
        source = 2
        percent_to_change = 0.2

        new_img, new_target = sniper_backdoor(data=x_combined, targets=y_combined, source_label=source,
                                                target_label=target,
                                                epsilon=percent_to_change)

        # Convertir a formato tensor

        x_tensor = torch.tensor(new_img, dtype=torch.float32)
        y_tensor = torch.tensor(new_target, dtype=torch.long)

        # Crear DataLoader
        dataset_adversarial = TensorDataset(x_tensor, y_tensor)
        dataloader = DataLoader(dataset_adversarial, batch_size=256, shuffle=True)

        self.adversarial_model = deepcopy(self.first_server_model)
        optimizer = torch.optim.Adam(self.adversarial_model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        # Entrenamiento
        self.adversarial_model.train()
        for _ in range(epoch):
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = self.adversarial_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

    def calculate_scaling_factor(self, selected_clients, client_data_sizes):
        """Calcula β = (Σn_k) / n_malicioso."""
        total_data = sum(client_data_sizes[k] for k in selected_clients)
        malicious_data = client_data_sizes[selected_clients[0]]  # Asume 1 cliente malicioso
        self.scaling_factor = total_data / malicious_data

    def poison_model_update(self, client_model):
        if self.adversarial_model is None:
            raise ValueError("Adversarial model not trained!")

            # Calcular Δw = β * (w_mal - w_global)
        mal_weights = self.adversarial_model.state_dict()
        global_weights = self.first_server_model.state_dict()
        delta = {k: self.scaling_factor * (mal_weights[k] - global_weights[k]) for k in mal_weights}

        client_model.load_state_dict(delta)

        return client_model
