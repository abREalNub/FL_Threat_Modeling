import random
from copy import deepcopy
from typing import Callable, Optional

import numpy as np
from torchvision.datasets import MNIST
from PIL import Image
import os


class TriggerHandler(object):

    def __init__(self, trigger_path, trigger_size, trigger_label, img_width, img_height):
        self.trigger_img = Image.open(trigger_path).convert('RGB')
        self.trigger_size = trigger_size
        self.trigger_img = self.trigger_img.resize((trigger_size, trigger_size))
        self.trigger_label = trigger_label
        self.img_width = img_width
        self.img_height = img_height

    def put_trigger(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(np.uint8(img))

        img.paste(self.trigger_img, (self.img_width - self.trigger_size, self.img_height - self.trigger_size))

        return np.array(img)


class MNISTPoison(MNIST):

    def __init__(
            self,
            args,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height = self.__shape_info__()
        self.channels = 1

        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.trigger_label, self.width,
                                              self.height)
        self.poisoning_rate = args.poisoning_rate if train else 1.0
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "processed")

    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="L")
        # NOTE: According to the threat model, the trigger should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poi_indices:
            target = self.trigger_handler.trigger_label
            img = self.trigger_handler.put_trigger(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class ImagePoison:

    def __init__(
            self,
            images: np.ndarray,
            targets: np.ndarray,
            trigger_path: str,
            trigger_size: int,
            trigger_label: int,
            poisoning_rate: float,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None
    ) -> None:
        self.images = images
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

        # Suponemos que las imágenes tienen la misma forma y extraemos ancho y alto
        self.width, self.height = self.__shape_info__()
        self.channels = self.images.shape[1]  # Esto puede cambiar según el dataset

        self.trigger_handler = TriggerHandler(trigger_path, trigger_size, trigger_label, self.width, self.height)
        self.poisoning_rate = poisoning_rate

        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples (poisoning rate {self.poisoning_rate})")

    def __shape_info__(self):
        # Suponemos que la primera imagen en el dataset tiene la forma correcta

        return self.images.shape[2], self.images.shape[3]

    def __getitem__(self, index):
        img, target = self.images[index], self.targets[index]
        img = np.transpose(img, (1, 2, 0))
        img_pil = Image.fromarray(img.squeeze(), mode="L")  # Adaptar según el formato del dataset

        # Aplicar el trigger si el índice está en la lista de envenenados
        if index in self.poi_indices:
            target = self.trigger_handler.trigger_label
            img_pil = self.trigger_handler.put_trigger(img_pil)

        if self.transform is not None:
            img_pil = self.transform(img_pil)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_pil, target

    def __len__(self):
        return len(self.images)


def poison_mnist_dataset(data, trigger_path, trigger_size, trigger_label, poisoning_rate):
    """
  Envenena un dataset MNIST añadiendo un trigger a una proporción aleatoria de las imágenes.

  Args:
    data: Un dataset MNIST ya cargado y transformado.
    target: Es el conjunto de caracter'isticas del dataset
    trigger_path: Ruta al archivo de imagen del trigger.
    trigger_size: Tamaño del trigger.
    trigger_label: Etiqueta a la que se asignarán las imágenes envenenadas.
    poisoning_rate: Proporción de imágenes a envenenar.

  Returns:
    Un nuevo dataset con las imágenes envenenadas.
  """
    new_data, new_targets = data
    _, _, width, height = new_data.shape

    # Crear un objeto TriggerHandler
    trigger_handler = TriggerHandler(trigger_path, trigger_size, trigger_label, width, height)

    # Determinar las imágenes a envenenar
    num_samples = len(new_targets)
    num_poisoned_samples = int(num_samples * poisoning_rate)
    poison_indices = random.sample(range(num_samples), num_poisoned_samples)

    # Envenenar las imágenes

    for i, img in enumerate(new_data):
        if i in poison_indices:

            new_data[i] = trigger_handler.put_trigger(img)
            new_targets[i] = trigger_handler.trigger_label

    poisoned_data = np.column_stack((new_data, new_targets))

    return poisoned_data
