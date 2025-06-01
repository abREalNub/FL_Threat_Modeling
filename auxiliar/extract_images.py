import os
import torch
from torchvision import transforms
from PIL import Image


def extract_digits_from_directory(directory_path, grid_size=7):
    """
    Extrae dígitos individuales de todas las imágenes PNG en un directorio y los almacena como tensores.

    - directory_path: Ruta del directorio donde están las imágenes.
    - grid_size: Tamaño de la cuadrícula de dígitos (por defecto 7x7).

    Retorna una lista de tensores con los dígitos de todas las imágenes.
    """
    # Transformaciones para preprocesamiento
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Ajustar al tamaño de MNIST
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalización estándar
    ])

    gan_images = []

    # Recorrer todos los archivos PNG en el directorio
    for filename in os.listdir(directory_path):
        if filename.endswith(".png"):
            image_path = os.path.join(directory_path, filename)
            img = Image.open(image_path).convert("L")  # Cargar imagen en escala de grises
            img_width, img_height = img.size

            # Definir tamaño de cada celda
            digit_width, digit_height = img_width // grid_size, img_height // grid_size

            # Extraer cada dígito y convertirlo en tensor
            for row in range(grid_size):
                for col in range(grid_size):
                    left, top = col * digit_width, row * digit_height
                    right, bottom = left + digit_width, top + digit_height
                    digit = img.crop((left, top, right, bottom))

                    # Convertir a tensor y agregar a la lista
                    gan_images.append(transform(digit))
    gan_images = torch.stack(gan_images)
    return gan_images



