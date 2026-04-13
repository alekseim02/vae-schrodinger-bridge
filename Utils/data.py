from torchvision import transforms
import os
import torch
from glob import glob
from PIL import Image
from torch.utils.data import Dataset


# Definir la clase de transformación CelebAi
from torchvision import transforms

class CelebAi:
    def __init__(self, image_size=128):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # redimensiona a tamaño fijo
            transforms.ToTensor(),  # convierte a tensor
        ])

    def __call__(self, x):
        return self.transform(x)

# Nueva clase para preprocesar imágenes MNIST
class MNISTTransform:
    def __init__(self, image_size=28):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # asegura tamaño 28x28
            transforms.Grayscale(num_output_channels=1),  # asegura 1 canal
            transforms.ToTensor(),  # convierte a tensor
        ])

    def __call__(self, x):
        return self.transform(x)
    
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        # Ordena los archivos por nombre alfabético
        self.image_paths = sorted([f for f in glob(os.path.join(image_dir, "*") ) if f.lower().endswith(('.jpg', '.png'))])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image  # no hay etiquetas
    
class LatentDataset(Dataset):
    def __init__(self, pt_file, min_val=None, max_val=None):
        data = torch.load(pt_file)
        self.latents = torch.cat(data, dim=0).view(-1, 4, 16, 16)

        if min_val is None or max_val is None:
            self.min_val = self.latents.min()
            self.max_val = self.latents.max()
        else:
            self.min_val = min_val
            self.max_val = max_val

    def __len__(self):
        return self.latents.shape[0]

    def __getitem__(self, idx):
        latent = self.latents[idx]
        latent = 2 * (latent - self.min_val) / (self.max_val - self.min_val) - 1
        return latent


class PairedLatentDataset(Dataset):
    def __init__(self, pt_file_faces, min_val=None, max_val=None):
        # Cargar el archivo .pt que contiene una lista de tensores
        data_faces = torch.load(pt_file_faces)
        self.face_latents = torch.cat(data_faces, dim=0).view(-1, 4, 16, 16)
        self.length = len(self.face_latents)

        # Calcular min y max globales si no se pasan (solo debe hacerse en entrenamiento)
        if min_val is None or max_val is None:
            self.min_val = self.face_latents.min().item()
            self.max_val = self.face_latents.max().item()
            print(f"If:[Dataset] Calculated min: {self.min_val}, max: {self.max_val}")
        else:
            self.min_val = float(min_val)
            self.max_val = float(max_val)
            print(f"Else:[Dataset] Using provided min: {self.min_val}, max: {self.max_val}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        face = self.face_latents[idx]
        face = 2 * (face - self.min_val) / (self.max_val - self.min_val) - 1
        return  face
# Asegúrate de que esta clase esté accesible desde el módulo
__all__ = ['CelebAi', 'ImageDataset', 'LatentDataset', 'PairedLatentDataset']

