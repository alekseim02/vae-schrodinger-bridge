import os
from Options.options import parse

# Cargar las opciones desde el archivo YML
path_options = 'Options/vae.yml'  # Ruta del yml
opt = parse(path_options)   # Se crea el diccionario de opciones

# Configurar las GPUs visibles (usualmente esto se maneja en el entorno)
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt['device']['gpus'])  # Esto es opcional si ya lo gestionas desde el script

import torch
import torch.multiprocessing as mp
from Utils.setup_distributed import setup_distributed
from Datasets.dataset import dataloader
from Models._init_ import *
from Utils.loss import VAELoss
from Scripts.train import entrenamiento
#from Scripts.eval import AllSky_eval

def run_training(rank, world_size, opt):

    # Inicializamos el entorno distribuido para cada proceso
    setup_distributed(rank, world_size)

    print('setup done')

    # Cargamos los data loaders
    LOADER_ENTRENAMIENTO, samplers = dataloader(opt, rank, world_size)
    
    print('Loaders cargados')

    # Creamos el modelo
    model, flops, params = create_model(opt, rank, world_size)

    print(flops, params)
    # Creamos el optimizador y el scheduler
    optimizer, scheduler = create_optimizer_scheduler(opt, model, LOADER_ENTRENAMIENTO, rank, world_size)

    print('Optimizer y scheduler creados')

    # Iniciamos el entrenamiento
    perdida_entrenamiento = entrenamiento(opt, model, VAELoss, optimizer, scheduler, LOADER_ENTRENAMIENTO, samplers, rank, world_size)
    
    print('Entrenamiento completado')

def main():
    # Determinar el número de GPUs disponibles
    world_size = torch.cuda.device_count()

    # Usar mp.spawn para manejar la inicialización de múltiples procesos
    mp.spawn(run_training, args=(world_size, opt), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()




