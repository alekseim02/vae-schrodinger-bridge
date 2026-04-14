import os
from Options.options import parse

path_options = 'Options/vae.yml' 
opt = parse(path_options)   


os.environ["CUDA_VISIBLE_DEVICES"] = str(opt['device']['gpus']) 

import torch
import torch.multiprocessing as mp
from Utils.setup_distributed import setup_distributed
from Datasets.dataset import dataloader
from Models._init_ import *
from Utils.loss import VAELoss
from Scripts.train import entrenamiento


def run_training(rank, world_size, opt):

    setup_distributed(rank, world_size)
    print('setup done')

    LOADER_ENTRENAMIENTO, samplers = dataloader(opt, rank, world_size)
    print('Loaders cargados')

    model, flops, params = create_model(opt, rank, world_size)

    print(flops, params)

    optimizer, scheduler = create_optimizer_scheduler(opt, model, LOADER_ENTRENAMIENTO, rank, world_size)

    print('Optimizer y scheduler creados')

    perdida_entrenamiento = entrenamiento(opt, model, VAELoss, optimizer, scheduler, LOADER_ENTRENAMIENTO, samplers, rank, world_size)
    
    print('Entrenamiento completado')

def main():
    world_size = torch.cuda.device_count()

    mp.spawn(run_training, args=(world_size, opt), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()




