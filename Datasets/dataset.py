import os
import torch.distributed as dist
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, DistributedSampler
from Utils.data import *
from Utils.plot import *

def dataloader(opt, rank, world_size):

    root_path = opt['root_path']
    entrenamiento_path = opt['datasets']['entrenamiento']['entrenamiento_path']

    full_path_entrenamiento = os.path.join(root_path, entrenamiento_path)

    transform = None

    samplers = []

    if opt['datasets']['entrenamiento']['transform'] == 'CelebAi':
        transform = CelebAi()

    if transform is None:
        print("Advertencia: No se ha definido una transformación. Usando una transformación por defecto.")
        transform = transforms.ToTensor()

    dataset_ent = ImageDataset(image_dir=full_path_entrenamiento, transform=transform)

    print('Dataset info:')
    print('\t Imágenes train:', len(dataset_ent))
    print('world size:', world_size)

    if world_size > 1:

        sampler_entrenamiento = DistributedSampler(dataset_ent, num_replicas=world_size, shuffle= True, rank=rank, drop_last=True)

        LOADER_ENTRENAMIENTO = DataLoader(dataset_ent, batch_size=opt['datasets']['entrenamiento']['batch_size_entrenamiento'], shuffle=False, pin_memory=True, sampler=sampler_entrenamiento)
        
        samplers.append(sampler_entrenamiento)

    else:

        LOADER_ENTRENAMIENTO = DataLoader(dataset_ent, batch_size=opt['datasets']['entrenamiento']['batch_size_entrenamiento'], shuffle=True, pin_memory=True)

        samplers = None

    return LOADER_ENTRENAMIENTO, samplers

def shuffle_sampler(samplers, epoch):
    '''
    A function that shuffles all the Distributed samplers in the loaders.
    '''
    if not samplers: # if they are none
        return
    for sampler in samplers:
        sampler.set_epoch(epoch)
