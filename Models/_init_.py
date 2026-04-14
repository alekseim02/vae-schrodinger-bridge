import torch
import ptflops
import os
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, Adadelta, Adagrad, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
from Models.VAE import VAE

def load_model_weights(model, checkpoint_path, device, strict=True):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    is_wrapped = isinstance(model, torch.nn.parallel.DistributedDataParallel)

    if not is_wrapped and any(k.startswith('module.') for k in state_dict.keys()):
        print("Removing the ‘module.’ prefix from the weights...")
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    elif is_wrapped and not any(k.startswith('module.') for k in state_dict.keys()):
        print("Adding the prefix ‘module.’ to the weights...")
        state_dict = {f'module.{k}': v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=strict)
    print(f"Weights loaded from {checkpoint_path}")

def create_model(opt, rank, world_size):
    torch.cuda.set_device(rank)  
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    model_name = opt['network']['name'] 

    if model_name == 'VAE':
        model = VAE()
    else:
        raise NotImplementedError(f'The {model_name} network is not implemented')

    if rank == 0:
        print(f'Using {model_name}')
        input_size = tuple(opt['datasets']['input_size'])
        flops, params = ptflops.get_model_complexity_info(model, input_size, print_per_layer_stat=False)
        print(f'Computational complexity as a function of input size {input_size}: {flops}')
        print('Number of parameters: ', params)    
    else:
        flops, params = None, None

    model.to(device)

    if opt['train']["checkpoint"]:
        try:
            load_model_weights(model, opt['train']["checkpoint"], device)
        except Exception as e:
            print("Failed to load VAE", opt['train']["checkpoint"])
            print("Error:", e)


    return model, flops, params

def create_optimizer_scheduler(opt, model, loader, rank, world_size):
    optname = opt['train']['optimizer']
    scheduler = opt['train']['lr_scheduler']

    if optname == 'Adam':

        encoder_params = model.module.encoder.parameters() if hasattr(model, 'module') else model.encoder.parameters()
        decoder_params = model.module.decoder.parameters() if hasattr(model, 'module') else model.decoder.parameters()

        optimizer = Adam([
            {'params': encoder_params, 'lr': opt['train']['lr_encoder']},
            {'params': decoder_params, 'lr': opt['train']['lr_decoder']},
        ], weight_decay=opt['train']['weight_decay'])
        
        print('Optimizer Adam with different lr for encoder/decoder')
    elif optname == 'SGD':
        optimizer = SGD(model.parameters(), lr=opt['train']['lr_initial'], weight_decay=opt['train']['weight_decay'])
    elif optname == 'Adadelta':
        optimizer = Adadelta(model.parameters(), lr=opt['train']['lr_initial'], weight_decay=opt['train']['weight_decay'])
    elif optname == 'Adagrad':
        optimizer = Adagrad(model.parameters(), lr=opt['train']['lr_initial'], weight_decay=opt['train']['weight_decay'])
    else:
        optimizer = Adam(model.parameters(), lr=opt['train']['lr_initial'], weight_decay=opt['train']['weight_decay'])
        print(f"Warning: Unrecognized optimizer {optname}. Using Adam by default.")

    if scheduler == 'CosineAnnealing':
        scheduler = CosineAnnealingLR(optimizer, T_max=opt['train']['epochs'], eta_min=opt['train']['eta_min'])
    elif scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1, eta_min=opt['train']['eta_min'])
    else:
        scheduler = None

    return optimizer, scheduler

def save_weights(model, optimizer, scheduler=None, filename="model_weights.pth", rank=0):
    if rank != 0:
        return  

    if not filename.endswith(".pt"):
        filename += ".pt"

    Weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Weights")
    full_path = os.path.join(Weights_dir, filename)
    
    if not os.path.exists(Weights_dir):
        os.makedirs(Weights_dir)

    checkpoint = {
        'model_state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(checkpoint, full_path)
    print(f"Weights successfully saved to {full_path}")


__all__ = ['create_model', 'create_optimizer_scheduler', 'save_weights']
