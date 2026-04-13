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

    # Extraer el state_dict
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    # Determinar si hay que usar .module
    is_wrapped = isinstance(model, torch.nn.parallel.DistributedDataParallel)

    # Si los nombres de los pesos tienen 'module.' pero el modelo no está envuelto
    if not is_wrapped and any(k.startswith('module.') for k in state_dict.keys()):
        print("Quitando prefijo 'module.' de los pesos...")
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Si el modelo está envuelto en DDP pero los pesos no lo están
    elif is_wrapped and not any(k.startswith('module.') for k in state_dict.keys()):
        print("Agregando prefijo 'module.' a los pesos...")
        state_dict = {f'module.{k}': v for k, v in state_dict.items()}

    # Cargar los pesos
    model.load_state_dict(state_dict, strict=strict)
    print(f"Pesos cargados desde {checkpoint_path}")

def create_model(opt, rank, world_size):
    torch.cuda.set_device(rank)  # Asignar la GPU correspondiente
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    model_name = opt['network']['name'] 

    # Seleccionar el modelo basado en el nombre
    if model_name == 'VAE':
        model = VAE()
    else:
        raise NotImplementedError(f'La red {model_name} no está implementada')

    # Estimación de la complejidad y el número de operaciones
    if rank == 0:
        print(f'Usando la red {model_name}')
        input_size = tuple(opt['datasets']['input_size'])
        flops, params = ptflops.get_model_complexity_info(model, input_size, print_per_layer_stat=False)
        print(f'Complejidad computacional con entrada de tamaño {input_size}: {flops}')
        print('Número de parámetros: ', params)    
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

    # Crear el optimizador
    if optname == 'Adam':
        # En DDP el módulo real es model.module, si no usas DDP quita .module
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
        print(f"Advertencia: Optimizer {optname} no reconocido. Usando Adam por defecto.")

    # Crear el scheduler
    if scheduler == 'CosineAnnealing':
        scheduler = CosineAnnealingLR(optimizer, T_max=opt['train']['epochs'], eta_min=opt['train']['eta_min'])
    elif scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1, eta_min=opt['train']['eta_min'])
    else:
        scheduler = None

    return optimizer, scheduler

def save_weights(model, optimizer, scheduler=None, filename="model_weights.pth", rank=0):
    if rank != 0:
        return  # Solo el proceso con rank 0 guarda los pesos

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
    print(f"Pesos guardados exitosamente en {full_path}")


__all__ = ['create_model', 'create_optimizer_scheduler', 'save_weights']
