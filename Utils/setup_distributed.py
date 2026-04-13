import os
import gc
import torch
import torch.distributed as dist

def setup_distributed(rank, world_size):
    """Inicializa el entorno distribuido si hay múltiples GPUs."""
    
    # Configurar las variables de entorno necesarias
    os.environ['MASTER_ADDR'] = 'localhost'  # Dirección del nodo maestro
    os.environ['MASTER_PORT'] = '29500'     # Puerto para la comunicación

    # Iniciar el grupo de procesos distribuidos
    dist.init_process_group(
        backend="nccl",  # Usamos NCCL para la comunicación entre GPUs
        init_method="env://",  # Utilizamos las variables de entorno para la sincronización
        world_size=world_size,  # Número total de procesos
        rank=rank  # El identificador único del proceso
    )
    
    # Configurar el dispositivo para el proceso actual
    #torch.cuda.set_device(rank)  # Aseguramos que cada proceso use la GPU correspondiente al rank
    
    # Imprimir información para depuración: qué proceso está usando qué GPU
    print(f"Proceso {rank} de {world_size} está usando la GPU {rank}")

def cleanup():
    dist.destroy_process_group()

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()       # Libera caché de memoria GPU
    torch.cuda.ipc_collect()