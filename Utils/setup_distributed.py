import os
import gc
import torch
import torch.distributed as dist

def setup_distributed(rank, world_size):
    """Initialize the distributed environment if there are multiple GPUs."""
    
    os.environ['MASTER_ADDR'] = 'localhost'  
    os.environ['MASTER_PORT'] = '29500'     


    dist.init_process_group(
        backend="nccl",  
        init_method="env://",  
        world_size=world_size,  
        rank=rank  
    )
    print(f"Process {rank} of {world_size} is using GPU {rank}")

def cleanup():
    dist.destroy_process_group()

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()      
    torch.cuda.ipc_collect()