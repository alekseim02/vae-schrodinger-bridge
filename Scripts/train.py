import torch    
import wandb
import numpy as np 
from sklearn.manifold import TSNE
import os
import torch.distributed as dist
from Utils.loss import VAELoss
from Utils.plot import *
from Models._init_ import save_weights
from Utils.init_wandb import init_wandb
from Utils.setup_distributed import cleanup, clear_memory
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Datasets.dataset import shuffle_sampler
import matplotlib as plt

def entrenamiento(opt, model, criterion, optimizer, scheduler, LOADER_ENTRENAMIENTO, samplers, rank, world_size):

    best_loss = 10000 
    num_epochs = opt['train']['epochs']
    log_interval = opt['train']['verbose']

    train_losses = []
    train_recon_losses = []
    train_KL_losses = []

    if opt['train']['loss'] == 'VAELoss':
        # Instanciamos la clase
        criterion = VAELoss(recon_loss_type='mse', beta_max=4.0, kl_annealing_epochs=50, free_bits=1.0, sigmoid_midpoint=50, sigmoid_steepness=10)
    else:
        raise ValueError(f"Función de pérdida no reconocida: {opt['train']['loss']}")

    if rank == 0:
        init_wandb(opt)

    for epoch in range(num_epochs):

        print('Epoca actual:', epoch)
        shuffle_sampler(samplers, epoch)
        print('Samplers distribuidos')
        
        model.train()  
        
        criterion.update_beta(epoch)

        running_loss_train = 0.0
        running_recon_loss_train = 0.0
        running_KL_loss_train = 0.0

        for batch_idx, images in enumerate(LOADER_ENTRENAMIENTO):

            images = images.to(f'cuda:{rank}')

            optimizer.zero_grad()

            reconstruct_image, mean, log_variance = model(images)

            loss_train, recon_loss, kl_div, beta = criterion(reconstruct_image, images, mean, log_variance, epoch)

            loss_train.backward()

            optimizer.step()

            running_loss_train += loss_train.item()
            running_recon_loss_train += recon_loss.item()
            running_KL_loss_train += kl_div.item()

            global_step = epoch * len(LOADER_ENTRENAMIENTO) + batch_idx

            dist.barrier()

            if rank == 0:
                wandb.log({
                    'Batch Loss': loss_train.item(),
                    'Batch MSE': recon_loss.item(),
                    'Batch KL': kl_div.item(),
                    'Latent μ mean (0)':  mean.mean().item(),
                    'Latent μ std (1)':   mean.std().item(),
                    'Latent logvar mean (0)': log_variance.mean().item(),
                    'Latent logvar std (+-2)':  log_variance.std().item(),
                    'Latent σ² mean (1)': log_variance.exp().mean().item(),
                    'Latent σ² std (no 0 o inf)':  log_variance.exp().std().item(),
                    'beta': beta,
                }, step=global_step)

                if batch_idx % log_interval == 0:
                    for bi in range(min(2, images.size(0))):
                        inp = images[bi].permute(1,2,0).cpu().numpy().clip(0,1)
                        out = reconstruct_image[bi].permute(1,2,0).cpu().detach().numpy().clip(0,1)
                        wandb.log({
                            f"Input image {bi}":         wandb.Image(inp),
                            f"Reconstructed image {bi}": wandb.Image(out),
                        }, step=global_step)

        running_loss_train_tensor = torch.tensor(running_loss_train, dtype=torch.float32, device=f'cuda:{rank}')

        torch.distributed.reduce(running_loss_train_tensor, dst=0, op=torch.distributed.ReduceOp.SUM)

        running_recon_loss_train_tensor = torch.tensor(running_recon_loss_train, dtype=torch.float32, device=f'cuda:{rank}')

        torch.distributed.reduce(running_recon_loss_train_tensor, dst=0, op=torch.distributed.ReduceOp.SUM)

        running_KL_loss_train_tensor = torch.tensor(running_KL_loss_train, dtype=torch.float32, device=f'cuda:{rank}')

        torch.distributed.reduce(running_KL_loss_train_tensor, dst=0, op=torch.distributed.ReduceOp.SUM)

        if rank == 0:
            running_loss_train = running_loss_train_tensor.item()
            running_recon_loss_train = running_recon_loss_train_tensor.item()
            running_KL_loss_train = running_KL_loss_train_tensor.item()

        dist.barrier()

        if rank == 0:

            avg_train_loss = running_loss_train / len(LOADER_ENTRENAMIENTO)

            train_losses.append(avg_train_loss)

            avg_train_recon_loss = running_recon_loss_train / len(LOADER_ENTRENAMIENTO)

            train_recon_losses.append(avg_train_recon_loss)

            avg_train_KL_loss = running_KL_loss_train / len(LOADER_ENTRENAMIENTO)

            train_KL_losses.append(avg_train_KL_loss)

            perdida_entrenamiento = running_loss_train / len(LOADER_ENTRENAMIENTO)

            print(f"Época [{epoch+1}/{num_epochs}], Perdida: {perdida_entrenamiento:.4f}, Perdida MSE: {avg_train_recon_loss:.4f}, Perdida KL divergence: {avg_train_KL_loss:.4f}")

            if perdida_entrenamiento < best_loss:
                best_loss = perdida_entrenamiento
                save_weights(
                    model,
                    optimizer,
                    scheduler,
                    opt['network']['save_weights'],  # p.ej. 'checkpoint_best.pth'
                    rank=0
                )
                print("Mejor modelo guardado")

            last_path = opt['network']['save_weights'].replace('_best', '_last')
            save_weights(
                model,
                optimizer,
                scheduler,
                last_path,           # p.ej. 'checkpoint_last.pth'
                rank=0
            )
            print(f"Pesos de la última época guardados en {last_path}")

        dist.barrier()

        clear_memory()

    cleanup()

    return perdida_entrenamiento

