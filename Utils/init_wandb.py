#import os
import wandb

def init_wandb(opt):
    
    if opt['wandb']['init']:
        wandb.login()
        wandb.init(
            # set the wandb project where this run will be logged
            project=opt['wandb']['project'], entity=opt['wandb']['entity'], 
            name=opt['wandb']['name'], save_code=opt['wandb']['save_code'],
            resume = opt['wandb']['resume'],
            id = opt['wandb']['id']
        )       

__all__ = ['init_wandb']
