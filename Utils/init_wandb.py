#import os
import wandb

def init_wandb(opt):

    #os.environ['http_proxy'] = 'http_proxy=http://proxy.cidaut.es:3128'
    #os.environ['https_proxy'] = 'http_proxy=http://proxy.cidaut.es:3128'
    
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
