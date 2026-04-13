import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class VAELoss(nn.Module):
    def __init__(self,
                 recon_loss_type='mse',
                 beta_max=1.0,
                 kl_annealing_epochs=10,
                 free_bits=0.5,
                 sigmoid_midpoint=50,
                 sigmoid_steepness=10):
        """
        recon_loss_type: 'mse' or 'bce'
        beta_max: maximum beta value
        kl_annealing_epochs: epochs to linearly ramp beta (if using linear)
        free_bits: threshold gamma in nats per dimension
        sigmoid_midpoint: epoch at which beta reaches half of beta_max
        sigmoid_steepness: controls slope of sigmoid
        """
        super().__init__()
        self.recon_loss_type = recon_loss_type.lower()
        self.beta_max = beta_max
        self.kl_annealing_epochs = kl_annealing_epochs
        self.free_bits = free_bits
        self.sigmoid_midpoint = sigmoid_midpoint
        self.sigmoid_steepness = sigmoid_steepness
        self.beta = 0.0

    def update_beta(self, epoch):
        # Sigmoidal schedule for beta
        exp_term = math.exp(-self.sigmoid_steepness * (epoch - self.sigmoid_midpoint) / self.sigmoid_midpoint)
        self.beta = self.beta_max / (1 + exp_term)

    def forward(self, recon_x, x, mu, logvar, epoch=None):
        batch_size = x.size(0)

        # Compute reconstruction loss
        if self.recon_loss_type == 'bce':
            recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum') / batch_size
        elif self.recon_loss_type == 'mse':
            recon_loss = F.mse_loss(recon_x, x, reduction='sum') / batch_size
        else:
            raise ValueError("recon_loss_type must be 'mse' or 'bce'.")

        # Compute KL divergence per dimension
        var = torch.exp(logvar)
        kl_per_dim = 0.5 * (mu.pow(2) + var - 1 - logvar)  # shape: [batch, latent_dim]

        # Apply free bits: only penalize KL above the gamma threshold
        free_bits_tensor = torch.full_like(kl_per_dim, self.free_bits)
        kl_penalty = torch.sum(torch.clamp(kl_per_dim - free_bits_tensor, min=0.0)) / batch_size

        # Optionally update beta if epoch is provided
        if epoch is not None:
            self.update_beta(epoch)

        # Total loss
        total_loss = recon_loss + self.beta * kl_penalty
        return total_loss, recon_loss, kl_penalty, self.beta


__all__ = ['VAELoss']
