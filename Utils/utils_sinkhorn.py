import os  
import torch
import ot
from torch.utils.data import Subset, DataLoader
from Models.Encoder import Encoder
from Models.Decoder import Decoder
from Utils.data import ImageDataset, CelebATransform


def calculate_pt(batch_size, image_dir, checkpoint, device, n_samples, data='celeba'):
    dataset = ImageDataset(image_dir, transform=CelebATransform(image_size=128))
    print("Dataset size:", len(dataset))
    print("Requested samples:", n_samples)
    indices = list(range(n_samples))
    subset = Subset(dataset, indices)

    checkpoint = torch.load(checkpoint, map_location=device)

    model_state_dict = checkpoint['model_state_dict']

    model = Encoder().to(device)

    encoder_state_dict = {
        k.replace("module.", ""): v
        for k, v in model_state_dict.items()
        if "encoder." in k
    }

    model.load_state_dict(encoder_state_dict, strict=True)
    model.eval()

    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    resultados = []

    with torch.inference_mode():  
        for batch in dataloader:
            batch = batch.to(device)
            latent_vector = model(batch)
            resultados.append(latent_vector.cpu())

    os.makedirs("Latents", exist_ok=True)  
    torch.save(resultados, f'Latents/latents_{n_samples}_{data}.pt')


def sinkhorn_potentials(source, target, eps, n_source, n_target, device, iters_max):
    """
    Compute the dual potentials for the optimal transport problem
    with distributions of different sizes.
    """
 
    a = torch.ones((n_source,), device=device) / n_source
    b = torch.ones((n_target,), device=device) / n_target


    M = torch.cdist(source, target) ** 2 / 2.0  


    M_np = M.detach().cpu().numpy()
    a_np = a.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()

    _, log_ = ot.sinkhorn(
        a_np,
        b_np,
        M_np,
        eps,
        method='sinkhorn_log',
        log=True,
        stopThr=1e-9,
        numItermax=iters_max
    )

    logv_opt = torch.tensor(log_['log_v'], dtype=torch.float32, device=device)  
    logu_opt = torch.tensor(log_['log_u'], dtype=torch.float32, device=device)  

    os.makedirs("Potentials", exist_ok=True)  
    torch.save(  
        {
            "type": "discrete",
            "logv": logv_opt,
            "logu": logu_opt,
            "eps": float(eps),
            "n_source": int(n_source),
            "n_target": int(n_target),
        },
        f'Potentials/logv_{n_source}_{n_target}_{eps}.pt'
    )


def calculate_potentials(eps, n_source, n_target, device, iters_max):
    file_path_caras = f'./Latents/latents_{n_target}_celeba.pt'
    
    data_caras = torch.load(file_path_caras, map_location='cpu')  

    matrix_caras = torch.cat(data_caras, dim=0)
    matrix_noise = sample_noise(n_source, matrix_caras.shape[1], device, seed=0)
    matrix_caras = matrix_caras.to(device)

    
    sinkhorn_potentials(
        source=matrix_noise,
        target=matrix_caras,
        eps=eps,
        n_source=n_source,
        n_target=n_target,
        device=device,
        iters_max=iters_max
    )


class ent_drift:
    def __init__(self, data, potential, eps):
        self.data = data
        self.potential = potential
        self.eps = eps

        assert self.potential.shape[0] == self.data.shape[0], \
            "Mismatch between potential size and target samples"

    def estimator(self, x, t):
        t = max(float(t), 1e-6)  

        M = torch.cdist(x, self.data) ** 2 / (2.0 * t)

        K = -M / self.eps + self.potential[None, :]  

        gammaz = -torch.max(K, dim=1).values  
        K_shift = K + gammaz.reshape(-1, 1)

        exp_ = torch.exp(K_shift)
        top_ = exp_ @ self.data
        bot_ = exp_.sum(dim=1, keepdim=True)  

        drift_without_x = top_ / bot_

        return (-x + drift_without_x) / t

    def __call__(self, x, t):
        return self.estimator(x, t)


def sample(latent_noise, matrix_caras, logv_opt, device, eps, tau, Nsteps):
    """
    Sampling using the drift estimator and discrete Langevin method.
    """
    dt = tau / Nsteps
    sigma = torch.sqrt(torch.tensor(dt * eps, device=device)) 

    x = latent_noise.to(device).clone()
    drift_estimator = ent_drift(matrix_caras, logv_opt, eps)

    for k in range(Nsteps):
        t = k * dt
        bteps_x = drift_estimator(x, 1.0 - t) 
        eta = torch.randn_like(x)  

        x = x + (dt * bteps_x) + sigma * eta

    return x  


def generate(noise, decoder, matrix_caras, logv_opt, eps, tau, Nsteps, device):
    latent = sample(noise, matrix_caras, logv_opt, device, eps, tau, Nsteps)
    return decoder(latent)

def sample_noise(n, d, device, seed=None):
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
    return torch.randn(n, d, device=device, generator=gen)