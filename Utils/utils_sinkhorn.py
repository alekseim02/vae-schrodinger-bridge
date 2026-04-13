import torch
from torch.utils.data import Subset, DataLoader
from Models.Encoder import Encoder
from Utils.data import ImageDataset, CelebAi
from torchvision.utils import save_image
import os

def calculate_pt(batch_size, image_dir, checkpoint, device, n_samples, data = "celeba"):

    dataset = ImageDataset(image_dir, transform=CelebAi(image_size=128))

    indices = list(range(n_samples)) 
    subset = Subset(dataset, indices)

    model = Encoder().to(device)
    checkpoint = torch.load(checkpoint, map_location=device)

    model_state_dict = checkpoint['model_state_dict']
    # Checkpoint trainned in DDP, remove .module
    encoder_state_dict = {
        k.replace("module.", ""): v
        for k, v in model_state_dict.items()
        if "encoder." in k
    }

    model.load_state_dict(encoder_state_dict, strict=True)
    model.eval()

    # Crear DataLoader
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    # Lista para guardar los tensores procesados
    resultados = []

    # Procesar el dataset
    with torch.no_grad():
        for batch in dataloader:

            batch = batch.to(device)
            latent_vector= model(batch)
            resultados.append(latent_vector.cpu())

    torch.save(resultados, f'Latents/latents_{n_samples}_{data}.pt')

def sinkhorn_vanilla_torch(a, b, M, eps, max_iters=20000, tol=1e-9, verbose=True):
    device = M.device
    n, m = M.shape

    K = torch.exp(-M / eps)

    u = torch.ones(n, device=device) / n
    v = torch.ones(m, device=device) / m

    for it in range(max_iters):
        Kv = K @ v
        Kv = torch.clamp(Kv, min=1e-300)
        u_new = a / Kv

        KTu = K.t() @ u_new
        KTu = torch.clamp(KTu, min=1e-300)
        v_new = b / KTu

        err = torch.norm(u_new - u, p=1)
        u, v = u_new, v_new

        if verbose and it % 500 == 0:
            print(f"[{it}] err_u={err:.2e}")

        if err < tol:
            break

    return v


def calculate_potentials(eps, n_source, n_target, device, iters_max):

    file_path_caras = f'./Latents/latents_{n_target}_celeba.pt'
    data_caras = torch.load(file_path_caras)
    matrix_caras = torch.cat(data_caras, dim=0).to(device)  # (n_target, d)

    # fuente: ruido en el latente
    matrix_noise = sample_noise(n_source, matrix_caras.shape[1], device, seed=0)

    # distribuciones empíricas
    a = torch.ones(n_source, device=device) / n_source
    b = torch.ones(n_target, device=device) / n_target

    # coste cuadrático
    d = matrix_caras.shape[1]
    M = torch.cdist(matrix_noise, matrix_caras) ** 2 / d

    # Sinkhorn vanilla
    v = sinkhorn_vanilla_torch(
        a=a,
        b=b,
        M=M,
        eps=eps,
        max_iters=iters_max,
        tol=1e-9,
        verbose=True
    )
    logv = eps * torch.log(v + 1e-12)

    torch.save({
    "type": "discrete",
    "logv": logv.detach(),
    "eps": eps,
    "n_source": n_source,
    "n_target": n_target
}, f'Potentials/logv_{n_source}_{n_target}_{eps}_discrete.pt')


class ent_drift:
    def __init__(self, data, v, eps):
        self.data = data      # (n_target, d)
        self.v = v            # (n_target,)
        self.eps = eps

    def __call__(self, x, t):
        one_minus_t = max(1e-6, 1.0 - t)

        dist2 = torch.cdist(x, self.data) ** 2

        exponent = -dist2 / (2.0 * self.eps * one_minus_t )

        max_exp = exponent.max(dim=1, keepdim=True).values
        exp_shifted = torch.exp(exponent - max_exp)

        weights = self.v[None, :] * exp_shifted

        Z = weights.sum(dim=1, keepdim=True)
        bary = (weights @ self.data) / Z

        drift = (-x + bary) / one_minus_t
        return drift


def sample(latent_noise, matrix_caras, g, device, eps, tau, Nsteps, decoder=None, save_intermediate=False, save_folder=None, save_grid=False, grid_output=None, grid_steps=10, step_interval=50):
    """
    Realiza el proceso de muestreo utilizando el estimador de drift y el método de Langevin.
    Si save_intermediate=True y decoder está definido, guarda imágenes intermedias en save_folder.
    step_interval: guarda imágenes cada step_interval pasos, empezando en el paso 1.
    """
    dt = tau / Nsteps  
    x = latent_noise.clone()

    sigma = torch.sqrt(torch.tensor(dt*eps, device = device))
    drift_estimator = ent_drift(matrix_caras, g, eps)
    
    for k in range(1, Nsteps + 1):
        t = k * dt

        drift = drift_estimator(x,1-t)
        noise = torch.randn_like(x)

        x = x + dt * drift + sigma * noise 
        
        # Guardar imágenes intermedias cada step_interval, empezando en el paso 1
        if save_intermediate and decoder is not None and save_folder is not None:
            if k == 1 or (k % step_interval == 0) or k == Nsteps:
                img = decoder(x)
                save_path = os.path.join(save_folder, f'step_{k:03d}.png')
                save_image(img, save_path)

    return x
    
def generate(noise, decoder, matrix_caras, g, eps, tau, Nsteps, device):
    latent = sample(
        noise,
        matrix_caras,
        g,
        device,
        eps,
        tau,
        Nsteps
    )

    return decoder(latent)

def sample_noise(n, d, device, seed=None):
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
    return torch.randn(n, d, device=device, generator=gen)



