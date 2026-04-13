import argparse
import os
import time
import warnings

import torch
from torchvision.utils import save_image

from Options.options import parse
from Utils.utils_sinkhorn import (
    calculate_pt,
    calculate_potentials,
    generate,
    sample_noise,
)
from Models.Encoder import Encoder
from Models.Decoder import Decoder


def Sinkhorn(cfg,device):

    if cfg['mode'] == 'calculate_pt':

        calculate_pt(batch_size=cfg['batch_size'], image_dir=cfg['image_dir_celeba'], checkpoint=cfg['ckpt'], device=device, n_samples=cfg['n_target_pt'], data='celeba')

    if cfg['mode'] == 'calculate_potentials':

        calculate_potentials(eps=cfg['eps'], n_source=cfg['n_source_potentials'],
                              n_target=cfg['n_target_potentials'], device=device, iters_max=cfg['iters_max'])
        
    if cfg['mode'] == 'generate':
        n_generated = cfg['n_generated']
        n_source = cfg['n_source_generate']
        n_target = cfg['n_target_generate']
        eps = cfg['eps_generate']
        tau = cfg['tau']
        Nsteps= cfg['Nsteps']
        checkpoint = cfg['ckpt_generate']


        try:
            checkpoint_data = torch.load(checkpoint, map_location=device)
        except FileNotFoundError:
            raise RuntimeError(f"[ERROR] Checkpoint no encontrado: {checkpoint}")
        except Exception as e:
            raise RuntimeError(f"[ERROR] Fallo al cargar checkpoint: {e}")

        try:
            full_state_dict = checkpoint_data['model_state_dict']
        except KeyError:
            raise RuntimeError("[ERROR] El checkpoint no contiene 'model_state_dict'")
        

        # load encoder
        encoder = Encoder().to(device)
        encoder_state_dict = {
            k.replace("module.", ""): v
            for k, v in full_state_dict.items()
            if "encoder." in k
        }
        try:
            encoder.load_state_dict(encoder_state_dict, strict=True)
        except RuntimeError as e:
            raise RuntimeError(f"[ERROR] Error cargando Encoder: {e}")
        encoder.eval() 

        # load decoder
        decoder = Decoder().to(device)
        decoder_state_dict = {
            k.replace("module.", ""): v
            for k, v in full_state_dict.items()
            if "decoder." in k
        }
        try:
            decoder.load_state_dict(decoder_state_dict, strict=True)
        except RuntimeError as e:
            raise RuntimeError(f"[ERROR] Error cargando Decoder: {e}")
        decoder.eval() 

        # Obtain matrix target distribution
        file_path_caras = f'./Latents/latents_{n_target}_celeba.pt' # --> list of tensors [1, 1024]
        if not os.path.exists(file_path_caras):
            print("Warning: The file", file_path_caras, "does not exist. Make sure you have changed the epsilon in sinkhorn.yml.")
            return 
        data_caras = torch.load(file_path_caras)
        matrix_caras = torch.cat(data_caras, dim=0) # --> concatenate the list of tensors --> matrix [n_source, 1024]
        matrix_caras = matrix_caras.to(device)

        if 'pot_path' in cfg and cfg['pot_path'] is not None:

            file_path_pot = cfg['pot_path']
        else:
            file_path_pot = f"./Potentials/logv_{n_source}_{n_target}_{eps}_discrete.pt"

        try:
            pot = torch.load(file_path_pot, map_location=device)
        except Exception as e:
            raise RuntimeError(f"[ERROR] No se pudo cargar potencial: {file_path_pot} | {e}")

        assert isinstance(pot, dict), "Potential file must be a dict"
        assert pot["type"] == "discrete", "Only discrete potentials supported"
        assert "logv" in pot, "Potential file must contain key 'g'"
        assert "eps" in pot, "Potential file must contain key 'eps'"

        logv = pot["logv"].to(device)
        
        eps_pot = float(pot["eps"])
        print(f"[INFO] eps from potential: {eps_pot}")

        pot_name = os.path.splitext(os.path.basename(file_path_pot))[0]
        save_folder = f'Images/{pot_name}_eps{eps}_tau{tau}_Ns{Nsteps}_N{n_generated}/'

        os.makedirs(save_folder, exist_ok=True)


        total_time = 0.0  # Inicializa el acumulador de tiempo
        for i in range(n_generated):
            noise = sample_noise(1, 1024, device, seed=i)
            start = time.perf_counter()
            with torch.no_grad():
                final_image = generate(
                    noise,
                    decoder,
                    matrix_caras,
                    logv,
                    eps,
                    tau,
                    Nsteps,
                    device
                )
            end = time.perf_counter()
            elapsed = end - start
            total_time += elapsed
            filename = os.path.join(save_folder, f"{i:06d}.png")
            save_image(final_image, filename)
            del final_image
        avg_time = total_time / n_generated if n_generated > 0 else 0
        print(f"Tiempo promedio de inferencia por imagen: {avg_time:.4f} segundos")
        print(f"Imagenes generadas en: {save_folder}")
        

def prompt_param(param, default):
    print(f"\n[INPUT] Parámetro: {param}")
    val = input(f"Introduce valor (deja vacío para usar el valor por defecto [{default}]): ")
    if val == "":
        return default
    try:
        return type(default)(val)
    except Exception:
        return val

def force_types(cfg, mode):
    type_map = {
        'generate': {
            'n_generated': int,
            'n_source_generate': int,
            'n_target_generate': int,
            'eps_generate': float,
            'tau': float,
            'Nsteps': int,
        },
        'calculate_pt': {
            'batch_size': int,
            'n_source_pt': int,
            'n_target_pt': int,
        },
        'calculate_potentials': {
            'n_source_potentials': int,
            'n_target_potentials': int,
            'eps': float,
            'iters_max': int,
        }
    }
    for key, typ in type_map.get(mode, {}).items():
        if key in cfg:
            try:
                cfg[key] = typ(cfg[key])
            except Exception:
                pass
    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sinkhorn pipeline runner")
    parser.add_argument('--mode', type=str, help="Modo de operación: generate, calculate_pt, calculate_potentials, vae_generate, vae_reconstruct, etc.")
    parser.add_argument('--manual', action='store_true', help="Si se activa, pide los parámetros por consola en vez de usar el YAML")
    parser.add_argument('--config', type=str, default='Options/sinkhorn.yml', help="Ruta al archivo de configuración YAML")
    parser.add_argument('--pot_path', type=str, default=None,
                    help="Ruta al potencial OT (.pt). Si no se pasa, usa el YAML")

    args = parser.parse_args()

    # Usar el archivo de configuración pasado por --config
    path_options = args.config
    yaml_cfg = parse(path_options)
    cfg = yaml_cfg.copy()  # Usar copia para no modificar el original
    GPU = cfg.get('device', 0)
    device = torch.device(f'cuda:{GPU}' if torch.cuda.is_available() else "cpu")
    if args.pot_path is not None:
        cfg['pot_path'] = args.pot_path
    if args.mode is not None:
        cfg['mode'] = args.mode
        mode_params = {
            'calculate_pt': ['n_source_pt', 'n_target_pt'],
            'calculate_potentials': ['n_source_potentials', 'n_target_potentials', 'eps'],
            'generate': ['n_generated', 'n_source_generate', 'n_target_generate', 'eps_generate', 'tau', 'Nsteps'],
        }
        params = mode_params.get(args.mode, [])
        if args.manual:
            print("[INFO] Modo manual activado: introduce los parámetros por consola (deja vacío para usar el valor por defecto).\n")
            for param in params:
                if param in cfg:
                    cfg[param] = prompt_param(param, cfg[param])
            cfg = force_types(cfg, args.mode)
    # Si no se pasa modo, usa todo del YAML
    Sinkhorn(cfg,device)

