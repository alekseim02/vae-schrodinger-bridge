from tqdm import tqdm
import torch
import wandb

def train(epochs, dataloader, batchsize, model, optimizer, scheduler, device, val_dataloader=None):
    train_loss = []
    train_mse = []
    criterion = torch.nn.MSELoss()
    model.train()

    for period in range(epochs):
        running_loss = 0.0

        for batch in tqdm(dataloader, desc=f"period {period}"):
            batch = batch.to(device)
            batch_noise = torch.randn_like(batch, device=device)  # Ruido aleatorio del mismo tamaño que el batch
            current_batch_size = batch.size(0)
            # Print sample values from batch_noise (ruido) and batch (caras)
            # print("Cara sample:", batch[0, 1, :2, :2].cpu().detach().numpy())
            # input("Presiona cualquier tecla para continuar...")
            # print(f"Batch latents min: {batch[0].min().item()}, max: {batch[0].max().item()}")
            # input("Presiona cualquier tecla pa)ra continuar...2")
            x_0 = batch_noise
            x_1 = batch
            alpha = torch.rand(current_batch_size, device=device)
            x_alpha = (1 - alpha[:, None, None, None]) * x_0 + alpha[:, None, None, None] * x_1

            optimizer.zero_grad()
            pred = model(x_alpha, alpha)
            target = x_1 - x_0
            loss = criterion(pred, target)
            # Chequeo de NaN o infinito en la pérdida
            if not torch.isfinite(loss):
                print(f"[ADVERTENCIA] Loss no finita en entrenamiento. Batch ignorado. Valor: {loss.item()}")
                wandb.log({"loss_nan": 1, "epoch": period})
                continue
            loss.backward()
            optimizer.step()

            trainmse = loss.detach()
            train_loss.append(loss.item())
            train_mse.append(trainmse.item())
            running_loss += loss.item()

            wandb.log({
                "loss": loss.item(),
                "train_mse": trainmse.item(),
                "epoch": period
            })

        avg_loss = running_loss / len(dataloader)

        # VALIDACIÓN
        if val_dataloader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_dataloader:
                    batch = batch.to(device)
                    batch_noise = torch.randn_like(batch, device=device) 
                    current_batch_size = batch.size(0)
                    x_0 = batch_noise
                    x_1 = batch
                    alpha = torch.rand(current_batch_size, device=device)
                    x_alpha = (1 - alpha[:, None, None, None]) * x_0 + alpha[:, None, None, None] * x_1
                    pred = model(x_alpha, alpha)
                    target = x_1 - x_0
                    loss = criterion(pred, target)
                    # Chequeo de NaN o infinito en la pérdida de validación
                    if not torch.isfinite(loss):
                        print(f"[ADVERTENCIA] Loss no finita en validación. Batch ignorado. Valor: {loss.item()}")
                        wandb.log({"val_loss_nan": 1, "epoch": period})
                        continue
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_dataloader)
            scheduler.step(avg_val_loss)
            wandb.log({"val_loss": avg_val_loss, "epoch": period})
            print(f"Epoch {period}: Validation loss: {avg_val_loss:.4f}")
        else:
            scheduler.step(avg_loss)
            wandb.log({"val_loss": avg_loss, "epoch": period})  # 🛠 loguea avg_loss como si fuera val
            print(f"Epoch {period}: Validation loss (train fallback): {avg_loss:.4f}")

        # Log LR y guardar pesos
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"learning_rate": current_lr, "epoch": period})

        if (period + 1) % 100 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"Weights/Diffusion_perc_epoch_{period+1}.pt")

        torch.cuda.empty_cache()
        model.train()
