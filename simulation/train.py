import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_autoencoder(model, train_loader, val_loader, optimizer, device, num_epochs=20):
    criterion = nn.BCELoss()
    model.to(device)

    train_losses = []
    val_losses = []

    epoch_bar = tqdm(range(num_epochs), desc="Training", leave=True)

    for epoch in epoch_bar:
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            x = batch[0].to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, x)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * x.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                output = model(x)
                loss = criterion(output, x)
                total_val_loss += loss.item() * x.size(0)

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        # Logging
        epoch_bar.set_description(f"Epoch {epoch+1}/{num_epochs}")
        epoch_bar.set_postfix(train_loss=f"{avg_train_loss:.4f}", val_loss=f"{avg_val_loss:.4f}")

    # Plot loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("BCELoss")
    plt.title("Autoencoder Training Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return train_losses, val_losses

def train_rebeca_prior(
    model,
    trainloader,
    optimizer,
    noise_scheduler,
    scheduler=None,
    device="cuda",
    num_epochs=20,
    print_every=1,
):
    model.train()
    loss_fn = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress = tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

        for batch in progress:
            optimizer.zero_grad(set_to_none=True)

            enc_image_tensor = batch[0][:, :16].to(device)         # (B, latent_dim)
            user_ids_tensor = batch[0][:, -1].long().to(device)    # (B,)
            ratings_tensor = batch[0][:, -2].long().to(device)     # (B,)

            noise = torch.randn_like(enc_image_tensor)             # (B, latent_dim)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (enc_image_tensor.size(0),),
                device=device
            ).long()

            # Diffusion step
            noisy = noise_scheduler.add_noise(enc_image_tensor, noise, timesteps)

            # Prediction
            pred = model(noisy, timesteps, user_ids_tensor, ratings_tensor)

            # Loss
            loss = loss_fn(pred, noise)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * enc_image_tensor.size(0)
            progress.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(trainloader.dataset)

        if print_every and (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - avg loss: {avg_loss:.4f}")

        if scheduler is not None:
            scheduler.step(avg_loss)
    return model