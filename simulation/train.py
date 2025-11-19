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



def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss function combining reconstruction loss and KL divergence.
    
    Args:
        recon_x: reconstructed images
        x: original images
        mu: mean of latent distribution
        logvar: log variance of latent distribution
        beta: weight for KL divergence (for beta-VAE)
    
    Returns:
        total_loss, recon_loss, kl_loss
    """
    # Reconstruction loss (Binary Cross Entropy for images in [0,1])
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss

def train_vae(model, train_loader, val_loader, optimizer, device, num_epochs=20, beta=1.0):
    """
    Train a VAE model.
    
    Args:
        model: VAE model
        train_loader: training data loader
        val_loader: validation data loader
        optimizer: optimizer
        device: device to train on
        num_epochs: number of training epochs
        beta: weight for KL divergence (for beta-VAE)
    """
    model.to(device)

    train_losses = []
    val_losses = []
    train_recon_losses = []
    train_kl_losses = []
    val_recon_losses = []
    val_kl_losses = []

    epoch_bar = tqdm(range(num_epochs), desc="Training VAE", leave=True)

    for epoch in epoch_bar:
        model.train()
        total_train_loss = 0.0
        total_train_recon_loss = 0.0
        total_train_kl_loss = 0.0

        for batch in train_loader:
            x = batch[0].to(device)

            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss_function(recon_x, x, mu, logvar, beta)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_recon_loss += recon_loss.item()
            total_train_kl_loss += kl_loss.item()

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        avg_train_recon_loss = total_train_recon_loss / len(train_loader.dataset)
        avg_train_kl_loss = total_train_kl_loss / len(train_loader.dataset)
        
        train_losses.append(avg_train_loss)
        train_recon_losses.append(avg_train_recon_loss)
        train_kl_losses.append(avg_train_kl_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        total_val_recon_loss = 0.0
        total_val_kl_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                recon_x, mu, logvar = model(x)
                loss, recon_loss, kl_loss = vae_loss_function(recon_x, x, mu, logvar, beta)
                
                total_val_loss += loss.item()
                total_val_recon_loss += recon_loss.item()
                total_val_kl_loss += kl_loss.item()

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        avg_val_recon_loss = total_val_recon_loss / len(val_loader.dataset)
        avg_val_kl_loss = total_val_kl_loss / len(val_loader.dataset)
        
        val_losses.append(avg_val_loss)
        val_recon_losses.append(avg_val_recon_loss)
        val_kl_losses.append(avg_val_kl_loss)

        # Logging
        epoch_bar.set_description(f"Epoch {epoch+1}/{num_epochs}")
        epoch_bar.set_postfix(
            train_loss=f"{avg_train_loss:.4f}", 
            val_loss=f"{avg_val_loss:.4f}",
            train_recon=f"{avg_train_recon_loss:.4f}",
            train_kl=f"{avg_train_kl_loss:.4f}"
        )

    # Plot loss curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Total loss
    axes[0, 0].plot(train_losses, label='Train Loss')
    axes[0, 0].plot(val_losses, label='Validation Loss')
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Total Loss")
    axes[0, 0].set_title("VAE Training - Total Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Reconstruction loss
    axes[0, 1].plot(train_recon_losses, label='Train Recon Loss')
    axes[0, 1].plot(val_recon_losses, label='Validation Recon Loss')
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Reconstruction Loss")
    axes[0, 1].set_title("VAE Training - Reconstruction Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # KL divergence loss
    axes[1, 0].plot(train_kl_losses, label='Train KL Loss')
    axes[1, 0].plot(val_kl_losses, label='Validation KL Loss')
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("KL Divergence Loss")
    axes[1, 0].set_title("VAE Training - KL Divergence Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Loss ratio
    loss_ratio = [kl/recon if recon > 0 else 0 for kl, recon in zip(train_kl_losses, train_recon_losses)]
    axes[1, 1].plot(loss_ratio, label='KL/Recon Ratio')
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("KL/Reconstruction Ratio")
    axes[1, 1].set_title("VAE Training - Loss Ratio")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_recon_losses': train_recon_losses,
        'train_kl_losses': train_kl_losses,
        'val_recon_losses': val_recon_losses,
        'val_kl_losses': val_kl_losses
    }