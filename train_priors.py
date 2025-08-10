import torch
import time
import math


########################################## DIFFUSION #######################################


def validate_diffusion_model(model, noise_scheduler, val_dataloader, device, objective="noise-pred"):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            user_ids = batch["user_id"].to(device).to(torch.long)
            ratings = batch["rating"].to(device).to(torch.long)
            likes = batch["like"].to(device).to(torch.long)
            image_embs = batch["image_emb"].to(device)
            batch_size = user_ids.shape[0]
            
            # Generate noise and timesteps
            noise = torch.randn((batch_size, image_embs.shape[1]), device=device)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device
            ).long()
            
            # Add noise to image embeddings
            noisy = noise_scheduler.add_noise(image_embs, noise, timesteps)
            
            # Forward pass
            pred = model(noisy, timesteps, user_ids, likes)
            
            # Compute loss based on objective
            if objective == "noise-pred":
                loss = torch.nn.functional.mse_loss(pred, noise)
            elif objective == "z0-pred":
                loss = torch.nn.functional.mse_loss(pred, image_embs)
            elif objective == "v_prediction":
                alpha_bar = noise_scheduler.alphas_cumprod.to(timesteps.device)
                alpha_bar_t = alpha_bar[timesteps] 
                v = alpha_bar_t.sqrt().view(-1,1) * noise - (1 - alpha_bar_t).sqrt().view(-1,1) * image_embs
                loss = torch.nn.functional.mse_loss(v, pred)
            else:
                raise ValueError("Invalid objective. Choose 'noise-pred', 'z0-pred', or 'v_prediction'.")
            
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(val_dataloader)
    return avg_val_loss



def train_diffusion_prior(
    model,
    noise_scheduler,
    train_dataloader,
    val_dataloader,
    optimizer,
    scheduler,
    cfg_drop_prob=0.1,
    num_unique_users=94,
    objective="noise-pred",
    device="cuda",
    num_epochs=100,
    patience=20,
    savepath="best_model.pt",
    return_losses=False,
    verbose=True
):
    best_val_loss = float('inf')
    no_improve = 0
    
    all_grad_norms = []
    all_train_losses = []
    all_val_losses = []
    
    global_step = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_grad_norm = 0
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad(set_to_none=True)
            
            optimizer.zero_grad(set_to_none=True)
            # Extract batch data
            user_ids = batch["user_id"].to(device).to(torch.long)
            likes = batch["like"].to(device).to(torch.long)
            image_embs = batch["image_emb"].to(device)
            batch_size = user_ids.shape[0]
            
            # Generate noise and timesteps
            noise = torch.randn((batch_size, image_embs.shape[1]), device=device)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device
            ).long()
            
            # Add noise to image embeddings
            noisy = noise_scheduler.add_noise(image_embs, noise, timesteps)

            ## CFG!
            do_drop = (torch.rand(batch_size, device=device) < cfg_drop_prob)
            user_id_uncond = torch.full_like(user_ids, num_unique_users)
            score_uncond   = torch.full_like(likes, 2)
            final_user_ids  = torch.where(do_drop, user_id_uncond, user_ids)
            final_likes    = torch.where(do_drop, score_uncond, likes)
            
            # Forward pass
            pred = model(noisy, timesteps, final_user_ids, final_likes)
            
            if objective == "noise-pred":
                loss = torch.nn.functional.mse_loss(pred, noise)
            elif objective == "z0-pred":
                loss = torch.nn.functional.mse_loss(pred, image_embs)
            elif objective == "v_prediction":
                alpha_bar = noise_scheduler.alphas_cumprod.to(timesteps.device)
                alpha_bar_t = alpha_bar[timesteps] 
                v = alpha_bar_t.sqrt().view(-1,1) * noise - (1 - alpha_bar_t).sqrt().view(-1,1) * image_embs
                loss = torch.nn.functional.mse_loss(v, pred)
            else:
                raise ValueError("Invalid objective. Choose 'noise-pred', 'z0-pred', or 'v_prediction'.")
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            # Compute gradient norms for all parameters
            total_loss += loss.item()
            total_grad_norm += sum(
                p.grad.norm(2).item() for p in model.parameters() if p.grad is not None
            )
                
            global_step += 1
        
        # Average metrics for the epoch
        avg_train_loss = total_loss / len(train_dataloader)
        avg_grad_norm = total_grad_norm / len(train_dataloader)
        
        all_train_losses.append(avg_train_loss)
        all_grad_norms.append(avg_grad_norm)
        
        # Validation step
        avg_val_loss = validate_diffusion_model(
            model, noise_scheduler, val_dataloader, device, objective
        )
        all_val_losses.append(avg_val_loss)
        
        # Scheduler step
        scheduler.step(avg_val_loss)

        # Print epoch results
        elapsed_time = time.time() - start_time
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}, Time Elapsed: {elapsed_time:.2f}s, Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Grad Norm: {avg_grad_norm:.4f}")

        

        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            train_loss_at_best_val_loss = avg_train_loss
            torch.save(model.state_dict(), savepath)
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve == patience:
            print(f"Early stopping with best val loss: {best_val_loss}!")
            break
    
    # Load best model
    model.load_state_dict(torch.load(savepath, weights_only=True))
    if return_losses:
        return(train_loss_at_best_val_loss, best_val_loss)
    return model


if __name__ == "__main__":
    import pandas as pd
    from utils import (
        map_embeddings_to_ratings,
        save_rebeca_config,
        set_seeds
    )
    from Datasets import EmbeddingsDataset, RecommenderUserSampler
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from torch.utils.data import DataLoader
    import os
    # 1. Load raw embeddings and ratings
    image_features = torch.load("./data/flickr/processed/ip-adapters/SD15/sd15_image_embeddings.pt", weights_only=True)
    ratings_df = pd.read_csv("./data/flickr/processed/ratings.csv")

    # 2. Filter users by threshold (before splitting)
    usr_threshold = 100
    liked_counts = (
        ratings_df[ratings_df["score"] >= 4]
        .groupby("worker_id")["score"]
        .count()
        .reset_index()
    )
    liked_counts.columns = ["worker_id", "liked_count"]
    valid_users = liked_counts[liked_counts["liked_count"] >= usr_threshold]["worker_id"].tolist()
    filtered_ratings_df = ratings_df[ratings_df["worker_id"].isin(valid_users)].copy()
    print(f"User loss: {210-len(valid_users)}")
    print(f"Data loss: {100*(1 - filtered_ratings_df.shape[0]/ratings_df.shape[0])}%")

    # 3. Remap worker IDs to contiguous indices
    worker_mapping = {old_id: new_id for new_id, old_id in enumerate(valid_users)}
    filtered_ratings_df = filtered_ratings_df.rename(columns={"worker_id": "old_worker_id"})
    filtered_ratings_df["worker_id"] = filtered_ratings_df["old_worker_id"].map(lambda x: worker_mapping.get(x, -1))
    filtered_ratings_df = filtered_ratings_df.reset_index(drop=True)
    worker_mapping_df = pd.DataFrame(list(worker_mapping.items()), columns=["old_worker_id", "worker_id"])
    worker_mapping_df.to_csv(f"./data/flickr/processed/worker_id_mapping_usrthr_{usr_threshold}.csv", index=False)
    filtered_ratings_df.to_csv(f"./data/flickr/processed/filtered_ratings_df_usrthrs_{usr_threshold}.csv", index=False)

    # 4. Split data (train/val/test) BEFORE normalization
    from utils import split_recommender_data
    train_df, val_df, test_df = split_recommender_data(
        ratings_df=filtered_ratings_df,
        val_spu=10,
        test_spu=10,
        seed=42
    )
    d = image_features.shape[-1]
    # L2 normalization with sqrt(d) scaling for diffusion stability
    norms = image_features.norm(dim=-1, keepdim=True)
    # Avoid division by zero
    norms = torch.clamp(norms, min=1e-8)
    image_features_normed = image_features / norms * math.sqrt(d)
    emb_final  = torch.clamp(image_features_normed, -5, 5) / 5   

    # 7. Map normalized embeddings to ratings (expanded_features)
    expanded_features = map_embeddings_to_ratings(emb_final, ratings_df)

    # 8. Build datasets and dataloaders
    set_seeds(0)
    batch_size = 64
    samples_per_user = 500
    unique_users = filtered_ratings_df["worker_id"].unique()
    train_dataset = EmbeddingsDataset(
        train_df,
        image_embeddings=expanded_features[train_df.original_index]
    )
    val_dataset = EmbeddingsDataset(
        val_df,
        image_embeddings=expanded_features[val_df.original_index]
    )
    train_user_sampler = RecommenderUserSampler(train_df, num_users=len(unique_users), samples_per_user=samples_per_user)
    train_dataloader = DataLoader(train_dataset, sampler=train_user_sampler, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # 9. Define model (replace with your model if needed)
    from prior_models import CrossAttentionDiffusionPrior
    model = CrossAttentionDiffusionPrior(
        img_embed_dim=1024,
        num_users=len(unique_users),
        num_tokens=8,
        n_heads=8,
        num_layers=6,
        dim_feedforward=2048
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # 10. Optimizer, scheduler, etc.
    diffusion_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="laplace")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(diffusion_optimizer, 'min', patience=5, factor=0.5)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    # 11. Train
    from train_priors import train_diffusion_prior
    savepath = f"./data/flickr/evaluation/diffusion_priors/models/weights/test_xattn_v5.pth"
    train_loss, val_loss = train_diffusion_prior(
        model=model,
        noise_scheduler=noise_scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=diffusion_optimizer,
        scheduler=scheduler,
        num_unique_users=len(unique_users),
        objective="v_prediction",  # or "noise-pred" or "z0-pred"
        device=str(device),
        num_epochs=2001,
        patience=20,
        savepath=savepath,
        return_losses=True,
        verbose=True
    )

    # 12. (Optional) Save REBECA config for downstream use
    save_rebeca_config(
        model_class="CrossAttentionDiffusionPrior",
        model_kwargs={
            "img_embed_dim": 1024,
            "num_users": len(unique_users),
            "num_tokens": 8,
            "n_heads": 8,
            "num_layers": 6,
            "dim_feedforward": 2048
        },
        weights_file=os.path.basename(savepath),
        output_dir=os.path.dirname(savepath),
        num_train_timesteps=1000,
        num_users=len(unique_users),
        img_embed_dim=1024
    )