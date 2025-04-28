import torch
import time



########################################## DIFFUSION #######################################


def validate_diffusion_model(model, noise_scheduler, val_dataloader, device):
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
            noise_pred = model(noisy, timesteps, user_ids, likes)
            
            # Compute loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
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
            
            # Extract batch data
            user_ids = batch["user_id"].to(device).to(torch.long)
            likes = batch["like"].to(device).to(torch.long)
            image_embs = batch["image_emb"].to(device)
            batch_size = user_ids.shape[0]
            
            # Generate noise and timesteps
            noise = torch.randn((batch_size, image_embs.shape[1]), device=device)
            timesteps = torch.randint(
                0, len(noise_scheduler.timesteps), (batch_size,), device=device
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
            else:
                raise ValueError("Invalid objective. Choose 'noise-pred' or 'z0-pred'.")
            
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
            model, noise_scheduler, val_dataloader, device,
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
