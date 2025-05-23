import torch
import itertools
from tqdm import tqdm
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from train_priors import train_diffusion_prior
from diffusers import DDPMScheduler
from prior_models import TransformerEmbeddingDiffusionModel, TransformerEmbeddingDiffusionModelv2
from torch.utils.data import DataLoader
from Datasets import RecommenderUserSampler
import pandas as pd
from time import time


def create_model(
        layers, 
        heads, 
        num_image_tokens=2, 
        num_user_tokens=1, 
        img_embed_dim=1024, 
        num_users=94,
        use_ue=True,
        dim_feedforward=1024
        ):
    return TransformerEmbeddingDiffusionModelv2(
        img_embed_dim=img_embed_dim,
        num_users=num_users,
        num_tokens=num_image_tokens,
        num_user_tokens=num_user_tokens,
        n_heads=heads,
        num_layers=layers,
        whether_use_user_embeddings=use_ue,
        dim_feedforward=dim_feedforward,
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def get_optimizer(optimizer_name, model_params, learning_rate):
    if optimizer_name == 'adamw':
        return AdamW(model_params, lr=learning_rate, weight_decay=1e-5)
    elif optimizer_name == 'sgd':
        return SGD(model_params, lr=learning_rate, momentum=0.9)

def get_scheduler(scheduler_name, optimizer):
    if scheduler_name == 'reduce_on_plateau':
        return ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    elif scheduler_name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

def get_noise_scheduler(schedule_name, timesteps, clipsample, rescale_betas):
    if schedule_name == 'linear':
        return DDPMScheduler(num_train_timesteps=timesteps, beta_schedule="linear", clip_sample=clipsample, rescale_betas_zero_snr=rescale_betas)
    elif schedule_name == "squaredcos_cap_v2":
        return DDPMScheduler(num_train_timesteps=timesteps, beta_schedule="squaredcos_cap_v2", clip_sample=clipsample, rescale_betas_zero_snr=rescale_betas)
    elif schedule_name == 'quadratic':
        return DDPMScheduler(num_train_timesteps=timesteps, beta_schedule="quadratic", clip_sample=clipsample, rescale_betas_zero_snr=rescale_betas)
    elif schedule_name == "laplace":
        return DDPMScheduler(num_train_timesteps=timesteps, beta_schedule="laplace", clip_sample=clipsample, rescale_betas_zero_snr=rescale_betas)
def save_results_to_csv(results, path):
    df = pd.DataFrame(results)
    savepath = f"{path}/results.csv"
    df.to_csv(savepath, index=False)

# Experimentation loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_grid_search(train_df, train_dataset, val_dataset, param_grid, savedir, unique_users):
    combinations = list(itertools.product(*param_grid.values()))
    results = []
    for params in tqdm(combinations, desc="Hyperparameter combinations"):
        (timesteps, layers, heads, dim_feedforward, num_image_tokens, num_user_tokens, learning_rate, optimizer_name, scheduler_name, batch_size, noise_schedule, samples_per_user, clip_sample, rescale_betas, objective, use_ue, img_embed_dim) = params

        print(f"Running configuration: timesteps={timesteps}, layers={layers}, heads={heads}, image_tokens={num_image_tokens}, user_tokens={num_user_tokens}, learning_rate={learning_rate}, clip_sample={clip_sample}, rescale_betas={rescale_betas}, optimizer={optimizer_name}, scheduler={scheduler_name}, batch_size={batch_size}, noise_schedule={noise_schedule}, samples_per_user={samples_per_user}, objective={objective}, use_ue={use_ue}")

        # Create model, optimizer, scheduler, and noise scheduler
        model = create_model(layers=layers, heads=heads, num_users=unique_users, num_image_tokens=num_image_tokens, num_user_tokens=num_user_tokens, use_ue=use_ue, dim_feedforward=dim_feedforward, img_embed_dim=img_embed_dim)
        optimizer = get_optimizer(optimizer_name, model.parameters(), learning_rate)
        scheduler = get_scheduler(scheduler_name, optimizer)
        noise_scheduler = get_noise_scheduler(noise_schedule, timesteps, clip_sample, rescale_betas)
        # Define save path for the weights
        
        savepath = f"{savedir}/sd15_bs{batch_size}_ns{noise_schedule}_spu{samples_per_user}_timesteps{timesteps}_clipsample{clip_sample}_rescalebetas{rescale_betas}.pth"

        # Adjust dataloader batch size
        train_user_sampler = RecommenderUserSampler(train_df.reset_index(), num_users=unique_users, samples_per_user=samples_per_user)

        train_dataloader = DataLoader(train_dataset, sampler=train_user_sampler, batch_size=batch_size)
        test_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        
        # Train the model
        start = time()
        train_loss, val_loss = train_diffusion_prior(
                model=model,
                noise_scheduler=noise_scheduler,
                train_dataloader=train_dataloader,
                val_dataloader=test_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                objective=objective,
                device=device,
                num_epochs=2001,      # Ensure config.num_epochs is defined
                patience=50,
                savepath=savepath,
                return_losses=True,
                verbose=False
            )
        end = time()
        # Evaluate and save results
        validation_loss = val_loss
        results.append({
            'timesteps': timesteps,
            'layers': layers,
            'heads': heads,
            'image_tokens': num_image_tokens,
            'user_tokens': num_user_tokens,
            'dim_feedforward': dim_feedforward,
            'learning_rate': learning_rate,
            'optimizer': optimizer_name,
            'scheduler': scheduler_name,
            'batch_size': batch_size,
            'noise_schedule': noise_schedule,
            'samples_per_user': samples_per_user,
            'objective':objective,
            'clip_sample':clip_sample,
            'rescale_betas':rescale_betas,
            'train_loss': train_loss,
            'validation_loss': validation_loss,
            'use_ue': use_ue,
            'time': end - start
        })
        #print(f"Final validation loss for parameter configuration is {val_loss}")
        # Save results periodically
        save_results_to_csv(results, path=savedir)

    print(f"Experimentation complete. Results saved to results.csv at {savedir}")
