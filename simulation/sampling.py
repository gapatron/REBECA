import torch

def sample_latents_from_prior(
    model,
    noise_scheduler,
    user_id,
    score,
    latent_dim=16,
    num_samples=8,
    num_timesteps=1000,
    device="cuda"
):
    model.eval()
    x_t = torch.randn((num_samples, latent_dim), device=device)
    user_id_tensor = torch.full((num_samples,), user_id, device=device, dtype=torch.long)
    score_tensor = torch.full((num_samples,), score, device=device, dtype=torch.long)
    for t in reversed(range(num_timesteps)):
        t_scalar = torch.tensor([t], device=device, dtype=torch.long)
        t_tensor = t_scalar.expand(num_samples)
        noise_pred = model(x_t, t_tensor, user_id_tensor, score_tensor)
        step_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=t,
            sample=x_t
        )
        x_t = step_output.prev_sample
    return x_t  