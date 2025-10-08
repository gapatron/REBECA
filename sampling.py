import torch
from tqdm import tqdm

def sample_from_diffusion(
    model,
    user_ids_cond,
    scores_cond,
    user_ids_uncond,
    scores_uncond,
    img_embedding_size,
    scheduler,
    guidance_scale=5.0,
    device="cuda",
):
    """
    Generate samples from a trained diffusion model using the NoiseScheduler.

    model: The trained diffusion model that predicts noise.
    num_samples: Number of samples to generate.
    img_embedding_size: The size of the generated embeddings (e.g., 1280 for CLIP embeddings).
    scheduler: The instance of NoiseScheduler.
    device: The device to run the sampling on (e.g., "cuda" or "cpu").
    """
    num_samples = user_ids_cond.size(0)
    timesteps = scheduler.config.num_train_timesteps

    model.eval()

    with torch.no_grad():
        x_t = torch.randn((num_samples, img_embedding_size), device=device)

        # Reverse sampling loop
        for t in reversed(range(timesteps)):
            t_tensor = torch.full((num_samples,), t, dtype=torch.long, device=device)

            model_output_uncond = model(x_t, t_tensor, user_ids_uncond, scores_uncond)
            model_output_cond = model(x_t, t_tensor, user_ids_cond, scores_cond)
            model_output = model_output_uncond + guidance_scale * (model_output_cond - model_output_uncond)
            x_t = scheduler.step(model_output, t, x_t).prev_sample

    return x_t


def sample_user_images(
        diffusion_prior_model, 
        diffusion_pipe, 
        users,
        images_per_user,
        noise_scheduler,
        guidance_scale,
        prompt, 
        negative_prompt,
        img_embedding_size,
        pipe_cfg=7.5,
        device="cuda"
        ):
    # prompts have to be lists
    #prompt=["Realistic image, finely detailed, with balanced composition and harmonious elements. Natural lighting, dynamic yet subtle tones, versatile style adaptable to diverse themes and aesthetics, prioritizing clarity and authenticity."]*samples_per_user
    #negative_prompt = ["deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality"]
    prompt = prompt 
    negative_prompt = negative_prompt 
    data = dict.fromkeys(users)
    
    # Ensure model is in eval mode
    diffusion_prior_model.eval()
    results = {}
    for user_idx in tqdm(users):
        user_dict = dict.fromkeys(["prior_embeddings", "images", "posterior_embeddings"])
        like = 1
        score_tensor = torch.tensor(like).expand(images_per_user).long().to(device)
        user_tensor = torch.tensor(user_idx).expand(images_per_user).to(device)
        user_uncond   = torch.full_like(user_tensor, fill_value=210)
        score_uncond  = torch.full_like(score_tensor, fill_value=2)

        # Generate embeddings with memory cleanup
        with torch.no_grad():
            sampled_img_embs = sample_from_diffusion(
                    model=diffusion_prior_model,
                    user_ids_cond=user_tensor,
                    scores_cond=score_tensor,
                    user_ids_uncond=user_uncond,
                    scores_uncond=score_uncond,
                    img_embedding_size=img_embedding_size,
                    scheduler=noise_scheduler,
                    guidance_scale=guidance_scale,
                    device="cuda",
                )
        
        # Convert to appropriate dtype and device
        sampled_img_embs = sampled_img_embs.to(diffusion_pipe.device)
        user_dict["prior_embeddings"] = sampled_img_embs
        # Generate images with memory optimization
        gen_images = []
        with torch.inference_mode():
            for i in range(images_per_user):
                pos = sampled_img_embs[i].unsqueeze(0).unsqueeze(0)  # (1,1,D)
                neg = torch.zeros_like(pos)
                ip  = torch.cat([neg, pos], dim=0)                   # (2,1,D) for CFG

                imgs = diffusion_pipe(
                    prompt="",
                    negative_prompt="",
                    guidance_scale=pipe_cfg,
                    ip_adapter_image_embeds=[ip],
                    num_inference_steps=50,
                ).images

                gen_images.append(imgs[0])

                # free ASAP
                del pos, neg, ip, imgs
                torch.cuda.empty_cache()

        posterior_embeddings = []
        with torch.inference_mode():
            for pil_img in gen_images:
                emb, _ = diffusion_pipe.encode_image(pil_img, device=diffusion_pipe.device, num_images_per_prompt=1)
                posterior_embeddings.append(emb[0].float().cpu())  # move off GPU immediately
                torch.cuda.empty_cache()

        results[user_idx] = {
            "prior_embeddings": sampled_img_embs.float().cpu(),   # store on CPU
            "images": gen_images,                                 # PILs live on CPU
            "posterior_embeddings": torch.stack(posterior_embeddings) if posterior_embeddings else None,
        }

        # release per-user tensors
        del sampled_img_embs, gen_images, posterior_embeddings, user_tensor, score_tensor, user_uncond, score_uncond
        torch.cuda.empty_cache()

    return results