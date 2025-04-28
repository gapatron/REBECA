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
    prediction_type="epsilon",
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
        scheduler.config.prediction_type = prediction_type

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
        device="cuda"
        ):
    # prompts have to be lists
    #prompt=["Realistic image, finely detailed, with balanced composition and harmonious elements. Natural lighting, dynamic yet subtle tones, versatile style adaptable to diverse themes and aesthetics, prioritizing clarity and authenticity."]*samples_per_user
    #negative_prompt = ["deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality"]
    prompt = prompt * images_per_user
    negative_prompt = negative_prompt * images_per_user
    print(users)
    data = dict.fromkeys(users)
    for user_idx in tqdm(users):
        user_dict = dict.fromkeys(["prior_embeddings", "images", "posterior_embeddings"])
        like = 1
        score_tensor = torch.tensor(like).expand(images_per_user).long().to(device)
        user_tensor = torch.tensor(user_idx).expand(images_per_user).to(device)
        user_ids_uncond_tensor = torch.full_like(user_tensor, fill_value = 94).to(device)
        score_uncond_tensor = torch.full_like(score_tensor, fill_value = 2).to(device)

        sampled_img_embs = sample_from_diffusion(
                model=diffusion_prior_model,
                user_ids_cond=user_tensor,
                scores_cond=score_tensor,
                user_ids_uncond=user_ids_uncond_tensor,
                scores_uncond=score_uncond_tensor,
                img_embedding_size=img_embedding_size,
                scheduler=noise_scheduler,
                guidance_scale=guidance_scale,
                prediction_type="epsilon",
                device="cuda",
            )
        sampled_img_embs = sampled_img_embs.to(diffusion_pipe.device)  # Ensure same device
        sampled_img_embs = sampled_img_embs.half() if diffusion_pipe.unet.dtype == torch.float16 else sampled_img_embs.float()
        user_dict["prior_embeddings"] = sampled_img_embs
        sampled_img_embs = sampled_img_embs.to(diffusion_pipe.device)
        sampled_img_embs = sampled_img_embs.half() if diffusion_pipe.unet.dtype == torch.float16 else sampled_img_embs.float()
        with torch.no_grad():
            gen_images = diffusion_pipe(
                prompt=prompt,
                ip_adapter_image_embeds=[sampled_img_embs.unsqueeze(1)],
                negative_prompt=negative_prompt,
                num_inference_steps=100,
                ).images
            torch.cuda.empty_cache()

            #posterior_embeddings = []
            #for image in gen_images:
            #        image_emb = diffusion_pipe.encode_image(image, device="cuda", num_images_per_prompt=1)[0].squeeze()
            #        posterior_embeddings.append(image_emb.cpu())
            
            #user_dict["posterior_embeddings"] = torch.stack(posterior_embeddings)
        data[user_idx] = user_dict
        user_dict["images"] = gen_images
    data[user_idx] = user_dict
    return data