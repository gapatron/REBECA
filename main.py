import torch
from utils import set_seeds
from prior_models import TransformerEmbeddingDiffusionModelv2
from diffusers import DDPMScheduler
from sampling import sample_user_images
#from utils import save_generated_data_to_user_dir
from diffusers import StableDiffusionPipeline
#import random
import os
import glob
import torch
import argparse
import random
import pandas as pd

def save_generated_data_to_user_dir(data, base_dir):
    """
    Saves new images with an offset so we don't overwrite old images.
    Each user gets a folder user_{id}/images and user_{id}/embeddings.
    """
    for user_id, user_dict in data.items():
        user_dir = os.path.join(base_dir, f"user_{user_id}")
        images_dir = os.path.join(user_dir, "images")
        emb_dir = os.path.join(user_dir, "embeddings")

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(emb_dir, exist_ok=True)

        # Count how many *.png files are already in images_dir.
        offset = len(glob.glob(os.path.join(images_dir, "*.png")))

        # Save images (avoiding overwrites via offset).
        for i, image in enumerate(user_dict["images"]):
            image_path = os.path.join(images_dir, f"img_{offset + i}.png")
            image.save(image_path)

        # Save embeddings
        torch.save(user_dict["prior_embeddings"], os.path.join(emb_dir, "prior_embeddings.pth"))
        if "posterior_embeddings" in user_dict and user_dict["posterior_embeddings"] is not None:
            torch.save(user_dict["posterior_embeddings"], os.path.join(emb_dir, "embeddings.pth"))

def main():
    parser = argparse.ArgumentParser(description='Generate baseline images')
    parser.add_argument('-prompt_level', type=int, help='Prompt level')

    args = parser.parse_args()

    prompt_level  = args.prompt_level


    set_seeds(2024)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    pipe.safety_checker = None  # disable safety checker
    pipe.vae.eval()            
    pipe.unet.eval()

    diffusion_prior_model = TransformerEmbeddingDiffusionModelv2(
    img_embed_dim=1024,
    num_users=94,  
    n_heads=16,
    num_tokens=1,
    num_user_tokens=4,
    num_layers=8,
    dim_feedforward=2048,
    whether_use_user_embeddings=True
    ).to(device)

    # Load the weights
    savepath = ("data/flickr/evaluation/diffusion_priors/models/weights/sd15_nl8_heads16_dim_feedforward2048_lr0.0001_it1_ut4_adamw_reduce_on_plateau_bs64_nslinear_spu80_timesteps6000_objnoise-pred_useueTrue.pth")

    diffusion_prior_model.load_state_dict(
        torch.load(savepath, map_location=device, weights_only=True),
        strict=False
    )
    diffusion_prior_model.eval()
    noise_scheduler = DDPMScheduler(num_train_timesteps=6000)
    all_users = list(range(95))
    # for example, pick 50 random users; or use all_users if you prefer
    #users = random.sample(all_users, 10)
    users = all_users
    # 7. We want SPU=80, GS=7 only, with this single prompt
    # used spu 50 for the new usrthrs_100
    spu = 80
    gs = 10.0
    if prompt_level == 0:
        
        prompt = [""]
        neg_prompt = [""]

    if prompt_level == 1:   

        prompt = ["high quality photo"]
        neg_prompt =  ["bad quality photo, letters"]

    if prompt_level == 2:
        prompt = [
            "Realistic image, finely detailed, with balanced composition and harmonious elements. "
            "Dynamic yet subtle tones, versatile style adaptable to diverse themes and aesthetics, "
            "prioritizing clarity and authenticity."
        ]
        neg_prompt = [
            "deformed, ugly, wrong proportion, frame, watermark, low res, bad anatomy, worst quality, low quality"
        ]

    #  We’ll generate 50 images per user 
    batches = [10, 10, 10, 10, 10]

    print(prompt)
    safe_prompt = prompt[0].replace(" ", "_")[:30] if prompt else "empty"
    safe_neg = neg_prompt[0].replace(" ", "_")[:30] if neg_prompt else "empty"
    save_dir = (
        f"./data/flickr/evaluation/diffusion_priors/models/samples/final/"
        f"SPU_{spu}_GS_{gs}_{safe_prompt}_NEG_{safe_neg}_usrthres_100_v1"
    )

    print(f"\nGenerating 50 images per user for SPU={spu}, GS={gs}")
    print(f"  Prompt: {prompt}")
    print(f"  Neg:    {neg_prompt}")
    from tqdm import tqdm
    for batch_size in tqdm(batches):
        print(f"  Sub-batch of {batch_size} images per user...")
        gen_data = sample_user_images(
            diffusion_prior_model=diffusion_prior_model,
            diffusion_pipe=pipe,
            users=users,
            images_per_user=batch_size,
            noise_scheduler=noise_scheduler,
            guidance_scale=gs,
            prompt=prompt,
            negative_prompt=neg_prompt,
            img_embedding_size=1024,  # for SD1.5
            device=device,
        )
        torch.cuda.empty_cache()

        # Save them in the same directory; naming in 'save_generated_data_to_user_dir'
        # should keep them distinct if it’s time-stamped or user-labeled properly.
        save_generated_data_to_user_dir(data=gen_data, base_dir=save_dir)
        torch.cuda.empty_cache()

    print("Done! Each user should now have a total of 50 images.")


if __name__ == "__main__":
    main()
