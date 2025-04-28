import torch
from utils import set_seeds
from prior_models import TransformerEmbeddingDiffusionModelv2
from diffusers import DDPMScheduler
from sampling import  sample_user_images
from utils import save_generated_data_to_user_dir
from diffusion_adapters import StableDiffusionPipelineAdapterEmbeddings
import random

def main():
    set_seeds(2024)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = "runwayml/stable-diffusion-v1-5"                                                                                                                                                                                                                
    pipe = StableDiffusionPipelineAdapterEmbeddings.from_pretrained(model_id).to("cuda")
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")     
    pipe.safety_checker = None # Prudes
    device = "cuda"


    diffusion_prior_model = TransformerEmbeddingDiffusionModelv2(
    img_embed_dim=1024,
    num_users=94,    
    n_heads=16,
    num_tokens=1,
    num_user_tokens=4,
    num_layers=8,
    dim_feedforward=2048
    ).to(device)


    noise_scheduler = DDPMScheduler(num_train_timesteps=6000)

    spus = [80]    
    guidance_scales = [8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    prompt_candidates = [
        ([""], [""]), 
       # (
       #     ["high quality photo"], 
       #     ["bad quality photo, letters"]
        #),
        #(
        #    ["Realistic image, finely detailed, with balanced composition and harmonious elements. "
        #     "Dynamic yet subtle tones, versatile style adaptable to diverse themes and aesthetics, "
        #     "prioritizing clarity and authenticity."],
        #    ["deformed, ugly, wrong proportion, frame, watermark, low res, bad anatomy, worst quality, low quality"]
        #)
    ]

    users = random.sample(list(range(94)), 30)
    images_per_user = 12

    for spu in spus:
        savepath = f"./data/flickr/evaluation/diffusion_priors/models/weights/sd15_nl8_heads16_dim_feedforward2048_lr0.0001_it1_ut4_adamw_reduce_on_plateau_bs64_nslinear_spu{spu}_timesteps6000_objnoise-pred_useueTrue.pth"
        diffusion_prior_model.load_state_dict(torch.load(savepath, map_location="cuda"), strict=False)
        diffusion_prior_model.eval()
        for gs in guidance_scales:
            for (prompt, neg_prompt) in prompt_candidates:

                print(f"\nRunning for SPU={spu}, Guidance={gs}")
                print(f"  Prompt: {prompt}")
                print(f"  Neg:    {neg_prompt}")

                gen_data = sample_user_images(
                    diffusion_prior_model=diffusion_prior_model,
                    diffusion_pipe=pipe,
                    users=users,
                    images_per_user=images_per_user,
                    noise_scheduler=noise_scheduler,
                    guidance_scale=gs,
                    prompt=prompt,
                    negative_prompt=neg_prompt,
                    img_embedding_size=1024,  # for SD1.5
                    device="cuda",
                )
                torch.cuda.empty_cache()
                safe_prompt = prompt[0].replace(" ", "_")[:30] if prompt else "empty"
                safe_neg = neg_prompt[0].replace(" ", "_")[:30] if neg_prompt else "empty"
                save_dir = f"./data/flickr/evaluation/diffusion_priors/SPU_{spu}_GS_{gs}_{safe_prompt}_NEG_{safe_neg}_usrthrs_100_exp2"
                
                save_generated_data_to_user_dir(
                    data=gen_data,
                    dir=save_dir,
                )
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()