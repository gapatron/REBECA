from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import save_file, load_file
import os
import json
import torch

def main():
    output_dir = "./data/flickr/evaluation/lora/weights/sd15_lora_r512_unet"
    device = "cuda"
    global_step = 12000
    pipeline = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        revision=None,
    ).to(device)
    pipeline.safety_checker = None

    # Load our tiny config
    with open(os.path.join(output_dir, f"{global_step}_peft_config.json"), "r") as f:
        cfg = json.load(f)

    # UNet adapter
    unet_cfg = LoraConfig(**cfg["unet"])
    pipeline.unet = get_peft_model(pipeline.unet, unet_cfg)
    unet_sd = load_file(os.path.join(output_dir, f"{global_step}_unet_peft.safetensors"))
    set_peft_model_state_dict(pipeline.unet, unet_sd)

    # Text-encoder adapter (optional)
    if cfg["text_encoder"] is not None:
        te_cfg = LoraConfig(**cfg["text_encoder"])
        pipeline.text_encoder = get_peft_model(pipeline.text_encoder, te_cfg)
        te_sd = load_file(os.path.join(output_dir, f"{global_step}_te_peft.safetensors"))
        set_peft_model_state_dict(pipeline.text_encoder, te_sd)

    pipe_cfg = 5.0
    images_per_user = 25
    seed = 42
    torch.manual_seed(seed)

    savedir = "./data/flickr/evaluation/lora/samples"
    savepath = f"{savedir}/pipecfg_{pipe_cfg}_seed_{seed}"
    users = list(range(210))


    
    for user_id in users:
        gen_images = []
        with torch.inference_mode():
            for i in range(images_per_user):

                imgs = pipeline(
                        prompt=f"<u{user_id}> Realistic image, finely detailed, with balanced composition and harmonious elements.",
                        negative_prompt="deformed, ugly, wrong proportion, frame, watermark, low res, bad anatomy, worst quality, low quality",
                        guidance_scale=pipe_cfg,
                        num_inference_steps=50,
                    ).images

                gen_images.append(imgs[0])
        torch.save(gen_images, f"{savepath}_user_{user_id}.imgs")
if __name__=="__main__":
    main()