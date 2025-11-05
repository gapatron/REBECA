import os, json, glob, torch
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file

def _find_latest_user_lora(user_dir):
    """
    Find latest {step}_peft_config.json in a user's dir and return (step, cfg_path).
    Returns (None, None) if nothing found.
    """
    cfgs = glob.glob(os.path.join(user_dir, "*_peft_config.json"))
    if not cfgs:
        return None, None
    def _step(p):
        # expects ".../<STEP>_peft_config.json"
        base = os.path.basename(p)
        try:
            return int(base.split("_")[0])
        except Exception:
            return -1
    cfgs.sort(key=_step)
    step = _step(cfgs[-1])
    return step, cfgs[-1]

def _apply_lora_to_pipeline(pipeline, cfg_path, user_dir, step):
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    # UNet adapter
    unet_cfg = LoraConfig(**cfg["unet"])
    pipeline.unet = get_peft_model(pipeline.unet, unet_cfg)
    unet_sd = load_file(os.path.join(user_dir, f"{step}_unet_peft.safetensors"))
    set_peft_model_state_dict(pipeline.unet, unet_sd)

    # Text-encoder adapter (optional)
    if cfg.get("text_encoder") is not None:
        te_cfg = LoraConfig(**cfg["text_encoder"])
        pipeline.text_encoder = get_peft_model(pipeline.text_encoder, te_cfg)
        te_sd = load_file(os.path.join(user_dir, f"{step}_te_peft.safetensors"))
        set_peft_model_state_dict(pipeline.text_encoder, te_sd)

def main(args):
    if args.model == "shared":
        output_dir = "./data/flickr/evaluation/lora/weights/shared/"
        device = args.device
        global_step = 12000

        pipeline = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            revision=None,
        ).to(device)
        pipeline.safety_checker = None

        with open(os.path.join(output_dir, f"{global_step}_peft_config.json"), "r") as f:
            cfg = json.load(f)

        unet_cfg = LoraConfig(**cfg["unet"])
        pipeline.unet = get_peft_model(pipeline.unet, unet_cfg)
        unet_sd = load_file(os.path.join(output_dir, f"{global_step}_unet_peft.safetensors"))
        set_peft_model_state_dict(pipeline.unet, unet_sd)

        if cfg["text_encoder"] is not None:
            te_cfg = LoraConfig(**cfg["text_encoder"])
            pipeline.text_encoder = get_peft_model(pipeline.text_encoder, te_cfg)
            te_sd = load_file(os.path.join(output_dir, f"{global_step}_te_peft.safetensors"))
            set_peft_model_state_dict(pipeline.text_encoder, te_sd)

        pipe_cfg = 5.0
        images_per_user = 25
        seed = 42
        torch.manual_seed(seed)

        savedir = "./data/flickr/evaluation/lora/samples_T1/shared"
        savepath = f"{savedir}/pipecfg_{pipe_cfg}_seed_{seed}"
        users = list(range(210))

        os.makedirs(savedir, exist_ok=True)

        for user_id in users:
            gen_images = []
            with torch.inference_mode():
                for _ in range(images_per_user):
                    imgs = pipeline(
                        prompt=f"<u{user_id}> Realistic image, finely detailed.",
                        negative_prompt="deformed, ugly, wrong proportion, frame, watermark, low res",
                        guidance_scale=pipe_cfg,
                        num_inference_steps=50,
                    ).images
                    gen_images.append(imgs[0])
            torch.save(gen_images, f"{savepath}_user_{user_id}.imgs")

    elif args.model == "per_user":
        device = args.device
        base_model = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        per_user_root = "./data/flickr/evaluation/lora/weights/per_user"  
        pipe_cfg = 5.0
        images_per_user = 25
        seed = 42
        torch.manual_seed(seed)

        savedir = "./data/flickr/evaluation/lora/samples_T1/per_user"
        os.makedirs(savedir, exist_ok=True)
        users = list(range(210))
        for user_id in users:
            user_dir = os.path.join(per_user_root, f"u{user_id}")
            step, cfg_path = _find_latest_user_lora(user_dir)
            if step is None:
                print(f"[skip] u{user_id}: no LoRA files found in {user_dir}")
                continue

            # fresh base pipeline per user (simple & avoids adapter stacking)
            pipe = StableDiffusionPipeline.from_pretrained(base_model, revision=None).to(device)
            pipe.safety_checker = None

            # attach this user's LoRA
            _apply_lora_to_pipeline(pipe, cfg_path, user_dir, step)

            gen_images = []
            with torch.inference_mode():
                for _ in range(images_per_user):
                    imgs = pipe(
                        prompt=f"<u{user_id}> Realistic image, finely detailed.",
                        negative_prompt="deformed, ugly, wrong proportion, frame, watermark, low res",
                        guidance_scale=pipe_cfg,
                        num_inference_steps=50,
                    ).images
                    gen_images.append(imgs[0])

            savepath = f"{savedir}/pipecfg_{pipe_cfg}_seed_{seed}_user_{user_id}.imgs"
            torch.save(gen_images, savepath)
            del pipe
            torch.cuda.empty_cache()

if __name__=="__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="shared", choices=["shared", "per_user"], help="Type: shared or per_user")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)
