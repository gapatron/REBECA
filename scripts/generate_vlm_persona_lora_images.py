import os, json, torch, random, math
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from diffusers import StableDiffusionPipeline
from generate_vlm_persona_images import load_profiles_jsonl, choose_pos_keywords, trim_persona, build_prompts_for_user, seed_for
from generate_lora_images import _find_latest_user_lora, _apply_lora_to_pipeline
# ----------------------- Config -----------------------
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
OUTPUT_DIR = "./data/flickr/evaluation/vlm_personas_plus_lora"

PROFILES_JSONL = f"./data/flickr/evaluation/vlm_personas/weights/personas_llava.jsonl"
LORA_WEIGHTS =  "./data/flickr/evaluation/lora/weights/per_user"
SAVE_DIR = f"{OUTPUT_DIR}/samples"

GUIDANCE = 5.0
IMAGES_PER_USER = 25
NUM_STEPS = 50
BASE_SEED = 0
USERS = list(range(210))
DEVICE = "cuda"


FALLBACK_POS = ["nature", "landscape", "portrait", "cityscape", "animals"]
FALLBACK_STYLE = "high quality, detailed, natural lighting"
FALLBACK_NEG = ["low quality", "blurry", "deformed", "overexposed", "underexposed"]
K_POS_PER_PROMPT = (1, 2) 


def main():
    profiles = load_profiles_jsonl(PROFILES_JSONL)
    if not profiles:
        raise RuntimeError(f"No profiles loaded from {PROFILES_JSONL}")

    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    save_prefix = f"{SAVE_DIR}/pipecfg_{GUIDANCE}_seed_{BASE_SEED}"

    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID).to(DEVICE)
    pipe.safety_checker = None


    # Generation loop
    for user_id in USERS:

        # Apply LoRA to pipeline
        user_dir = os.path.join(LORA_WEIGHTS, f"u{user_id}")
        step, cfg_path = _find_latest_user_lora(user_dir)
        if step is None:
                print(f"[skip] u{user_id}: no LoRA files found in {user_dir}")
                continue
        pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, revision=None).to(DEVICE)
        pipe.safety_checker = None
        _apply_lora_to_pipeline(pipe, cfg_path, user_dir, step)

        # Get VLM profiles
        prof = profiles.get(user_id, {"persona":"", "pos":[], "neg":[]})

        gen_images = []

        for i in range(IMAGES_PER_USER):
            prompt, negative_prompt = build_prompts_for_user(prof, user_id, i)
            g = torch.Generator(device=DEVICE).manual_seed(seed_for(user_id, i))
            lora_vlm_prompt = f"<u{user_id}> {prompt}"
            with torch.inference_mode():
                out = pipe(
                    prompt=lora_vlm_prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=GUIDANCE,
                    num_inference_steps=NUM_STEPS,
                    generator=g,
                )
            img = out.images[0]
            gen_images.append(img)

        out_path = f"{save_prefix}_user_{user_id}.imgs"
        torch.save(gen_images, out_path)


    # Clean up
    del pipe
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
