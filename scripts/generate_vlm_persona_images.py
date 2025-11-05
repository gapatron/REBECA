import os, json, torch, random, math
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from diffusers import StableDiffusionPipeline


# ----------------------- Config -----------------------
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
OUTPUT_DIR = "./data/flickr/evaluation/vlm_personas"
PROFILES_JSONL = f"{OUTPUT_DIR}/weights/personas_llava.jsonl"

SAVE_DIR = f"{OUTPUT_DIR}/samples"
GUIDANCE = 5.0
IMAGES_PER_USER = 25
NUM_STEPS = 50
BASE_SEED = 42
USERS = list(range(210))
DEVICE = "cuda"

# Fallbacks when profiles are sparse
FALLBACK_POS = ["nature", "landscape", "portrait", "cityscape", "animals"]
FALLBACK_STYLE = "high quality, detailed, natural lighting"
FALLBACK_NEG = ["low quality", "blurry", "deformed", "overexposed", "underexposed"]

# How many positive keywords to include per image prompt
K_POS_PER_PROMPT = (1, 2)  # inclusive range: choose 1 or 2


# ----------------------- Utils ------------------------
def load_profiles_jsonl(path: str) -> Dict[int, Dict[str, Any]]:
    """
    Load personas JSONL -> dict keyed by user_id.
    Each line should be a JSON object with at least:
      user_id, persona (str), keywords_positive (list[str]), keywords_negative (list[str])
    """
    profiles: Dict[int, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                uid = int(obj.get("user_id"))
                profiles[uid] = {
                    "persona": obj.get("persona", "") or "",
                    "pos": list(dict.fromkeys([str(x).strip().lower() for x in (obj.get("keywords_positive") or []) if str(x).strip()])),
                    "neg": list(dict.fromkeys([str(x).strip().lower() for x in (obj.get("keywords_negative") or []) if str(x).strip()])),
                }
            except Exception as e:
                print(f"[warn] Bad JSON at line {ln}: {e}")
    return profiles


def choose_pos_keywords(pos: List[str], image_idx: int) -> List[str]:
    """
    Deterministic cycle through pos keywords but vary count (1–2) with idx.
    """
    if not pos:
        return []
    k_min, k_max = K_POS_PER_PROMPT
    k = k_min if k_min == k_max else (k_min + (image_idx % (k_max - k_min + 1)))
    # cycle deterministically
    start = image_idx % len(pos)
    chosen = []
    for i in range(k):
        chosen.append(pos[(start + i) % len(pos)])
    return chosen


def trim_persona(persona: str, max_chars: int = 220) -> str:
    persona = " ".join(persona.strip().split())
    if len(persona) <= max_chars:
        return persona
    # trim to nearest sentence end if possible
    cut = persona[:max_chars]
    if "." in cut:
        cut = cut[:cut.rfind(".")+1]
    return cut


def build_prompts_for_user(profile: Dict[str, Any], user_id: Optional[int], img_idx: int) -> Tuple[str, Optional[str]]:
    """
    Compose (prompt, negative_prompt) robustly.
    """
    persona = trim_persona(profile.get("persona", "") or "")
    pos = profile.get("pos") or []
    neg = profile.get("neg") or []

    # Choose 1–2 positive keywords (cycle deterministically)
    pos_terms = choose_pos_keywords(pos, img_idx)

    # Subject line: if no pos terms, fallback
    if pos_terms:
        subject = ", ".join(pos_terms)
    else:
        # include user token if you actually condition on <uX> elsewhere; otherwise drop it
        subject = random.choice(FALLBACK_POS)

    # Add persona as style tail if present
    style_tail = f" {persona}" if persona else f" {FALLBACK_STYLE}"

    # Final positive prompt
    prompt = f"{subject}.{style_tail}".strip()

    # Negative prompt
    neg_terms = neg if neg else FALLBACK_NEG
    negative_prompt = ", ".join(neg_terms) if neg_terms else None

    return prompt, negative_prompt


def seed_for(user_id: int, img_idx: int) -> int:
    # Deterministic seed per (user, image)
    return BASE_SEED + 10_000 * user_id + img_idx




# ---------------------- Main --------------------------
def main():
    # Load profiles
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
        prof = profiles.get(user_id, {"persona":"", "pos":[], "neg":[]})

        # Per-user container
        gen_images = []

        for i in range(IMAGES_PER_USER):
            prompt, negative_prompt = build_prompts_for_user(prof, user_id, i)

            # Deterministic generator per image
            g = torch.Generator(device=DEVICE).manual_seed(seed_for(user_id, i))

            with torch.inference_mode():
                out = pipe(
                    prompt=prompt,
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
