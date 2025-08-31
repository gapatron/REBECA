import torch
from diffusers import StableDiffusionPipeline
import argparse


from utils import save_generated_data




def main():
    parser = argparse.ArgumentParser(description='Generate baseline images')
    parser.add_argument('-n_images', type=int, help='Number of images')
    parser.add_argument('-prompt_level', type=int, help='Prompt level')
    parser.add_argument('-dst_dir', type=str, help='Destination directory')
    args = parser.parse_args()
    n_images = args.n_images
    prompt_level  = args.prompt_level
    dst_dir = args.dst_dir

    model_id = "runwayml/stable-diffusion-v1-5"                                                                                                                                                                                                                
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to("cuda")
    pipe.safety_checker = None

    prompt_candidates = [
        ([""], [""]), 
        (
            ["high quality photo"], 
            ["bad quality photo, letters"]
        ),
        (
            ["Realistic image, finely detailed, with balanced composition and harmonious elements. "
             "Dynamic yet subtle tones, versatile style adaptable to diverse themes and aesthetics, "
             "prioritizing clarity and authenticity."],
            ["deformed, ugly, wrong proportion, frame, watermark, low res, bad anatomy, worst quality, low quality"]
        )
    ]

    if prompt_level==0:
        prompts = prompt_candidates[0]
    elif prompt_level==1:
        prompts = prompt_candidates[1]
    elif prompt_level==2:
        prompts = prompt_candidates[2]
    else:
        raise ValueError("Please select a prompt level from 0, 1 and 2.")
    
    pos_prompt, neg_prompt = prompts[0], prompts[1]
    save_images = list()

    for _ in range(n_images):
        with torch.no_grad():
                gen_images = pipe(
                    prompt=pos_prompt,
                    negative_prompt=neg_prompt,
                    num_inference_steps=100,
                    ).images
                torch.cuda.empty_cache()

        save_images.extend(gen_images)
    save_generated_data(save_images, dst_dir)



if __name__=="__main__":
    
    main()