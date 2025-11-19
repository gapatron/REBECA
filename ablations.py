import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler
import argparse
from sampling import sample_user_images
from prior_models import RebecaDiffusionPrior
from tqdm import tqdm
from utils import save_generated_data
import gc




def main(args):
    if args.experiment_type == "cross-cfgs":
         
        device = "cuda"
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        pipe.safety_checker = None
        filedir = "./data/flickr/evaluation/diffusion_priors/models/weights/"
        winner = "comprehensive_study_20250830_013540/modelrdp_num_layers6_num_heads8_hidden_dim128_tokens32_lr0.0001_optadamw_schreduce_on_plateau_bs64_nssquaredcos_cap_v2_ts1000_spu100_csFalse_objsample_normnone_uthr0"
        rdp = RebecaDiffusionPrior(
                            img_embed_dim=1024,
                            num_users=210,
                            num_tokens=32,
                            hidden_dim=128,
                            n_heads=8,
                            num_layers=6,
                            score_classes=2,
                        ).to(device)
        
        rdp.load_state_dict(torch.load(f'{filedir}/{winner}.pth'))
        noise_scheduler = DDPMScheduler(
                            num_train_timesteps=1000,
                            beta_schedule="squaredcos_cap_v2",
                            clip_sample=False,
                            prediction_type="sample"
                        )
        
        cfgs  = [3, 6, 9, 12, 15] 
        pipe_cfg = [1, 3, 5, 7, 10]
        images_per_user = 10
        savedir = "./data/flickr/evaluation/ablations/cross-cfgs/"
        users = list(range(210))
        seed = 2
        torch.manual_seed(seed)
        
        for cfg in tqdm(cfgs):
            for sd_cfg in pipe_cfg:
                gen_data = sample_user_images(
                                diffusion_prior_model=rdp,
                                diffusion_pipe=pipe,
                                users=users,
                                images_per_user=images_per_user,  # Fewer images per user to save memory
                                noise_scheduler=noise_scheduler,
                                guidance_scale=cfg,
                                prompt=[""],
                                negative_prompt=[""],
                                img_embedding_size=1024,
                                pipe_cfg=sd_cfg,
                                device=device,
                            )
                torch.save(gen_data, f"{savedir}/embcfg_{cfg}_pipecfg_{sd_cfg}_seed_{seed}.data")
                del gen_data
                torch.cuda.empty_cache()
                gc.collect()

    if args.experiment_type == "cross-prompts":
         
        device = "cuda"
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        pipe.safety_checker = None
        filedir = "./data/flickr/evaluation/diffusion_priors/models/weights/"
        winner = "comprehensive_study_20250830_013540/modelrdp_num_layers6_num_heads8_hidden_dim128_tokens32_lr0.0001_optadamw_schreduce_on_plateau_bs64_nssquaredcos_cap_v2_ts1000_spu100_csFalse_objsample_normnone_uthr0"
        rdp = RebecaDiffusionPrior(
                            img_embed_dim=1024,
                            num_users=210,
                            num_tokens=32,
                            hidden_dim=128,
                            n_heads=8,
                            num_layers=6,
                            score_classes=2,
                        ).to(device)
        
        rdp.load_state_dict(torch.load(f'{filedir}/{winner}.pth'))
        noise_scheduler = DDPMScheduler(
                            num_train_timesteps=1000,
                            beta_schedule="squaredcos_cap_v2",
                            clip_sample=False,
                            prediction_type="sample"
                        )
        
        rebeca_cfg  = 7.0
        pipe_cfg = 5.0
        images_per_user = 10
        savedir = "./data/flickr/evaluation/ablations/cross-prompts/"
        users = list(range(210))
        seed = 3
        torch.manual_seed(seed)
        
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

        for i, prompt_level in enumerate(prompt_candidates):

            pos_prompt, neg_prompt = prompt_level[0], prompt_level[1]
            gen_data = sample_user_images(
                                    diffusion_prior_model=rdp,
                                    diffusion_pipe=pipe,
                                    users=users,
                                    images_per_user=images_per_user,  # Fewer images per user to save memory
                                    noise_scheduler=noise_scheduler,
                                    guidance_scale=rebeca_cfg,
                                    prompt=pos_prompt,
                                    negative_prompt=neg_prompt,
                                    img_embedding_size=1024,
                                    pipe_cfg=pipe_cfg,
                                    device=device,
                                )
            torch.save(gen_data, f"{savedir}/embcfg_{rebeca_cfg}_pipecfg_{pipe_cfg}_promptlevel_{i}_seed_{seed}.data")
            del gen_data
            torch.cuda.empty_cache()
            gc.collect()

    elif args.experiment_type == "baseline-prompts":
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
                        num_inference_steps=50,
                        ).images
                    torch.cuda.empty_cache()

            save_images.extend(gen_images)
        torch.save(save_images, f"./data/flickr/evaluation/ablations/baseline-prompts/images_prompt_level_{prompt_level}.imgs")
        #save_generated_data(save_images, dst_dir)
    else:
        raise NotImplementedError("{args.experiment_type} is not implemented yet.")



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Generate baseline images')
    parser.add_argument('--experiment_type', type=str, required=True, default="cross-cfg", help='Experiment type. Implemented cross-cfg, varying embedding generator CFG with Image generator CFG.')
    parser.add_argument('--n_images', type=int, default=100, help='Number of images')
    parser.add_argument('--prompt_level', type=int, help='Prompt level')
    parser.add_argument('--dst_dir', type=str, help='Destination directory')
    args = parser.parse_args()
    main(args)