import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler
from sampling import sample_user_images
from prior_models import RebecaDiffusionPrior
import os



def main():
    
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
        
    rebeca_cfg  = 5.0
    pipe_cfg = 5.0
    images_per_user = 25

    seed = 42
    torch.manual_seed(seed)
    
    savedir = f"./data/flickr/evaluation/diffusion_priors/models/samples/embcfg_{rebeca_cfg}_pipecfg_{pipe_cfg}_seed_{seed}"
    savepath = f"{savedir}/embcfg_{rebeca_cfg}_pipecfg_{pipe_cfg}_seed_{seed}"
    users = list(range(210))
    os.makedirs(f"{savedir}", exist_ok=True)

    gen_data = sample_user_images(
                                diffusion_prior_model=rdp,
                                diffusion_pipe=pipe,
                                users=users,
                                images_per_user=images_per_user,  
                                noise_scheduler=noise_scheduler,
                                guidance_scale=rebeca_cfg,
                                prompt=[""],
                                negative_prompt=[""],
                                img_embedding_size=1024,
                                pipe_cfg=pipe_cfg,
                                device=device,
                                savepath=savepath
                                )
    
    del gen_data


if __name__=="__main__":
    main()