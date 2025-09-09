from diffusers import StableDiffusionPipeline, DDPMScheduler,DDIMInverseScheduler, AutoencoderKL, DDIMScheduler
import torch
from PIL import Image
from torchvision import transforms as tvt

class Editor():
    def __init__(
            self,
            pipe=None,
            method=None,
            num_steps=None,
            classifier=None,
            device="cuda"
    )
        self.device = device if torch.cuda.is_available() else "cpu"
        if num_steps is None:
            self.num_steps = 50
        if pipe is None:
            self.pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                   safety_checker=None,
                                                   torch_dtype=dtype).to(device)
            self.vae = pipe.vae
            self.pip.safety_checker = None

    def load_image(
            self,
            impath
    ):
        return Image.open(impath).convert("RGB").resize((512, 512))
        
    def img_to_latents(x: torch.Tensor, vae: AutoencoderKL):
        x = 2. * x - 1.
        posterior = vae.encode(x).latent_dist
        latents = posterior.mean * 0.18215
        return latents
    
    def ddim_invert(
            self,
            image=None,
            inverse_scheduler=None,
            num_steps=None,
    )
        if type(image)=="str":
            image = self.load(image)
        if num_steps == None:
            num_steps = self.num_steps
        assert type(image)==Image
        # Convert image to tensor before 
        latents = self.img_to_latents(tvt.ToTensor()(image)[None, ...])
        
        if inverse_scheduler is None:
            inverse_scheduler = DDIMInverseScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder='scheduler')

        pipe.scheduler = inverse_scheduler
        inv_latents, _ = pipe(prompt="", negative_prompt="", guidance_scale=1.,
                          width=input_img.shape[-1], height=input_img.shape[-2],
                          output_type='latent', return_dict=False,
                          num_inference_steps=num_steps, latents=latents)
        
        return inv_latents
    


    def slerp(self, emb_1, emb_2, t=0.01):
        # Emb 1 stands for the original image embedding
        # Emb 2 stands for the sampled personalized embedding
        p0 = emb_1/emb_1.norm()
        p1 = emb_2/emb_2.norm()

        Omega = torch.arccos((p0 * p1).sum())

        slerp = torch.sin((1 - t)*Omega) * p0 + torch.sin(t * Omega) * p1
        slerp /= torch.sin(Omega)

        return slerp
    

    def edit(
            self,
            image=None,
            method=None,
            num_steps=None,
            personalized_embs=None
    ):
        if image is None:
            raise ValueError("Image can't be None")
        if method is None:
            method = self.method
        if num_steps is None:
            num_steps = self.num_steps
        if personalized_embs is None:
            raise ValueError("Personalization Embeddings can't be None")
        
        if method == "ddminv":
            inv_latents = self.ddim_invert(
                image=image,
            )
            self.pipe.unload_ip_adapter()
            with torch.inference_mode():
                sampled_img_embs = pipe.encode_image(input_img, device=device, num_images_per_prompt=1)


            self.pipe.scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder='scheduler')
            self.pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")

            image = self.pipe(prompt="", negative_prompt="", guidance_scale=1.,
                     num_inference_steps=num_steps, latents=inv_latents)
            
            
        elif method == "clipslerp":
            raise NotImplementedError("Not implemented quite fucking yet")
