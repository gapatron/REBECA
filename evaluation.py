from diffusers.utils import load_image
import torch
from torch.utils.data import DataLoader
from torcheval.metrics import FrechetInceptionDistance
from torchvision import transforms
from Datasets import FlatImageDataset
from tqdm import tqdm




class ModelEvaluator:
    """
    Model evaluation
    """
    def __init__(self, 
                diffusion_prior_model,
                scheduler,
                loaded_pipe=None,
                num_users=210,
                original_img_dir="./data/raw/flickr/40K/", 
                model_dir=".",
                fid_batch_size=64, 
                fid_feature_dim=2048,
                device="cuda",
                ):
        """
        Assuming a Stable Diffusion Model for now, requires changes for Kandinsky v2.2

        diffusion_prior_model: the diffusion prior model to be evaluated
        scheduler: the noise scheduler used to sample from the diffusion prior.
        loaded_pipe: text2img (+ipadapter) pipe to inject embeddings into
        eval_dir: evaluation directory where evaluation data is.
        """
        self.num_users = num_users
        self.diffusion_prior = diffusion_prior_model
        self.scheduler = scheduler


        self.original_img_dir = original_img_dir

        self.model_dir = model_dir
        self.image_emb_dir = f"{self.model_dir}/embeddings/"
        self.image_dir = f"{self.model_dir}/images/"
        self.originals_dir = f"{self.model_dir}/original_eval_data/"

        self.sdpipe = loaded_pipe

        self.fid_batch_size = fid_batch_size
        self.fid_feature_dim = fid_feature_dim
        self.fid = FrechetInceptionDistance(feature_dim=2048)

        self.device = device

    def eval_fid(self, path_original=None, path_generated=None):

        transform_originals = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

        transform_generated = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Use FlatImageDataset instead of ImageFolder
        real_dataset = FlatImageDataset(root=path_original, transform=transform_originals)
        real_dataloader = DataLoader(real_dataset, batch_size=self.fid_batch_size)

        generated_dataset = FlatImageDataset(root=path_generated, transform=transform_generated)
        generated_dataloader = DataLoader(generated_dataset, batch_size=self.fid_batch_size)

        for real_batch in tqdm(real_dataloader):
            self.fid.update(real_batch, is_real=True)

        for generated_batch in tqdm(generated_dataloader):
            self.fid.update(generated_batch, is_real=False)

        score = self.fid.compute()

        return score

    def eval_cmmd(self,
                    path_original=None,
                    path_generated=None,
                    path_original_emb=None,
                    path_generated_emb=None,
                    batch_size=32,
                    max_count=-1,
                    device=torch.device('cuda'),
                    emb_flag = False,
                  ):
        
        """
        Taken from here: https://github.com/sayakpaul/cmmd-pytorch/blob/main/embedding.py
        Input: embeddings
        Output: permutation test
        """

        # becomes slow if we are using embeddings directly -- sufficient to import only mmd in this case (can give us a good speedup)
        from eval_utils import mmd
        if emb_flag:
            from eval_utils import compute_embeddings_for_dir, ClipEmbeddingModel
        if path_original and path_original_emb:
            raise ValueError("`path_original` and `path_original_emb` both cannot be set at the same time.")
        if path_generated and path_generated_emb:
            raise ValueError("`path_generated` and `path_generated_emb` both cannot be set at the same time.")
        if emb_flag:
            embedding_model = ClipEmbeddingModel()

        if path_original_emb is not None:
            original_embs = torch.load(path_original_emb, map_location=device)
        else:
            original_embs = compute_embeddings_for_dir(path_original, embedding_model, batch_size=batch_size, max_count=max_count)
            torch.save(original_embs, f"{path_original}/../embeddings/cmmd_embeddings.pth")
        if path_generated_emb is not None:
            generated_embs = torch.load(path_generated_emb, map_location=device)
        else:
            generated_embs = compute_embeddings_for_dir(path_generated, embedding_model, batch_size=batch_size, max_count=max_count)
            torch.save(generated_embs, f"{path_generated}/../embeddings/cmmd_embeddings.pth")
        
        cmmd_distance = mmd(original_embs, generated_embs)
        return cmmd_distance

    
    def eval_cosine_similarity(self,
                               path_original_emb=None,
                               path_generated_emb=None,
                               device=torch.device('cuda'),):
        """
        Compute cosine similarity between original and generated embeddings
        """

        csim = torch.nn.CosineSimilarity(dim=0, eps=1e-8)
        original_embs = torch.load(path_original_emb, map_location=device)
        generated_embs = torch.load(path_generated_emb, map_location=device)
        # first compute average embeddings by reducing along image dimension
        original_embs_averaged = torch.mean(original_embs, dim=0)
        generated_embs_averaged = torch.mean(generated_embs, dim=0)
        return csim(original_embs_averaged, generated_embs_averaged)