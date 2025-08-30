import numpy as np
import os
import torch
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler
from torchvision.datasets.folder import default_loader


class EmbeddingsDataset(Dataset):
    def __init__(self, ratings_df, image_embeddings) -> None:
        super().__init__()
        self.users = ratings_df['worker_id'].values
        self.ratings = ratings_df["score"].values
        self.image_paths = ratings_df["imagePair"].values
        self.image_embeddings = image_embeddings


    def __getitem__(self, index):
        return {
            "user_id": self.users[index],
            "rating": self.ratings[index],
            "like":  (self.ratings[index]>=4),
            "image_path": self.image_paths[index],
            "image_emb": self.image_embeddings[index],
        }
        

    def __len__(self):
        return len(self.ratings)
    


class RecommenderUserSampler(Sampler):
    def __init__(self, ratings_df, num_users, samples_per_user) -> None:
        self.ratings_df = ratings_df
        self.unique_workers = ratings_df["worker_id"].unique()
        #assert len(self.unique_workers) >= num_users, "Unique workers can not be fewer than number of users"
        self.num_users = num_users
        self.samples_per_user = samples_per_user

        self.worker_to_ratings = {worker: ratings_df[ratings_df["worker_id"] == worker].index.tolist() 
                                  for worker in self.unique_workers}

    def __iter__(self):
        workers = np.random.choice(self.unique_workers, size=self.num_users, replace=False)
        all_samples = []
        for worker_id in workers:
            worker_samples = np.random.choice(self.worker_to_ratings[worker_id], 
                                              size=min(self.samples_per_user, len(self.worker_to_ratings[worker_id])), 
                                              replace=False)
            all_samples.extend(worker_samples)
        # Create a new shuffled array
        shuffled_samples = np.random.permutation(all_samples)
    
        # Convert to iterator and return
        return iter(shuffled_samples)


    def __len__(self):
        return self.num_users * self.samples_per_user
    
