import numpy as np
import os
import torch
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler
from torchvision.datasets.folder import default_loader

class FlatImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.files = [os.path.join(root, file) for file in os.listdir(root) if os.path.isfile(os.path.join(root, file))]
        self.transform = transform or (lambda x: x)
        self.loader = default_loader  # Default PIL image loader

    def __getitem__(self, idx):
        img = self.loader(self.files[idx])  # Load image
        if self.transform:
            img = self.transform(img)  # Apply transformation
        return img

    def __len__(self):
        return len(self.files)


class RecommenderDataset(Dataset):
    def __init__(self, dataframe, embeddings):
        self.users = dataframe['worker_id'].values
        self.items = dataframe['image_id'].values
        self.ratings = dataframe['score'].values
        self.embeddings = embeddings

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
            torch.tensor(self.ratings[idx], dtype=torch.float),
            (torch.tensor(self.ratings[idx], dtype=torch.float) >= 4).float(),
            self.embeddings[idx]
        )
    
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
            "image_emb": self.image_embeddings[index]
        }
        

    def __len__(self):
        return len(self.ratings)
    

class PinterestEmbeddingsDataset(Dataset):
    def __init__(self, dataframe, embeddings_dict) -> None:
        """
        Args:
        dataframe: DF containing image paths, board ids,
        embedding_dict: dictionary mapping image paths to image embeddings
        """
        super().__init__()
        self.dataframe = dataframe
        self.embeddings_dict = embeddings_dict
        self.board_index_dict = {board:i for i, board in enumerate(self.dataframe["board"].unique())}

    def __getitem__(self, index):
        return {
            "user_id": self.board_index_dict[self.dataframe["board"].iloc[index]],
            "board_id": self.dataframe["board"].iloc[index],
            "im_path": self.dataframe["im_path"].iloc[index],
            "pin": self.dataframe["pin"].iloc[index],
            "rating": 1.0, # All pinterest pins denote interest
            "like":  1.0,
            "image_emb": self.embeddings_dict[self.dataframe["im_path"].iloc[index]]
        }
        

    def __len__(self):
        return len(self.dataframe)


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
                                              replace=True)
            all_samples.extend(worker_samples)
        # Create a new shuffled array
        shuffled_samples = np.random.permutation(all_samples)
    
        # Convert to iterator and return
        return iter(shuffled_samples)


    def __len__(self):
        return self.num_users * self.samples_per_user
    

class RecommenderBoardSampler(Sampler):
    """
    Board Sampler for Pinteres Data
    """
    def __init__(self, board_pin_image_df, num_boards, samples_per_board) -> None:
        self.board_pin_image_df = board_pin_image_df
        self.unique_boards = self.board_pin_image_df["board"].unique()
        #assert len(self.unique_workers) >= num_users, "Unique workers can not be fewer than number of users"
        self.num_boards = num_boards
        self.samples_per_board = samples_per_board

        self.board_to_pinindex = board_pin_image_df.groupby("board").apply(lambda x: x.index.tolist()).to_dict()

    def __iter__(self):
        boards = np.random.choice(self.unique_boards, size=self.num_boards, replace=False)
        all_samples = []
        for board_id in boards:
            pins_samples = np.random.choice(self.board_to_pinindex[board_id], 
                                              size=min(self.samples_per_board, len(self.board_to_pinindex[board_id])), 
                                              replace=True)
            all_samples.extend(pins_samples)

        # Create a new shuffled array
        shuffled_samples = np.random.permutation(all_samples)
        # Convert to iterator and return
        return iter(shuffled_samples)


    def __len__(self):
        return self.num_boards * self.samples_per_board