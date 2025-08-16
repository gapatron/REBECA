import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import json
import os

def retrieve_user_imgs(user_id, ratings_df):
    """
    Returns the list of images rated by a user.
    """
    images = ratings_df[ratings_df["worker_id"] == user_id][["worker_id", "image_id", "imagePair", "score"]]

    return images

def compute_img_rating_statistics(ratings_df):
    """
    Returns the list of images rated by a user.
    """
    image_stats = ratings_df.groupby(["image_id", "imagePair"])["score"].agg(["min", "mean", "median", "max", "count", "std"]).reset_index()
    #image_stats = image_stats[image_stats["count"] >= min_count]
    return image_stats 

def plot_image(statistics_df, n, filters, rating):

    statistics_df[statistics_df[filters] == rating]

def plot_user_images(image_dir, user_df, n=6):
    # Check if the number of requested images is greater than the DataFrame's size
    if n > len(user_df):
        n = len(user_df)
        print(f"Requested number of images reduced to {n} due to DataFrame size.")
    
    # Sample n random rows from the DataFrame
    sampled_df = user_df.sample(n=n)
    
    # Prepare to plot in a grid layout, calculate number of rows and columns for the grid
    rows = int(np.sqrt(n))
    cols = (n // rows) + (0 if n % rows == 0 else 1)
    
    fig, axs = plt.subplots(rows, cols, figsize=(10,8))
    # Flatten the axes array if more than one row and column
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]
    
    for ax, (index, row) in zip(axs, sampled_df.iterrows()):
        image_path = f"{image_dir}/{row['imagePair']}"
        img = Image.open(image_path)
        sqrWidth = np.ceil(np.sqrt(img.size[0]*img.size[1])).astype(int)
        img_resize = img.resize((sqrWidth, sqrWidth))
        ax.imshow(img_resize)
        ax.axis('off')  # Hide axes
        #ax.set_title(f'Image ID: {row["image_id"]}\nScore: {row["score"]}', fontsize=16)
    
    # If there are any leftover axes, turn them off
    for ax in axs[len(sampled_df):]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_generated_images(image_dir, n=6):
    # Check if the number of requested images is greater than the DataFrame's size
    if n > len(os.listdir(image_dir)):
        n = len(os.listdir(image_dir))
        print(f"Requested number of images reduced to {n} due to limited generated images.")
    
    # Sample n random rows from the DataFrame
    global_idx = np.arange(0, len(os.listdir(image_dir)))
    sampled_idx = np.random.choice(global_idx, replace=False, size=n)

    # Prepare to plot in a grid layout, calculate number of rows and columns for the grid
    rows = int(np.sqrt(n))
    cols = (n // rows) + (0 if n % rows == 0 else 1)
    
    fig, axs = plt.subplots(rows, cols, figsize=(10,8))
    # Flatten the axes array if more than one row and column
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]
    
    for ax, (index, row) in zip(axs, enumerate(sampled_idx)):
        upath = os.listdir(image_dir)[row]
        image_path = f"{image_dir}/{upath}"
        img = Image.open(image_path)
        sqrWidth = np.ceil(np.sqrt(img.size[0]*img.size[1])).astype(int)
        img_resize = img.resize((sqrWidth, sqrWidth))
        ax.imshow(img_resize)
        ax.axis('off')  # Hide axes
        #ax.set_title(f'Image ID: {row["image_id"]}\nScore: {row["score"]}', fontsize=16)
    
    # If there are any leftover axes, turn them off
    for ax in axs[len(sampled_idx):]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_filtered_images(image_dir, img_df, n=6):
    # Check if the number of requested images is greater than the DataFrame's size
    if n > len(img_df):
        n = len(img_df)
        print(f"Requested number of images reduced to {n} due to DataFrame size.")
    
    # Sample n random rows from the DataFrame
    sampled_df = img_df.sample(n=n)
    
    # Prepare to plot in a grid layout, calculate number of rows and columns for the grid
    rows = int(np.sqrt(n))
    cols = (n // rows) + (0 if n % rows == 0 else 1)
    
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    # Flatten the axes array if more than one row and column
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]
    
    for ax, (index, row) in zip(axs, sampled_df.iterrows()):
        image_path = f"{image_dir}/{row['imagePair']}"
        img = Image.open(image_path)
        ax.imshow(img)
        ax.axis('off')  # Hide axes
        ax.set_title(f'Image ID: {row["image_id"]}\nMean: {row["mean"]}\nMin: {row["min"]}', fontsize=16)
    
    # If there are any leftover axes, turn them off
    for ax in axs[len(sampled_df):]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

###################################### Worker Embedding Utils

def compute_similarity_matrix(embeddings):
    # Normalize the embeddings to unit vectors
    norm_embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    # Compute cosine similarity
    similarity_matrix = torch.mm(norm_embeddings, norm_embeddings.t())
    return similarity_matrix

def plot_similarity_matrix(similarity_matrix):
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, interpolation='none', cmap='viridis')
    plt.colorbar()
    plt.title('Cosine Similarity Matrix')
    plt.xlabel('Worker Index')
    plt.ylabel('Worker Index')
    plt.grid(False)
    plt.show()

def find_top_n_similar(similarity_matrix, n=5):
    # Exclude the self-similarity by filling diagonal with lowest value
    similarity_matrix.fill_diagonal_(-1)
    # Get indices of the top n similar embeddings for each worker
    top_n_indices = torch.topk(similarity_matrix, k=n, dim=1)[1]
    return top_n_indices


# From @Zhiwei
def map_embeddings_to_ratings(image_features, ratings_df):
    unique_pairs = ratings_df['imagePair'].unique()
    if len(unique_pairs) != image_features.shape[0]:
        raise ValueError("Number of unique image pairs does not match number of image embeddings")
    pair_to_idx = {pair: idx for idx, pair in enumerate(unique_pairs)}
    indices = ratings_df['imagePair'].map(pair_to_idx).values
    # Use indices to expand embeddings tensor
    expanded_embeddings = image_features[indices]
    return expanded_embeddings


########################################################################################
################################### Data Utils #########################################
########################################################################################


from Datasets import RecommenderUserSampler
import os
import pandas as pd
import random
import shutil

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def split_recommender_data(
        ratings_df,
        val_spu=5, 
        test_spu=10,
        seed = 0
         ):
    set_seeds(seed)
    unique_users = ratings_df["worker_id"].unique()

    
    # The User Sampler samples 'eval_spu' observations per user uniformly, which addresses user skewness.  
    test_ratings_sampler = RecommenderUserSampler(ratings_df=ratings_df, num_users=len(unique_users), samples_per_user=test_spu)
    # Generate test_indices and turn it into a set for O(1) lookup
    test_indices_set = set([i for i in test_ratings_sampler])
    # Partition ratings_df into train_val_df by and eval_df
    train_val_mask = ~ratings_df.index.isin(test_indices_set)
    train_val_df = ratings_df[train_val_mask]
    test_df = ratings_df[~train_val_mask]

    # Further partition train_val_df into train_df and val_df
    val_ratings_sampler = RecommenderUserSampler(ratings_df=train_val_df, num_users=len(unique_users), samples_per_user=val_spu)
    val_indices_set = set([i for i in val_ratings_sampler])
    train_mask = ~train_val_df.index.isin(val_indices_set)
    train_df = train_val_df[train_mask]
    val_df = train_val_df[~train_mask]

    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Evaluation set size: {len(test_df)}")

    # When creating train_df, keep track of the original indices
    train_df = train_df.reset_index(drop=False)
    train_df = train_df.rename(columns={'index': 'original_index'})

    val_df = val_df.reset_index(drop=False)
    val_df = val_df.rename(columns={'index': 'original_index'})

    test_df = test_df.reset_index(drop=False)
    test_df = test_df.rename(columns={'index': 'original_index'})
    

    return train_df, val_df, test_df


def save_generated_data(
        image_list,
        dir,
):  
    """
    Stores generated images in image_list to dir. 
    savename has user_id and img_id based on that.
    """
    os.makedirs(dir, exist_ok=True)
    for i, image in enumerate(image_list):
        savepath = os.path.join(dir, f"image_{i}.png")
        image.save(savepath)


def save_generated_data_to_user_dir(
        data,
        dir,
):  
    """
    Stores generated images in image_list to dir. 
    Assumes same samples per user.
    savename has user_id and img_id based on that.
    """
    os.makedirs(dir, exist_ok=True)
    for key in data.keys():
        user_dir = f"./{dir}/user_{key}"
        img_user_dir = f"./{dir}/user_{key}/images"
        emb_user_dir = f"./{dir}/user_{key}/embeddings"
        os.makedirs(user_dir, exist_ok=True)

        os.makedirs(img_user_dir, exist_ok=True)
        os.makedirs(emb_user_dir, exist_ok=True)


        for img_id, image in enumerate(data[key]["images"]):
            savepath = f"{user_dir}/images/img_{img_id}.png"
            image.save(savepath)

        prior_embeddings_tensor = data[key]["prior_embeddings"]
        
        # Save prior embeddings
        torch.save(prior_embeddings_tensor, f"{emb_user_dir}/prior_embeddings.pth")
        
        # Save posterior embeddings if they exist
        if "posterior_embeddings" in data[key]:
            embeddings_tensor = data[key]["posterior_embeddings"]
            torch.save(embeddings_tensor, f"{emb_user_dir}/embeddings.pth")
    


def store_eval_images(
        paths_iter,
        src_dir,
        dst_dir
):
    for filename in paths_iter:
        shutil.copy(os.path.join(src_dir, filename), dst_dir)

def store_eval_images_per_user(
        liked_df,
        src_dir,
        dst_base_dir
):
    for user_id in liked_df.worker_id.unique():
        liked_user = liked_df[liked_df["worker_id"] == user_id]
        dst_dir = f"{dst_base_dir}/user_{user_id}/images/"
        os.makedirs(dst_dir, exist_ok=True)
        store_eval_images(
            paths_iter=liked_user["imagePair"],
            src_dir=src_dir,
            dst_dir=dst_dir
        )

def save_rebeca_config(model_class, model_kwargs, weights_file, output_dir, diffusion_model_id="runwayml/stable-diffusion-v1-5", ip_adapter=None, num_train_timesteps=6000, num_users=None, img_embed_dim=None):
    """
    Save a config.json file for REBECA model loading.
    Args:
        model_class (str): Name of the model class (e.g., 'CrossAttentionDiffusionPrior')
        model_kwargs (dict): Arguments for the model constructor
        weights_file (str): Filename of the model weights
        output_dir (str): Directory to save config.json
        diffusion_model_id (str): Huggingface model id for diffusion pipeline
        ip_adapter (dict or None): IP adapter config
        num_train_timesteps (int): Number of diffusion timesteps
        num_users (int or None): Number of users (optional)
        img_embed_dim (int or None): Embedding dimension (optional)
    """
    config = {
        "model_class": model_class,
        "model_kwargs": model_kwargs,
        "weights_file": weights_file,
        "diffusion_model_id": diffusion_model_id,
        "num_train_timesteps": num_train_timesteps
    }
    if ip_adapter is not None:
        config["ip_adapter"] = ip_adapter
    if num_users is not None:
        config["num_users"] = num_users
    if img_embed_dim is not None:
        config["img_embed_dim"] = img_embed_dim
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved REBECA config to {config_path}")

