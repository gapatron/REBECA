import matplotlib.pyplot as plt

import numpy as np
import torch
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def visualize_image(image_tensor):
    """
    Visualizes an RGB image tensor.
    
    Parameters:
        image_tensor (torch.Tensor): An RGB image tensor of shape [3, H, W] with values in [0,1].
    """
    # Convert tensor from shape [3, H, W] to [H, W, 3] and then to a NumPy array.
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    
    # Display the image.
    plt.imshow(image_np)
    plt.axis('off')  # Hide axis ticks and labels.
    plt.show()

def Rate(shape, color, shapes, colors, ps = [.01, .10, .95], seed=0):
    z = np.hstack((np.array(shapes)[:,None], np.array(colors)[:,None]))
    idx = (z == np.array([[shape, color]])).sum(1)
    np.random.seed(seed)
    return torch.tensor(np.random.binomial(1,[ps[i] for i in idx])).float()
    
def GenerateImages(shape, color, n, X, S, seed=0):
    """
    Generates n RGB images of the specified shape.
    
    Parameters:
        X (torch.Tensor): Tensor of shape [N, 64, 64] with grayscale images.
        S (torch.Tensor): Tensor of shape [N] with shape labels (0: square, 1: ellipse, 2: heart).
        shape (str): Desired shape to filter ('square', 'ellipse', or 'heart').
        n (int): Number of samples to generate.
        seed (int): Random seed for reproducibility.
        color (tuple or list): 'reg', 'green, 'blue'
    
    Returns:
        torch.Tensor: A tensor of shape [n, 3, 64, 64] with the colored images.
    """

    color_code = {'white':(1,1,1),'red':(1,0,0),'green':(0,1,0),'blue':(0,0,1)}[color]
    
    # Map shape string to its corresponding label.
    shape_mapping = {"square": 0, "ellipse": 1, "heart": 2}
    
    if shape not in shape_mapping:
        raise ValueError("Invalid shape. Please choose from 'square', 'ellipse', or 'heart'.")
    
    shape_label = shape_mapping[shape]
    
    # Find the indices of images matching the specified shape.
    indices = (S == shape_label).nonzero(as_tuple=True)[0]
    
    if indices.numel() < n:
        raise ValueError("Not enough images with the specified shape to sample from.")
    
    # Set the random seed and shuffle indices.
    torch.manual_seed(seed)
    permuted_indices = indices[torch.randperm(indices.numel())]
    
    # Select n indices.
    sampled_indices = permuted_indices[:n]
    
    # Get the corresponding images.
    selected_images = X[sampled_indices]  # shape: [n, 64, 64]
    
    # Expand the grayscale images to have a channel dimension.
    selected_images = selected_images.unsqueeze(1)  # shape: [n, 1, 64, 64]
    
    # Convert the color to a tensor and reshape to be broadcastable.
    # Assuming the color values are in the same scale as the grayscale images (0 to 1).
    color_tensor = torch.tensor(color_code, dtype=selected_images.dtype, device=selected_images.device)
    color_tensor = color_tensor.view(1, 3, 1, 1)  # shape: [1, 3, 1, 1]
    
    # Multiply the grayscale image (intensity) by the color.
    # This scales each channel of the RGB image according to the provided color.
    colored_images = selected_images * color_tensor  # shape: [n, 3, 64, 64]
    colors = colored_images.shape[0]*[color]
    shapes = colored_images.shape[0]*[shape]
    return colored_images, shapes, colors

def compute_user_mean_embeddings(encoded_image_stack, R):
    """
    encoded_image_stack: Tensor or array of shape (n_samples, latent_dim)
    R: binary matrix (n_samples, num_users), 1 if user liked the image
    Returns: (num_users, latent_dim)
    """
    if isinstance(encoded_image_stack, torch.Tensor):
        encoded_image_stack = encoded_image_stack.cpu().numpy()
    if isinstance(R, torch.Tensor):
        R = R.cpu().numpy()

    num_users = R.shape[1]
    latent_dim = encoded_image_stack.shape[1]
    user_means = np.zeros((num_users, latent_dim))

    for user in range(num_users):
        liked_indices = np.where(R[:, user] == 1)[0]
        if len(liked_indices) > 0:
            user_means[user] = encoded_image_stack[liked_indices].mean(axis=0)
        else:
            user_means[user] = np.zeros(latent_dim)  # fallback

    return user_means
