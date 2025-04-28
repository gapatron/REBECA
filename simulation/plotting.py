
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.stats import gaussian_kde


def plot_latents_umap_hearts(X_2d, shapes, colors, max_per_group=None, figsize=(8, 6), save_path=None):
    shape_to_marker = {
        'square': 's',
        'heart': '♥',
    }

    color_to_color = {
        'red': 'red',
        'blue': 'blue',
    }

    # Subsample
    shapes = np.array(shapes)
    colors = np.array(colors)
    selected_indices = []

    for shape in shape_to_marker:
        for color in color_to_color:
            mask = (shapes == shape) & (colors == color)
            indices = np.where(mask)[0]

            if max_per_group is not None and len(indices) > max_per_group:
                indices = np.random.choice(indices, max_per_group, replace=False)

            selected_indices.extend(indices.tolist())

    # Apply subset
    X_2d = X_2d[selected_indices]
    shapes = shapes[selected_indices]
    colors = colors[selected_indices]

    # Start plot
    fig, ax = plt.subplots(figsize=figsize)

    for shape in shape_to_marker:
        for color in color_to_color:
            mask = (shapes == shape) & (colors == color)

            if shape == 'square':
                ax.scatter(
                    X_2d[mask, 0],
                    X_2d[mask, 1],
                    marker='s',
                    color=color_to_color[color],
                    edgecolors='k',
                    alpha=0.6,
                    s=60,
                    linewidths=0.5,
                    label=f"{shape}-{color}"
                )

            elif shape == 'heart':
                for x, y in X_2d[mask]:
                    t = ax.text(
                        x, y, shape_to_marker['heart'],
                        fontsize=12,
                        color=color_to_color[color],
                        ha='center', va='center',
                        fontweight='bold',
                        zorder=3
                    )
                    # Add black outline to heart
                    t.set_path_effects([
                        path_effects.Stroke(linewidth=1.5, foreground='black'),
                        path_effects.Normal()
                    ])

    # Clean plot
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    #ax.set_title("Latent Space by Shape-Color (UMAP)", fontsize=16)

    # Custom legend
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label='square-red', markerfacecolor='red', markersize=10, markeredgecolor='k'),
        Line2D([0], [0], marker='s', color='w', label='square-blue', markerfacecolor='blue', markersize=10, markeredgecolor='k'),
        Line2D([0], [0], marker='$♥$', color='red', label='heart-red', markersize=14),
        Line2D([0], [0], marker='$♥$', color='blue', label='heart-blue', markersize=14),
    ]
    #ax.legend(handles=legend_elements, frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left')

    # Expand limits slightly to avoid cutoffs
    x_margin = (X_2d[:, 0].max() - X_2d[:, 0].min()) * 0.05
    y_margin = (X_2d[:, 1].max() - X_2d[:, 1].min()) * 0.05
    ax.set_xlim(X_2d[:, 0].min() - x_margin, X_2d[:, 0].max() + x_margin)
    ax.set_ylim(X_2d[:, 1].min() - y_margin, X_2d[:, 1].max() + y_margin)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()


def plot_user_latent_preferences(latent_np, shapes, colors, R, user_idx, max_per_group=300, save_path=None):
    assert latent_np.shape[0] == R.shape[0] == len(shapes) == len(colors)

    shape_to_marker = {
        'square': 's',
        'heart': '♥',  # Real heart as text
    }

    color_to_color = {
        'red': 'red',
        'blue': 'blue',
    }

    shapes = np.array(shapes)
    colors = np.array(colors)

    # Precompute masks for subsampling
    selected_indices = []
    for shape in shape_to_marker:
        for color in color_to_color:
            mask = (shapes == shape) & (colors == color)
            indices = np.where(mask)[0]
            if len(indices) > max_per_group:
                indices = np.random.choice(indices, max_per_group, replace=False)
            selected_indices.extend(indices.tolist())

    # Subset everything
    latent_np = latent_np[selected_indices]
    shapes = shapes[selected_indices]
    colors = colors[selected_indices]
    R = R[selected_indices]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Collect all liked points for user to compute mean after plotting
    all_liked_points = []

    for shape in shape_to_marker:
        for color in color_to_color:
            mask = (shapes == shape) & (colors == color)
            liked_mask = mask & (R[:, user_idx] == 1)
            not_liked_mask = mask & (R[:, user_idx] == 0)

            # Save liked points for mean
            all_liked_points.append(latent_np[liked_mask])

            # Plot not liked (gray, low alpha)
            if np.sum(not_liked_mask) > 0:
                if shape == 'square':
                    ax.scatter(
                        latent_np[not_liked_mask, 0],
                        latent_np[not_liked_mask, 1],
                        marker='s',
                        color='lightgray',
                        alpha=0.2,
                        s=40,
                        edgecolors='none'
                    )
                else:  # heart
                    for x, y in latent_np[not_liked_mask]:
                        t = ax.text(
                            x, y, shape_to_marker[shape],
                            fontsize=12,
                            color='lightgray',
                            ha='center', va='center', alpha=0.2,
                        )
                        t.set_path_effects([
                            path_effects.Stroke(linewidth=0.75, foreground='black'),
                            path_effects.Normal()
                        ])

            # Plot liked (color, high alpha)
            if np.sum(liked_mask) > 0:
                if shape == 'square':
                    ax.scatter(
                        latent_np[liked_mask, 0],
                        latent_np[liked_mask, 1],
                        marker='s',
                        color=color_to_color[color],
                        edgecolors='k',
                        alpha=0.9,
                        s=60,
                        linewidths=0.5,
                        label=f"{shape}-{color}"
                    )
                else:  # heart
                    for x, y in latent_np[liked_mask]:
                        t = ax.text(
                            x, y, shape_to_marker[shape],
                            fontsize=12,
                            color=color_to_color[color],
                            ha='center', va='center', fontweight='bold'
                        )
                        t.set_path_effects([
                            path_effects.Stroke(linewidth=1.5, foreground='black'),
                            path_effects.Normal()
                        ])

    

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    #ax.set_title(f"Latent Space – User {user_idx} Likes Highlighted")

    x_margin = (latent_np[:, 0].max() - latent_np[:, 0].min()) * 0.05
    y_margin = (latent_np[:, 1].max() - latent_np[:, 1].min()) * 0.05
    ax.set_xlim(latent_np[:, 0].min() - x_margin, latent_np[:, 0].max() + x_margin)
    ax.set_ylim(latent_np[:, 1].min() - y_margin, latent_np[:, 1].max() + y_margin)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from scipy.stats import gaussian_kde

def plot_user_latent_preferences_with_kde(latent_np, shapes, colors, R, user_idx, max_per_group=300, save_path=None):
    assert latent_np.shape[0] == R.shape[0] == len(shapes) == len(colors)

    shape_to_marker = {
        'square': 's',
        'heart': '♥',  # Real heart as text
    }

    color_to_color = {
        'red': 'red',
        'blue': 'blue',
    }

    shapes = np.array(shapes)
    colors = np.array(colors)

    # Precompute masks for subsampling
    selected_indices = []
    for shape in shape_to_marker:
        for color in color_to_color:
            mask = (shapes == shape) & (colors == color)
            indices = np.where(mask)[0]
            if len(indices) > max_per_group:
                indices = np.random.choice(indices, max_per_group, replace=False)
            selected_indices.extend(indices.tolist())

    # Subset everything
    latent_np = latent_np[selected_indices]
    shapes = shapes[selected_indices]
    colors = colors[selected_indices]
    R = R[selected_indices]

    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')



    # Create grid for KDE
    x_min, x_max = latent_np[:, 0].min(), latent_np[:, 0].max()
    y_min, y_max = latent_np[:, 1].min(), latent_np[:, 1].max()
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05
    x_grid = np.linspace(x_min - x_margin, x_max + x_margin, 100)
    y_grid = np.linspace(y_min - y_margin, y_max + y_margin, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = np.vstack([X.ravel(), Y.ravel()])

    # KDE per (shape, color) group
    for shape in shape_to_marker:
        for color in color_to_color:
            liked_mask = (shapes == shape) & (colors == color) & (R[:, user_idx] == 1)
            liked_points = latent_np[liked_mask]
            if len(liked_points) > 5:
                kde = gaussian_kde(liked_points.T)
                Z = kde(grid_points).reshape(X.shape)
                # Only show top density regions (avoid low-density painting)
                threshold = Z.max() * 0.05  # Show top 95% density
                levels = np.linspace(threshold, Z.max(), 10)

                ax.contourf(X, Y, Z, levels=levels, colors=[color_to_color[color]], alpha=0.2*len(liked_points)/150)
            

    # Plot the actual points
    for shape in shape_to_marker:
        for color in color_to_color:
            mask = (shapes == shape) & (colors == color)
            liked_mask = mask & (R[:, user_idx] == 1)
            not_liked_mask = mask & (R[:, user_idx] == 0)

            if np.sum(not_liked_mask) > 0:
                if shape == 'square':
                    ax.scatter(latent_np[not_liked_mask, 0], latent_np[not_liked_mask, 1],
                               marker='s', color='lightgray', alpha=0.2, s=40, edgecolors='none')
                else:
                    for x, y in latent_np[not_liked_mask]:
                        t = ax.text(x, y, shape_to_marker[shape], fontsize=12, color='lightgray',
                                    ha='center', va='center', alpha=0.2)
                        t.set_path_effects([
                            path_effects.Stroke(linewidth=0.75, foreground='black'),
                            path_effects.Normal()
                        ])

            if np.sum(liked_mask) > 0:
                if shape == 'square':
                    ax.scatter(latent_np[liked_mask, 0], latent_np[liked_mask, 1],
                               marker='s', color=color_to_color[color], edgecolors='k',
                               alpha=0.9, s=60, linewidths=0.5)
                else:
                    for x, y in latent_np[liked_mask]:
                        t = ax.text(x, y, shape_to_marker[shape], fontsize=12,
                                    color=color_to_color[color], ha='center', va='center',
                                    fontweight='bold')
                        t.set_path_effects([
                            path_effects.Stroke(linewidth=1.5, foreground='black'),
                            path_effects.Normal()
                        ])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    #ax.set_title(f"User {user_idx} - Likes", fontsize=16)

    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()


    
def plot_real_vs_generated_by_preference(X_umap, R, generated_embeddings, user_id, umap_model, color='blue', save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np

    projected_generated = umap_model.transform(generated_embeddings.cpu().numpy())
    
    liked_mask = R[:, user_id] == 1
    disliked_mask = R[:, user_id] == 0

    plt.figure(figsize=(10, 8))

    # Plot disliked items (gray)
    plt.scatter(
        X_umap[disliked_mask, 0], X_umap[disliked_mask, 1],
        alpha=0.15, color='gray', label='Disliked (Real)', s=30
    )

    # Plot liked items (colored)
    plt.scatter(
        X_umap[liked_mask, 0], X_umap[liked_mask, 1],
        color="#1f77b4", edgecolors='k', label='Liked (Real)', s=40, alpha=0.2
    )

    # Plot generated embeddings
    plt.scatter(
        projected_generated[:, 0], projected_generated[:, 1],
        color='gold', edgecolors='black', label='Generated', s=60, alpha=0.5
    )

    plt.title(f"User {user_id} – Real Liked/Disliked vs Generated Embeddings")
    plt.legend()
    plt.axis('off')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()

def plot_generated_images_on_umap(umap_coords, images, ref_coords=None, zoom=0.07, figsize=(12, 10), title=None, save_path=None):
    """
    Plots generated images at specified UMAP coordinates with fixed axis limits.
    
    Args:
        umap_coords (np.ndarray): (N, 2) array of UMAP coordinates for generated images.
        images (np.ndarray): (N, H, W, C) array of decoded images (RGB, values in [0, 1]).
        ref_coords (np.ndarray): (M, 2) array of reference coordinates (e.g. from real data)
                                 used to set the axis limits. If None, use umap_coords.
        zoom (float): Scaling factor for the image thumbnails.
        figsize (tuple): Figure size.
        title (str): Optional title for the plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    
    # Use reference coordinates if provided, otherwise use umap_coords.
    if ref_coords is not None:
        x_min = ref_coords[:, 0].min()
        x_max = ref_coords[:, 0].max()
        y_min = ref_coords[:, 1].min()
        y_max = ref_coords[:, 1].max()
    else:
        x_min = umap_coords[:, 0].min()
        x_max = umap_coords[:, 0].max()
        y_min = umap_coords[:, 1].min()
        y_max = umap_coords[:, 1].max()
    
    # Add 10% padding to the axis limits
    range_x = x_max - x_min if x_max > x_min else 1.0
    range_y = y_max - y_min if y_max > y_min else 1.0
    pad_x = 0.1 * range_x
    pad_y = 0.1 * range_y
    
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    
    if title is not None:
        ax.set_title(title, fontsize=16)
    
    # Plot each image at its UMAP coordinate.
    for (x, y), img in zip(umap_coords, images):
        im = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(im, (x, y), frameon=False, pad=0.0)
        ax.add_artist(ab)

    if save_path is not None:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.stats import gaussian_kde
import matplotlib.patheffects as path_effects

def plot_combined_kde_and_images(latent_np, shapes, colors, R, user_idx, 
                                 generated_coords, decoded_images, 
                                 max_per_group=300, zoom=0.07, figsize=(12, 10), 
                                 save_path=None):
    assert latent_np.shape[0] == R.shape[0] == len(shapes) == len(colors)

    shape_to_marker = {
        'square': 's',
        'heart': '♥',
    }

    color_to_color = {
        'red': 'red',
        'blue': 'blue',
    }

    shapes = np.array(shapes)
    colors = np.array(colors)

    selected_indices = []
    for shape in shape_to_marker:
        for color in color_to_color:
            mask = (shapes == shape) & (colors == color)
            indices = np.where(mask)[0]
            if len(indices) > max_per_group:
                indices = np.random.choice(indices, max_per_group, replace=False)
            selected_indices.extend(indices.tolist())

    latent_np = latent_np[selected_indices]
    shapes = shapes[selected_indices]
    colors = colors[selected_indices]
    R = R[selected_indices]

    fig, ax = plt.subplots(figsize=figsize, facecolor='white')

    x_min, x_max = latent_np[:, 0].min(), latent_np[:, 0].max()
    y_min, y_max = latent_np[:, 1].min(), latent_np[:, 1].max()
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05
    x_grid = np.linspace(x_min - x_margin, x_max + x_margin, 100)
    y_grid = np.linspace(y_min - y_margin, y_max + y_margin, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = np.vstack([X.ravel(), Y.ravel()])

    for shape in shape_to_marker:
        for color in color_to_color:
            liked_mask = (shapes == shape) & (colors == color) & (R[:, user_idx] == 1)
            liked_points = latent_np[liked_mask]
            if len(liked_points) > 5:
                kde = gaussian_kde(liked_points.T)
                Z = kde(grid_points).reshape(X.shape)
                threshold = Z.max() * 0.05
                levels = np.linspace(threshold, Z.max(), 10)
                ax.contourf(X, Y, Z, levels=levels, colors=[color_to_color[color]], alpha=0.2 * len(liked_points) / 150)

    for (x, y), img in zip(generated_coords, decoded_images):
        im = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(im, (x, y), frameon=False, pad=0.0)
        ax.add_artist(ab)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    ax.set_title(f"User {user_idx} - Generated Images", fontsize=16)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()
