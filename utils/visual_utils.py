# -*- coding: utf-8 -*-
"""
Visualization utilities for VLM analysis results.

Includes functions for plotting:
- Information flow metrics across layers.
- Attention weight heatmaps.
- Processed image tensors fed into the vision encoder.
- Logit lens token probability heatmaps and overlays.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from PIL import Image, ImageDraw # Added ImageDraw for potential future enhancements
from tqdm import tqdm # Added tqdm for progress bars in visualize_token_probabilities
from typing import Dict, Any, Optional, List, Tuple, Union
import torch

# Try importing skimage for potentially better heatmap resizing, with fallback
try:
    from skimage.transform import resize as skimage_resize
    HAS_SKIMAGE = True
except (ImportError, ModuleNotFoundError):
    HAS_SKIMAGE = False
    print("Warning: scikit-image not found. Falling back to simpler numpy-based resizing for heatmap visualization.")


def visualize_information_flow(
    metrics: Dict[int, Dict[str, float]],
    title: str = "VLM Information Flow Analysis",
    save_path: Optional[str] = None
):
    """
    Visualizes information flow metrics (mean and sum) across model layers.

    Creates line plots showing how attention flows from text, image, and
    previously generated tokens towards the current target token at each layer.

    Args:
        metrics (Dict[int, Dict[str, float]]): A dictionary where keys are layer indices (int)
            and values are dictionaries containing flow metrics (e.g., 'Siq_mean',
            'Stq_sum', 'Sgq_mean'). Assumes metrics 'Siq', 'Stq', 'Sgq'.
        title (str): The main title for the combined plot figure.
        save_path (Optional[str]): If provided, the path where the plot image will be saved.
    """
    if not metrics:
        print("Warning: No metrics data provided to visualize_information_flow.")
        return

    # Define consistent colors and markers for different flow types
    flow_styles = {
        "Siq_mean": {"color": "#FF4500", "marker": "o", "label": "Image→Target (Mean)"},  # Orange-Red
        "Stq_mean": {"color": "#1E90FF", "marker": "^", "label": "Text→Target (Mean)"},    # Dodger Blue
        "Sgq_mean": {"color": "#32CD32", "marker": "s", "label": "Generated→Target (Mean)"}, # Lime Green (Sgq replaces Soq)
        "Siq_sum": {"color": "#FF4500", "marker": "o", "label": "Image→Target (Sum)", "linestyle": '--'}, # Dashed for Sum
        "Stq_sum": {"color": "#1E90FF", "marker": "^", "label": "Text→Target (Sum)", "linestyle": '--'}, # Dashed for Sum
        "Sgq_sum": {"color": "#32CD32", "marker": "s", "label": "Generated→Target (Sum)", "linestyle": '--'}  # Dashed for Sum (Sgq replaces Soq)
    }

    # Extract layer indices and available metric keys
    layers = sorted(metrics.keys())
    if not layers:
        print("Warning: Metrics dictionary is empty or contains no valid layer indices.")
        return

    available_metric_keys = set()
    for layer_idx in layers:
        if isinstance(metrics[layer_idx], dict):
             available_metric_keys.update(metrics[layer_idx].keys())

    # Collect data for plotting, handling missing metrics gracefully
    plot_data: Dict[str, List[Optional[float]]] = {key: [] for key in flow_styles if key in available_metric_keys}
    for layer in layers:
         layer_metrics = metrics.get(layer, {})
         for metric_key in plot_data.keys():
              plot_data[metric_key].append(layer_metrics.get(metric_key)) # Append value or None

    # Create figure with two subplots (Mean and Sum)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharex=True) # Share x-axis

    # --- Plot 1: Mean Information Flow ---
    ax1.set_title("Mean Information Flow per Layer")
    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("Mean Attention / Saliency")
    mean_keys = ["Siq_mean", "Stq_mean", "Sgq_mean"]
    for key in mean_keys:
        if key in plot_data:
            style = flow_styles[key]
            valid_layers = [l for l, v in zip(layers, plot_data[key]) if v is not None]
            valid_values = [v for v in plot_data[key] if v is not None]
            if valid_layers:
                 ax1.plot(valid_layers, valid_values, marker=style["marker"], color=style["color"], label=style["label"], linewidth=2)
    ax1.legend(loc="best")
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- Plot 2: Sum Information Flow ---
    ax2.set_title("Total Information Flow per Layer")
    ax2.set_xlabel("Layer Index")
    ax2.set_ylabel("Summed Attention / Saliency")
    sum_keys = ["Siq_sum", "Stq_sum", "Sgq_sum"]
    for key in sum_keys:
         if key in plot_data:
             style = flow_styles[key]
             valid_layers = [l for l, v in zip(layers, plot_data[key]) if v is not None]
             valid_values = [v for v in plot_data[key] if v is not None]
             if valid_layers:
                 ax2.plot(valid_layers, valid_values, marker=style["marker"], color=style["color"], label=style["label"], linestyle=style.get("linestyle", '-'), linewidth=2)
    ax2.legend(loc="best")
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Add overall figure title
    fig.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    # Save if requested
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Information flow visualization saved to: {save_path}")
        except Exception as e:
            print(f"Error saving information flow plot to {save_path}: {e}")

    plt.show()
    plt.close(fig)


def visualize_attention_heatmap(
    attention_matrix: Union[np.ndarray, torch.Tensor],
    tokens: Optional[List[str]] = None,
    title: str = "Attention Heatmap",
    save_path: Optional[str] = None,
    colormap: str = "viridis",
    max_tokens_display: int = 60
):
    """
    Creates a heatmap visualization of an attention matrix using matplotlib.

    Args:
        attention_matrix (Union[np.ndarray, torch.Tensor]): 2D array/tensor of attention weights
            (Sequence Length x Sequence Length). Assumes weights are from destination (rows)
            attending to source (columns).
        tokens (Optional[List[str]]): List of token strings corresponding to the sequence length.
        title (str): Title for the heatmap plot.
        save_path (Optional[str]): Path to save the generated heatmap image. If None, not saved.
        colormap (str): Matplotlib colormap name.
        max_tokens_display (int): Maximum number of token labels to display on each axis.
    """
    if isinstance(attention_matrix, torch.Tensor):
        attention_data = attention_matrix.detach().cpu().numpy()
    elif isinstance(attention_matrix, np.ndarray):
        attention_data = attention_matrix
    else:
        raise TypeError("attention_matrix must be a NumPy array or PyTorch tensor.")

    if attention_data.ndim != 2:
        raise ValueError(f"attention_matrix must be 2D, but got shape {attention_data.shape}")

    seq_len_dst, seq_len_src = attention_data.shape
    if seq_len_dst != seq_len_src:
        print(f"Warning: Attention matrix shape {attention_data.shape} is not square.")

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(attention_data, cmap=colormap, aspect='auto', interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Attention Weight")

    if tokens:
        display_tokens = tokens # Assume tokens match dimensions for now
        if len(tokens) != seq_len_dst or len(tokens) != seq_len_src:
             print(f"Warning: Number of tokens ({len(tokens)}) differs from matrix dims ({seq_len_dst}x{seq_len_src}).")
             display_tokens = tokens[:max(seq_len_dst, seq_len_src)]

        num_ticks = min(max_tokens_display, max(seq_len_dst, seq_len_src))
        ticks_src = np.linspace(0, seq_len_src - 1, num_ticks, dtype=int)
        labels_src = [display_tokens[i] if i < len(display_tokens) else '?' for i in ticks_src]
        ticks_dst = np.linspace(0, seq_len_dst - 1, num_ticks, dtype=int)
        labels_dst = [display_tokens[i] if i < len(display_tokens) else '?' for i in ticks_dst]

        ax.set_xticks(ticks_src)
        ax.set_xticklabels(labels_src, rotation=90, fontsize=8)
        ax.set_yticks(ticks_dst)
        ax.set_yticklabels(labels_dst, rotation=0, fontsize=8)

        ax.set_xticks(np.arange(seq_len_src + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(seq_len_dst + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="grey", linestyle='-', linewidth=0.5, alpha=0.3)
        ax.tick_params(which='minor', size=0)
        ax.set_ylabel("Destination Token (Query)", fontsize=10)
        ax.set_xlabel("Source Token (Key/Value)", fontsize=10)
    else:
        ax.set_xlabel("Source Token Index")
        ax.set_ylabel("Destination Token Index")
        ax.grid(True, linestyle=':', alpha=0.4)

    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention heatmap saved to: {save_path}")
        except Exception as e:
            print(f"Error saving attention heatmap to {save_path}: {e}")

    plt.show()
    plt.close(fig)


def visualize_processed_image_input(analysis_data: Dict[str, Any], save_dir: Optional[str] = None):
    """
    Visualizes the actual processed image tensor(s) fed into the vision encoder.

    Handles both standard single image inputs ([B, C, H, W]) and tiled inputs
    used in high-resolution processing ([B, N, C, H, W]).

    Args:
        analysis_data (Dict[str, Any]): Dictionary containing analysis results,
            expected to have 'inputs_cpu' key which holds the processed tensors
            moved to CPU, including 'pixel_values'.
        save_dir (Optional[str]): Directory to save the visualization(s). If None, not saved.
    """
    print("\nVisualizing processed image tensor(s) input to vision encoder...")
    inputs_cpu = analysis_data.get("inputs_cpu")
    if not inputs_cpu or "pixel_values" not in inputs_cpu:
        print("Error: Missing 'pixel_values' in 'inputs_cpu' for visualization.")
        return

    pixel_values = inputs_cpu["pixel_values"]
    save_paths = []
    if save_dir:
         os.makedirs(save_dir, exist_ok=True)

    try:
        # Case 1: High-resolution Tiling [B, N, C, H, W]
        if pixel_values.ndim == 5:
            batch_idx = 0
            num_tiles, C, H, W = pixel_values.shape[1:]
            print(f"Input tensor shape: {pixel_values.shape}. Visualizing {num_tiles} tiles.")
            cols = int(np.ceil(np.sqrt(num_tiles)))
            rows = int(np.ceil(num_tiles / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), squeeze=False)
            axes_flat = axes.flatten()
            for i in range(num_tiles):
                tile_tensor = pixel_values[batch_idx, i]
                tile_np = tile_tensor.permute(1, 2, 0).float().numpy()
                min_val, max_val = tile_np.min(), tile_np.max()
                tile_np = (tile_np - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(tile_np)
                ax = axes_flat[i]
                ax.imshow(np.clip(tile_np, 0, 1))
                ax.set_title(f"Tile {i+1}", fontsize=9)
                ax.axis("off")
            for i in range(num_tiles, len(axes_flat)): axes_flat[i].axis("off")
            fig.suptitle(f"Processed Image Input Tiles (from {pixel_values.shape} shape)", fontsize=14)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            if save_dir:
                fpath = os.path.join(save_dir, "processed_image_tiles.png")
                plt.savefig(fpath, dpi=150)
                save_paths.append(fpath); print(f"  Saved tiled visualization to: {fpath}")
            plt.show(); plt.close(fig)

        # Case 2: Standard Single Image [B, C, H, W]
        elif pixel_values.ndim == 4:
            batch_idx = 0
            C, H, W = pixel_values.shape[1:]
            print(f"Input tensor shape: {pixel_values.shape}. Visualizing single processed image.")
            img_tensor = pixel_values[batch_idx]
            img_np = img_tensor.permute(1, 2, 0).float().numpy()
            min_val, max_val = img_np.min(), img_np.max()
            img_np = (img_np - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(img_np)
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(np.clip(img_np, 0, 1))
            ax.set_title(f"Processed Image Input Tensor\nShape: {list(img_tensor.shape)}")
            ax.axis("off"); plt.tight_layout()
            if save_dir:
                 fpath = os.path.join(save_dir, "processed_image_single.png")
                 plt.savefig(fpath, dpi=150)
                 save_paths.append(fpath); print(f"  Saved single image visualization to: {fpath}")
            plt.show(); plt.close(fig)
        else:
            print(f"Warning: Unexpected pixel_values shape: {pixel_values.shape}. Cannot visualize.")
    except Exception as e:
        print(f"An error occurred during visualization of processed image: {e}")
        import traceback; traceback.print_exc()
    return save_paths


# <<< Added Function >>>
def visualize_token_probabilities(
    token_probs: Dict[int, Dict[str, Any]],
    input_data: Dict[str, Any],
    selected_layers: Optional[List[int]] = None,
    output_dir: str = "logit_lens_visualization",
    colormap: str = "jet", # Colormap for heatmaps
    heatmap_alpha: float = 0.6 # Alpha for heatmap overlay
):
    """
    Visualize token probability maps from logit lens analysis using heatmaps and line plots.

    Handles visualization for 'base_feature' (grid heatmap), 'patch_feature' (overlay on spatial
    preview image), and 'newline_feature' (line plot).

    Args:
        token_probs (Dict[int, Dict[str, Any]]): Dictionary mapping layer index to probabilities.
            Expected inner structure: {'base_feature': {concept: np.array}, 'patch_feature': ...}.
            Probabilities are typically max probability for tracked concept token(s).
        input_data (Dict[str, Any]): Dictionary from the analyzer's `prepare_inputs`. Must contain
            'feature_mapping', 'original_image', and 'spatial_preview_image'.
        selected_layers (Optional[List[int]]): List of layer indices to visualize. If None, visualizes all available layers.
        output_dir (str): Directory path to save the visualization images. Subdirectories will be created.
        colormap (str): Matplotlib colormap name for heatmaps.
        heatmap_alpha (float): Alpha blending value for heatmap overlays (0.0 to 1.0).

    Returns:
        List[str]: File paths of the saved visualization images.
    """
    print(f"\n--- Generating Logit Lens Probability Visualizations ---")
    print(f"  Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # --- Validate Inputs ---
    if not token_probs:
        print("  Error: No token probabilities data provided. Cannot visualize.")
        return []
    if not input_data or not all(k in input_data for k in ['feature_mapping', 'spatial_preview_image']):
        print("  Error: Missing required 'feature_mapping' or 'spatial_preview_image' in input_data.")
        return []

    feature_mapping = input_data["feature_mapping"]
    spatial_preview_image = input_data["spatial_preview_image"] # PIL Image (resized + padded)

    if not feature_mapping:
        print("  Error: 'feature_mapping' in input_data is empty.")
        return []

    # Use the raw patch size for scaling the heatmap visualization (essential for patch features)
    raw_patch_size = feature_mapping.get("patch_size")
    if not raw_patch_size:
        print("  Error: 'patch_size' not found in feature_mapping.")
        return []

    # Determine layers and concepts to visualize
    available_layers = sorted(token_probs.keys())
    if not available_layers:
        print("  Error: token_probs dictionary contains no layer data.")
        return []

    if selected_layers is None:
        layers_to_plot = available_layers
    else:
        layers_to_plot = [l for l in selected_layers if l in available_layers]
        if not layers_to_plot:
             print(f"  Warning: None of the selected layers {selected_layers} have data in token_probs. Visualizing all available layers instead.")
             layers_to_plot = available_layers

    # Infer concepts from the first available layer's data
    first_layer_data = token_probs[layers_to_plot[0]]
    concepts = []
    if "base_feature" in first_layer_data and isinstance(first_layer_data["base_feature"], dict):
        concepts.extend(list(first_layer_data["base_feature"].keys()))
    if "patch_feature" in first_layer_data and isinstance(first_layer_data["patch_feature"], dict):
        concepts.extend(list(k for k in first_layer_data["patch_feature"].keys() if k not in concepts)) # Add unique concepts
    if "newline_feature" in first_layer_data and isinstance(first_layer_data["newline_feature"], dict):
        concepts.extend(list(k for k in first_layer_data["newline_feature"].keys() if k not in concepts))

    if not concepts:
        print("  Error: No concepts found in the token probability data. Cannot visualize.")
        return []

    print(f"  Visualizing for Layers: {layers_to_plot}")
    print(f"  Visualizing for Concepts: {concepts}")

    # --- Prepare Output Directories ---
    saved_paths = []
    base_dir = os.path.join(output_dir, "base_feature_grids")
    patch_dir = os.path.join(output_dir, "patch_feature_overlays")
    newline_dir = os.path.join(output_dir, "newline_feature_plots")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(patch_dir, exist_ok=True)
    os.makedirs(newline_dir, exist_ok=True)

    # --- 1. Visualize Base Feature Heatmaps (Grids) ---
    print("  Generating base feature grid heatmaps...")
    base_feature_map_info = feature_mapping.get("base_feature", {})
    base_grid = base_feature_map_info.get("grid") # Shape (grid_h, grid_w)

    if base_grid and isinstance(base_grid, tuple) and len(base_grid) == 2:
        base_grid_h, base_grid_w = base_grid
        for concept in concepts:
             # Using tqdm for progress bar per concept
            for layer_idx in tqdm(layers_to_plot, desc=f"Base '{concept}'", leave=False, ncols=100):
                layer_data = token_probs.get(layer_idx, {}).get("base_feature", {})
                base_prob_map = layer_data.get(concept) # Should be np.array

                if base_prob_map is None or not isinstance(base_prob_map, np.ndarray) or base_prob_map.size == 0:
                    # print(f"Debug: Skipping base layer {layer_idx}, concept '{concept}' - No data.")
                    continue
                if base_prob_map.shape != (base_grid_h, base_grid_w):
                     print(f"  Warning: Shape mismatch for base feature layer {layer_idx}, concept '{concept}'. Expected {base_grid}, got {base_prob_map.shape}. Skipping.")
                     continue

                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(base_prob_map, cmap=colormap, interpolation="nearest", vmin=0, vmax=1) # Prob range [0,1]
                ax.set_title(f"Base Feature Grid: '{concept}' - Layer {layer_idx}", fontsize=12)
                ax.set_xticks(np.arange(base_grid_w))
                ax.set_yticks(np.arange(base_grid_h))
                ax.set_xticklabels(np.arange(base_grid_w), fontsize=6)
                ax.set_yticklabels(np.arange(base_grid_h), fontsize=6)
                ax.tick_params(axis='both', which='major', labelsize=6)
                plt.setp(ax.get_xticklabels(), rotation=90) # Rotate x labels if many cols
                ax.grid(which='major', color='grey', linestyle='-', linewidth=0.5, alpha=0.2)

                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(f"Max Probability")

                filepath = os.path.join(base_dir, f"layer_{layer_idx:03d}_{concept}_base_grid.png")
                try: plt.savefig(filepath, dpi=150, bbox_inches="tight"); saved_paths.append(filepath)
                except Exception as e: print(f"    Error saving base grid plot: {e}")
                plt.close(fig)
    else: print("  Skipping base feature visualization: Grid info invalid or missing.")


    # --- 2. Visualize Spatial (Patch) Feature Heatmaps (Overlay) ---
    print("  Generating patch feature overlay heatmaps...")
    patch_feature_map_info = feature_mapping.get("patch_feature", {})
    # This is the grid for visualization (potentially padded)
    patch_vis_grid = patch_feature_map_info.get("grid_for_visualization") # (rows_padded, cols_padded)
    # This is the grid of the actual probability data (unpadded)
    patch_unpadded_grid = patch_feature_map_info.get("grid_unpadded") # (rows_unpadded, cols_unpadded)

    if patch_vis_grid and patch_unpadded_grid and isinstance(patch_vis_grid, tuple) and isinstance(patch_unpadded_grid, tuple):
        vis_grid_h, vis_grid_w = patch_vis_grid
        prob_grid_h, prob_grid_w = patch_unpadded_grid
        # Dimensions of the spatial preview image (the background)
        preview_w, preview_h = spatial_preview_image.size # W, H from PIL

        for concept in concepts:
            for layer_idx in tqdm(layers_to_plot, desc=f"Patch '{concept}'", leave=False, ncols=100):
                layer_data = token_probs.get(layer_idx, {}).get("patch_feature", {})
                patch_prob_map_unpadded = layer_data.get(concept) # np.array

                if patch_prob_map_unpadded is None or not isinstance(patch_prob_map_unpadded, np.ndarray) or patch_prob_map_unpadded.size == 0:
                    # print(f"Debug: Skipping patch layer {layer_idx}, concept '{concept}' - No data.")
                    continue
                if patch_prob_map_unpadded.shape != (prob_grid_h, prob_grid_w):
                    print(f"  Warning: Shape mismatch for patch feature layer {layer_idx}, concept '{concept}'. Expected {patch_unpadded_grid}, got {patch_prob_map_unpadded.shape}. Skipping.")
                    continue

                fig, ax = plt.subplots(figsize=(8, 8 * preview_h / preview_w )) # Keep aspect ratio
                # Show the spatial preview image (resized + padded) as background
                ax.imshow(spatial_preview_image, extent=(0, preview_w, preview_h, 0)) # extent=(left, right, bottom, top)

                # --- Create the heatmap corresponding to the unpadded probability map ---
                # Upscale the probability map to pixel dimensions using the raw patch size
                heatmap_unpadded = np.repeat(np.repeat(patch_prob_map_unpadded, raw_patch_size, axis=0), raw_patch_size, axis=1)
                heatmap_h_unpadded, heatmap_w_unpadded = heatmap_unpadded.shape

                # --- Determine where this unpadded heatmap sits within the padded preview image ---
                # We need the original dimensions *after* resizing but *before* padding
                resized_dims_wh = feature_mapping.get("resized_dimensions") # W, H
                if not resized_dims_wh:
                     print(f"  Warning: Missing 'resized_dimensions' in feature_mapping for layer {layer_idx}. Cannot accurately place heatmap overlay. Skipping.")
                     plt.close(fig)
                     continue
                resized_w_actual, resized_h_actual = resized_dims_wh

                # Calculate padding added (symmetric padding assumed)
                pad_h_total = preview_h - resized_h_actual
                pad_w_total = preview_w - resized_w_actual
                pad_top = pad_h_total // 2
                pad_left = pad_w_total // 2

                # --- Resize the unpadded heatmap to match the resized image dimensions ---
                # This ensures the heatmap aligns correctly with the actual image content before padding
                if HAS_SKIMAGE:
                    resized_heatmap = skimage_resize(heatmap_unpadded, (resized_h_actual, resized_w_actual),
                                                     order=1, mode='constant', cval=0, anti_aliasing=True, preserve_range=True)
                else: # Fallback numpy scaling (nearest neighbor effective)
                    scale_y = resized_h_actual / heatmap_h_unpadded
                    scale_x = resized_w_actual / heatmap_w_unpadded
                    y_indices = (np.arange(resized_h_actual) / scale_y).astype(int)
                    x_indices = (np.arange(resized_w_actual) / scale_x).astype(int)
                    np.clip(y_indices, 0, heatmap_h_unpadded - 1, out=y_indices)
                    np.clip(x_indices, 0, heatmap_w_unpadded - 1, out=x_indices)
                    resized_heatmap = heatmap_unpadded[y_indices[:, None], x_indices]

                # --- Overlay the resized heatmap with correct offset ---
                im = ax.imshow(resized_heatmap, alpha=heatmap_alpha, cmap=colormap, vmin=0, vmax=1, # Prob range [0,1]
                               extent=(pad_left, pad_left + resized_w_actual, # left, right
                                       pad_top + resized_h_actual, pad_top))   # bottom, top (inverted y-axis for imshow)

                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(f"Max Probability")
                ax.set_title(f"Patch Feature Overlay: '{concept}' - Layer {layer_idx}", fontsize=12)
                ax.axis("off")

                filepath = os.path.join(patch_dir, f"layer_{layer_idx:03d}_{concept}_patch_overlay.png")
                try: plt.savefig(filepath, dpi=150, bbox_inches="tight"); saved_paths.append(filepath)
                except Exception as e: print(f"    Error saving patch overlay plot: {e}")
                plt.close(fig)
    else: print("  Skipping patch feature visualization: Grid info invalid or missing.")


    # --- 3. Visualize Newline Token Probabilities (Line Plots) ---
    print("  Generating newline feature line plots...")
    has_newline_data = any(token_probs.get(l, {}).get("newline_feature") for l in layers_to_plot)

    if has_newline_data:
        max_row_overall = 0 # Find max row index across all data for consistent axis limits
        for layer_idx in layers_to_plot:
            newline_layer_data = token_probs.get(layer_idx, {}).get("newline_feature", {})
            for concept_probs in newline_layer_data.values():
                if concept_probs and isinstance(concept_probs, dict): # Check it's a non-empty dict
                    max_row_overall = max(max_row_overall, max(concept_probs.keys()))

        for concept in concepts:
            for layer_idx in tqdm(layers_to_plot, desc=f"Newline '{concept}'", leave=False, ncols=100):
                layer_data = token_probs.get(layer_idx, {}).get("newline_feature", {})
                newline_probs_concept = layer_data.get(concept) # {row_idx: prob}

                if newline_probs_concept is None or not isinstance(newline_probs_concept, dict) or not newline_probs_concept:
                    # print(f"Debug: Skipping newline layer {layer_idx}, concept '{concept}' - No data.")
                    continue

                rows = sorted(newline_probs_concept.keys())
                probs = [newline_probs_concept[r] for r in rows]

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(rows, probs, marker='o', linestyle='-', color='green')
                ax.set_xlabel("Row Index (Spatial Grid)")
                ax.set_ylabel(f"Max Probability")
                ax.set_title(f"Newline Feature Prob: '{concept}' - Layer {layer_idx}", fontsize=12)
                ax.set_ylim(-0.05, 1.05) # Probability range [0, 1] with padding
                ax.set_xlim(-0.5, max(max_row_overall, max(rows) if rows else 0) + 0.5) # Consistent x-axis
                ax.set_xticks(np.arange(0, max(max_row_overall, max(rows) if rows else 0) + 1)) # Integer ticks for rows
                ax.grid(True, linestyle=':', alpha=0.6)

                # Add value labels to points
                for r, p in zip(rows, probs):
                    ax.text(r, p + 0.03, f"{p:.3f}", ha="center", fontsize=8)

                filepath = os.path.join(newline_dir, f"layer_{layer_idx:03d}_{concept}_newline_plot.png")
                try: plt.savefig(filepath, dpi=150, bbox_inches="tight"); saved_paths.append(filepath)
                except Exception as e: print(f"    Error saving newline plot: {e}")
                plt.close(fig)
    else: print("  Skipping newline feature visualization: No newline data found.")

    print(f"--- Logit Lens Visualizations Generated. Total files: {len(saved_paths)} ---")
    return saved_paths