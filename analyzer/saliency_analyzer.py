# -*- coding: utf-8 -*-
"""
Saliency analysis utilities for VLM information flow.

Includes functions for:
- Calculating saliency scores from attention weights and gradients (Rollout/GRAD*ATTN style).
- Computing information flow metrics (e.g., Text->Target, Image->Target)
  based on attention or saliency matrices for a single layer.
- Analyzing layer-wise flow trends based on aggregated saliency scores.
"""

import torch
import numpy as np
import gc
from typing import Dict, Any, Optional, List, Tuple, Union
from tqdm import tqdm

# Note: This analyzer typically operates on data captured by hook_utils.GradientAttentionCapture

def compute_flow_metrics_optimized(
    attention_or_saliency_matrix: torch.Tensor,
    text_indices: torch.Tensor,
    image_indices: torch.Tensor,
    target_idx: int,
    normalize: bool = False # Optional normalization based on total incoming flow
) -> Dict[str, float]:
    """
    Computes information flow metrics from different token types (text, image, generated/other)
    towards a specific target token, based on a given attention or saliency matrix.

    Optimized implementation using boolean masks and tensor operations for efficiency.

    Args:
        attention_or_saliency_matrix (torch.Tensor): A 2D tensor [sequence_length, sequence_length]
            representing attention weights or saliency scores. Assumes rows are destinations (queries)
            and columns are sources (keys/values). Should be on CPU or GPU.
        text_indices (torch.Tensor): 1D tensor containing the indices of text tokens within the sequence.
                                     Must be on the same device as the attention/saliency matrix.
        image_indices (torch.Tensor): 1D tensor containing the indices of image tokens within the sequence.
                                      Must be on the same device.
        target_idx (int): The index of the target token (row index in the matrix) towards which
                          flow is being measured.
        normalize (bool): If True, normalizes the sum metrics (Stq_sum, Siq_sum, Sgq_sum) to represent
                          percentages of the total incoming flow to the target token (summing only over
                          valid source tokens *before* the target index, respecting causality).
                          Defaults to False.

    Returns:
        Dict[str, float]: A dictionary containing computed flow metrics:
            - 'Stq_sum': Total flow from Text source tokens -> Target token.
            - 'Stq_mean': Average flow from Text source tokens -> Target token.
            - 'Stq_count': Number of Text source tokens contributing flow.
            - 'Siq_sum': Total flow from Image source tokens -> Target token.
            - 'Siq_mean': Average flow from Image source tokens -> Target token.
            - 'Siq_count': Number of Image source tokens contributing flow.
            - 'Sgq_sum': Total flow from Generated/Other source tokens -> Target token.
            - 'Sgq_mean': Average flow from Generated/Other source tokens -> Target token.
            - 'Sgq_count': Number of Generated/Other source tokens contributing flow.
            - 'Stq_percent', 'Siq_percent', 'Sgq_percent': Normalized flow percentages (only if normalize=True).
            - 'token_counts': Dictionary with counts of {'text', 'image', 'generated', 'total'} tokens in the sequence.
    """
    metrics: Dict[str, float] = {}
    # Validate input matrix dimension
    if attention_or_saliency_matrix.ndim != 2:
        raise ValueError(f"Input matrix must be 2D, but got shape {attention_or_saliency_matrix.shape}")

    S = attention_or_saliency_matrix.shape[0] # Get sequence length
    device = attention_or_saliency_matrix.device # Get device of the matrix

    # Ensure index tensors are on the same device as the matrix for masking operations
    if text_indices.device != device: text_indices = text_indices.to(device)
    if image_indices.device != device: image_indices = image_indices.to(device)

    # --- Create Masks for Source Token Types ---
    # Initialize boolean masks for text, image, and generated/other source tokens
    text_mask_src = torch.zeros(S, dtype=torch.bool, device=device)
    image_mask_src = torch.zeros(S, dtype=torch.bool, device=device)
    # Populate masks based on provided indices
    if len(text_indices) > 0: text_mask_src[text_indices] = True
    if len(image_indices) > 0: image_mask_src[image_indices] = True
    # Generated/Other mask includes all tokens not classified as text or image
    # Note: This implicitly includes special tokens (like CLS, SEP, PAD) if they exist and are not in text/image indices.
    generated_mask_src = ~(text_mask_src | image_mask_src)

    # --- Create Causal Mask ---
    # A token cannot attend to future tokens (source index must be less than target index)
    causal_mask = torch.arange(S, device=device) < target_idx

    # --- Extract Target Row ---
    # Check if the target index is valid for the sequence length
    if not (0 <= target_idx < S):
        print(f"Warning: target_idx {target_idx} is out of bounds for sequence length {S}. Returning zero metrics.")
        # Initialize all metrics to zero if target_idx is invalid
        for prefix in ["Stq", "Siq", "Sgq"]:
            metrics[f"{prefix}_sum"] = 0.0
            metrics[f"{prefix}_mean"] = 0.0
            metrics[f"{prefix}_count"] = 0
            if normalize: metrics[f"{prefix}_percent"] = 0.0
        metrics["token_counts"] = {'text': 0, 'image': 0, 'generated': 0, 'total': S}
        return metrics

    # Get the row corresponding to the target token (representing flow *to* this token)
    target_row = attention_or_saliency_matrix[target_idx, :] # Shape [S]

    # --- Calculate Metrics for Each Source Type ---
    total_flow_sum_causal = 0.0 # Accumulator for total incoming flow (used for normalization)

    # Iterate through the source types (Text, Image, Generated/Other)
    for prefix, source_mask in [("Stq", text_mask_src), ("Siq", image_mask_src), ("Sgq", generated_mask_src)]:
        # Combine the source type mask with the causal mask to get valid source indices
        valid_sources_mask = source_mask & causal_mask
        # Count the number of valid source tokens for this type
        count = valid_sources_mask.sum().item()
        metrics[f"{prefix}_count"] = count

        if count > 0:
            # If there are valid sources, extract the corresponding flow values from the target row
            values = target_row[valid_sources_mask]
            # Calculate sum and mean of the flow values
            flow_sum = values.sum().item()
            flow_mean = values.mean().item()
            metrics[f"{prefix}_sum"] = flow_sum
            metrics[f"{prefix}_mean"] = flow_mean
            # Add the sum to the total causal flow for normalization
            total_flow_sum_causal += flow_sum
        else:
            # If no valid sources, set sum and mean to zero
            metrics[f"{prefix}_sum"] = 0.0
            metrics[f"{prefix}_mean"] = 0.0

    # --- Normalization (Optional) ---
    if normalize:
        # Avoid division by zero if the total causal flow is negligible
        if total_flow_sum_causal > 1e-8:
            for prefix in ["Stq", "Siq", "Sgq"]:
                # Calculate percentage contribution of each source type
                metrics[f"{prefix}_percent"] = (metrics[f"{prefix}_sum"] / total_flow_sum_causal) * 100.0
        else:
            # Assign 0 percent if total flow is zero or very small
            for prefix in ["Stq", "Siq", "Sgq"]:
                metrics[f"{prefix}_percent"] = 0.0

    # --- Store Token Counts ---
    metrics["token_counts"] = {
        "text": text_mask_src.sum().item(),
        "image": image_mask_src.sum().item(),
        "generated": generated_mask_src.sum().item(),
        "total": S
    }

    return metrics


def calculate_saliency_scores(
    attention_weights: Dict[str, torch.Tensor],
    attention_grads: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Calculate attention saliency scores using the GRAD * ATTENTION method.

    Saliency Score S = |A * (dA/dL)|
    where A is the attention weight matrix and dA/dL is the gradient of the loss
    with respect to the attention weights.

    Args:
        attention_weights (Dict[str, torch.Tensor]): Dictionary mapping layer names
            to attention weight tensors (e.g., shape [Batch, Heads, SeqLen, SeqLen]).
            These weights should ideally require gradients if grads were computed based on them.
        attention_grads (Dict[str, torch.Tensor]): Dictionary mapping layer names
            to gradient tensors of the loss w.r.t. attention weights. Must have the
            same shape and device as the corresponding weights. Gradients are typically
            detached outputs from a `backward()` call.

    Returns:
        Dict[str, torch.Tensor]: Dictionary mapping layer names to the computed
            saliency score tensors. Saliency tensors have the same shape as the
            input attention weights and are detached from the computation graph.
    """
    saliency_scores: Dict[str, torch.Tensor] = {}
    print(f"Calculating saliency scores for {len(attention_grads)} layers with gradients...")

    # Iterate through layers that have gradients available
    layers_with_grads = list(attention_grads.keys()) # Use list copy for safe iteration if needed

    for layer_name in layers_with_grads:
        # Check if corresponding attention weights are also available
        if layer_name in attention_weights:
            attn = attention_weights[layer_name]
            grad = attention_grads[layer_name]

            # --- Sanity Checks ---
            # Verify that shapes match
            if attn.shape != grad.shape:
                print(f"  Saliency Calc Warning: Shape mismatch for layer '{layer_name}'! Weights: {attn.shape}, Grad: {grad.shape}. Skipping.")
                continue
            # Verify that devices match, attempt to move gradient if necessary
            if attn.device != grad.device:
                print(f"  Saliency Calc Warning: Device mismatch for layer '{layer_name}'! Weights: {attn.device}, Grad: {grad.device}. Attempting to move grad.")
                try:
                    grad = grad.to(attn.device)
                except Exception as e:
                    print(f"    Saliency Calc Error: Failed to move gradient for layer '{layer_name}': {e}. Skipping layer.")
                    continue
            # Optional check: Warn if weights didn't require grad (though grads exist)
            # if not attn.requires_grad:
            #     print(f"  Saliency Calc Warning: Attention weights for layer '{layer_name}' do not require grad, yet gradients exist.")

            # --- Compute Saliency ---
            try:
                # Calculate S = |A * grad(A)|
                # Use float32 for the multiplication for potentially better precision, then take absolute value
                saliency = torch.abs(attn.float() * grad.float())
                # Store the detached saliency score tensor
                saliency_scores[layer_name] = saliency.detach()
            except Exception as e:
                print(f"  Saliency Calc Error: Failed computing saliency for layer '{layer_name}': {e}. Skipping.")

        else:
            # Warn if gradients exist but weights are missing
            print(f"  Saliency Calc Warning: Gradient found for layer '{layer_name}', but no corresponding attention weights were provided. Cannot compute saliency.")

    print(f"Calculated saliency scores for {len(saliency_scores)} layers.")
    return saliency_scores


def analyze_layerwise_saliency_flow(
    saliency_scores: Dict[str, torch.Tensor],
    text_indices: torch.Tensor,
    image_indices: torch.Tensor,
    target_token_idx: int,
    cpu_offload: bool = True
) -> Dict[int, Dict[str, float]]:
    """
    Analyzes layer-wise information flow based on computed saliency scores.

    For each layer's saliency score tensor, it averages over batch and head dimensions
    (if present) to get a 2D [SeqLen, SeqLen] matrix. Then, it computes flow metrics
    (Text->Target, Image->Target, Generated->Target) for that layer using
    `compute_flow_metrics_optimized`.

    Args:
        saliency_scores (Dict[str, torch.Tensor]): Dictionary mapping layer names
            to saliency tensors (output of `calculate_saliency_scores`).
            Expected tensor shapes are [Batch, Heads, SeqLen, SeqLen], [Batch, SeqLen, SeqLen],
            [Heads, SeqLen, SeqLen], or [SeqLen, SeqLen].
        text_indices (torch.Tensor): 1D tensor of indices for text tokens.
        image_indices (torch.Tensor): 1D tensor of indices for image tokens.
        target_token_idx (int): The index of the target token for flow analysis (row index).
        cpu_offload (bool): If True, attempts to move the averaged 2D saliency matrix to CPU
                            before computing metrics to conserve GPU memory. Defaults to True.

    Returns:
        Dict[int, Dict[str, float]]: A dictionary mapping the numerical layer index (extracted from the name)
                                     to the computed flow metrics dictionary for that layer. Returns empty if
                                     input is empty or errors occur.
    """
    # Dictionary to store the final layer-wise metrics
    layer_flow_metrics: Dict[int, Dict[str, float]] = {}
    print(f"\nAnalyzing layer-wise flow based on saliency scores towards target token index: {target_token_idx}")

    if not saliency_scores:
        print("Warning: No saliency scores provided for layer-wise flow analysis.")
        return {}

    # Helper function to extract the numerical layer index from a module name string
    def get_layer_num_from_name(name: str) -> int:
        parts = name.split('.')
        for p in parts:
            if p.isdigit():
                return int(p)
        # Return -1 or raise error if no number found, indicating potential naming issue
        return -1

    # Sort layer names based on their extracted numerical index for ordered processing
    # Handles cases where keys might not contain numbers gracefully (they get sorted based on string value)
    sorted_layer_names = sorted(saliency_scores.keys(), key=get_layer_num_from_name)

    # Determine the target device for index tensors based on cpu_offload flag
    # If offloading, indices should be on CPU. Otherwise, match the device of the saliency tensors.
    try:
        target_device = torch.device('cpu') if cpu_offload else next(iter(saliency_scores.values())).device
    except StopIteration:
        print("Warning: Saliency scores dictionary is empty. Cannot determine target device.")
        return {} # Cannot proceed without data

    print(f"  Index tensors (text, image) will be moved to target device: {target_device}")
    try:
        # Move index tensors to the target device if they are not already there
        if text_indices.device != target_device: text_indices = text_indices.to(target_device)
        if image_indices.device != target_device: image_indices = image_indices.to(target_device)
    except Exception as e:
        print(f"  Error moving index tensors to {target_device}: {e}. Analysis might fail or be incorrect.")
        # Continue analysis but results might be wrong if devices don't match later


    # Iterate through the sorted layer names
    for layer_name in tqdm(sorted_layer_names, desc="Analyzing Layer Saliency Flow", ncols=100):
        saliency_tensor = saliency_scores[layer_name]
        # Extract the numerical layer index
        layer_num = get_layer_num_from_name(layer_name)
        if layer_num == -1:
            print(f"Warning: Could not determine numerical index for layer '{layer_name}'. Skipping flow analysis for this layer.")
            continue

        # --- Prepare 2D Saliency Matrix ---
        # Average over batch and/or head dimensions to get a [SeqLen, SeqLen] matrix
        saliency_matrix_2d: Optional[torch.Tensor] = None
        if saliency_tensor.ndim == 4: # Assume [Batch, Heads, SeqLen, SeqLen]
            saliency_matrix_2d = saliency_tensor.mean(dim=(0, 1)).float() # Average over batch and heads
        elif saliency_tensor.ndim == 3: # Assume [Batch/Heads, SeqLen, SeqLen]
            print(f"Warning: Saliency tensor for layer {layer_num} has 3 dims ({saliency_tensor.shape}). Averaging over dim 0.")
            saliency_matrix_2d = saliency_tensor.mean(dim=0).float()
        elif saliency_tensor.ndim == 2: # Assume already [SeqLen, SeqLen]
            saliency_matrix_2d = saliency_tensor.float()
        else:
            # Skip if tensor shape is unexpected
            print(f"Warning: Unexpected saliency tensor shape {saliency_tensor.shape} for layer '{layer_name}' (num={layer_num}). Skipping.")
            continue

        # --- Offload Averaged Matrix if Requested ---
        matrix_device = saliency_matrix_2d.device
        if cpu_offload and matrix_device != torch.device('cpu'):
            try:
                saliency_matrix_2d = saliency_matrix_2d.cpu()
                matrix_device = torch.device('cpu') # Update device info
            except Exception as e:
                print(f"Warning: Failed to move averaged saliency matrix for layer {layer_num} to CPU: {e}. Computing metrics on {matrix_device}.")
                # If offload fails, ensure index tensors are on the matrix's device
                if text_indices.device != matrix_device: text_indices = text_indices.to(matrix_device)
                if image_indices.device != matrix_device: image_indices = image_indices.to(matrix_device)
        # Ensure index tensors match matrix device if *not* offloading, or if offload failed
        elif matrix_device != target_device: # Check if matrix device matches intended target (CPU or original GPU)
             if text_indices.device != matrix_device or image_indices.device != matrix_device:
                 print(f"  Moving index tensors to matrix device {matrix_device} for layer {layer_num}")
                 try:
                     if text_indices.device != matrix_device: text_indices = text_indices.to(matrix_device)
                     if image_indices.device != matrix_device: image_indices = image_indices.to(matrix_device)
                     target_device = matrix_device # Update target device for subsequent layers if mismatch occurred
                 except Exception as e:
                     print(f"  Error moving index tensors: {e}. Skipping layer {layer_num}.")
                     del saliency_matrix_2d # Clean up matrix
                     continue

        # --- Compute Flow Metrics for the Layer ---
        try:
            # Call the optimized metric computation function
            metrics = compute_flow_metrics_optimized(
                saliency_matrix_2d,
                text_indices,
                image_indices,
                target_token_idx,
                normalize=True # Calculate percentages (Stq_percent etc.)
            )
            # Store the computed metrics for this layer index
            layer_flow_metrics[layer_num] = metrics
        except Exception as e:
            print(f"Error computing flow metrics for layer {layer_num} ('{layer_name}'): {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging

        # --- Cleanup ---
        # Explicitly delete the potentially large 2D matrix to free memory
        del saliency_matrix_2d
        # Optional: More aggressive garbage collection and cache clearing
        if layer_num % 10 == 0: # Adjust frequency as needed
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    print(f"Saliency flow analysis complete for {len(layer_flow_metrics)} layers.")
    return layer_flow_metrics