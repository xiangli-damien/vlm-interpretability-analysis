# -*- coding: utf-8 -*-
"""
Generic functions for extracting internal representations from models,
such as hidden states and logits/probabilities projected from intermediate layers.
"""

import torch
from torch import nn
from tqdm import tqdm
from typing import Tuple, List, Optional, Dict, Any, Union
import gc # Import GC for potential memory management

def get_hidden_states_from_forward_pass(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    layers_to_extract: Optional[List[int]] = None
) -> Tuple[torch.Tensor, ...]:
    """
    Performs a forward pass and extracts hidden states from specified layers.

    Args:
        model (nn.Module): The model to run the forward pass on. Assumed to support `output_hidden_states=True`.
        inputs (Dict[str, torch.Tensor]): The input data prepared for the model (e.g., input_ids, pixel_values).
        layers_to_extract (Optional[List[int]]): A list of layer indices (0-based) to extract
                                                  from the 'hidden_states' tuple returned by the model.
                                                  Layer 0 typically corresponds to the initial embeddings.
                                                  If None, extracts all hidden states returned by the model.

    Returns:
        Tuple[torch.Tensor, ...]: A tuple containing the hidden state tensors for the requested layers.
                                  Tensors will be on the device the model executed on.

    Raises:
        ValueError: If the model does not return 'hidden_states' in its output.
        IndexError: If a requested layer index in `layers_to_extract` is out of bounds.
    """
    model.eval() # Ensure model is in evaluation mode for consistent outputs
    print(f"Performing forward pass to extract hidden states...")

    # Disable gradient calculations for efficiency as we only need activations
    with torch.no_grad():
        # Run the model's forward pass, requesting hidden states
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    # Retrieve the hidden states from the model output object
    all_hidden_states = outputs.get("hidden_states")

    # Validate that hidden states were returned
    if all_hidden_states is None:
        raise ValueError("Model did not return 'hidden_states'. Ensure the model supports this output and 'output_hidden_states=True' was passed.")

    print(f"  Model returned {len(all_hidden_states)} hidden state tensors (including embeddings).")

    # Return all states if no specific layers were requested
    if layers_to_extract is None:
        print(f"  Extracting all {len(all_hidden_states)} hidden states.")
        return all_hidden_states
    # Otherwise, extract only the requested layers
    else:
        extracted_states = []
        max_layer_idx = len(all_hidden_states) - 1
        print(f"  Extracting hidden states for specified layers: {layers_to_extract}")
        for layer_idx in layers_to_extract:
            # Validate the requested layer index
            if 0 <= layer_idx <= max_layer_idx:
                extracted_states.append(all_hidden_states[layer_idx])
                print(f"    Extracted layer {layer_idx} (Shape: {all_hidden_states[layer_idx].shape})")
            else:
                # Raise error if index is out of bounds
                raise IndexError(f"Requested layer index {layer_idx} is out of bounds (0 to {max_layer_idx}).")
        # Return the selected hidden states as a tuple
        return tuple(extracted_states)


def get_logits_from_hidden_states(
    hidden_states: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
    lm_head: nn.Module,
    layers_to_process: Optional[List[int]] = None,
    cpu_offload: bool = True,
    use_float32_for_softmax: bool = True,
) -> Dict[int, torch.Tensor]:
    """
    Projects hidden states from specified layers through the language model head (lm_head)
    to compute vocabulary probabilities (via softmax) for each layer.

    Args:
        hidden_states (Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]):
            A tuple or list of hidden state tensors (e.g., output from `get_hidden_states_from_forward_pass`).
            Assumes hidden_states[0] corresponds to layer 0 (embeddings), hidden_states[i] to layer i output.
            Each tensor typically has shape [batch_size, sequence_length, hidden_size].
        lm_head (nn.Module): The language model head module (usually a Linear layer mapping hidden_size to vocab_size).
        layers_to_process (Optional[List[int]]): A list of layer indices (corresponding to the indices in `hidden_states`)
                                                 for which to compute probabilities. If None, processes all available layers.
        cpu_offload (bool): If True, moves the resulting probability tensor to CPU after computation for each layer
                            to conserve GPU memory. Defaults to True.
        use_float32_for_softmax (bool): If True, casts logits to float32 before applying softmax for potentially
                                        better numerical stability, especially if the model uses lower precision (e.g., float16).
                                        Defaults to True. Note: This function returns *probabilities*, not raw logits.

    Returns:
        Dict[int, torch.Tensor]: A dictionary mapping each processed layer index (int) to its computed probability tensor.
                                 The tensor shape is typically [batch_size, sequence_length, vocab_size].
                                 Tensors will be on CPU if `cpu_offload` is True, otherwise on the device where
                                 softmax was computed (likely the `lm_head` device).
    """
    print("Computing probabilities from hidden states using LM head...")
    # Dictionary to store the results: layer_index -> probability_tensor
    layer_probabilities: Dict[int, torch.Tensor] = {}
    num_hidden_layers_available = len(hidden_states)

    # Determine which layers to process
    if layers_to_process is None:
        # Process all layers if none are specified
        layers_to_process_indices = list(range(num_hidden_layers_available))
    else:
        # Validate and filter the requested layer indices
        layers_to_process_indices = [l for l in layers_to_process if 0 <= l < num_hidden_layers_available]
        if not layers_to_process_indices:
             print("Warning: No valid layers specified in layers_to_process after validation. Returning empty dictionary.")
             return {}

    print(f"  Processing layers: {layers_to_process_indices}")

    # Determine the device where the LM head computations should occur
    try:
        # Assumes lm_head has parameters (typical case for nn.Linear)
        lm_head_device = next(lm_head.parameters()).device
    except StopIteration:
        # Fallback if lm_head has no parameters (e.g., functional, or only buffers)
        print("Warning: Could not determine LM head device from parameters. Checking buffers...")
        try:
            lm_head_device = next(lm_head.buffers()).device
        except StopIteration:
            # Further fallback if no parameters or buffers are found
            print("Warning: LM head has no parameters or buffers. Defaulting device to CPU.")
            lm_head_device = torch.device('cpu')
    except Exception as e:
        print(f"Warning: Error determining LM head device: {e}. Defaulting device to CPU.")
        lm_head_device = torch.device('cpu')

    print(f"  LM head computations will run on device: {lm_head_device}")

    # Ensure lm_head is in evaluation mode (disables dropout, etc.)
    lm_head.eval()

    # Disable gradient calculations during projection and softmax
    with torch.no_grad():
        # Iterate through the specified layer indices
        for layer_idx in tqdm(layers_to_process_indices, desc="Projecting Layers"):
            # Get the hidden state tensor for the current layer
            layer_hidden = hidden_states[layer_idx]
            original_hidden_device = layer_hidden.device # Remember the original device

            # --- Prepare hidden state for projection ---
            # Move hidden state tensor to the LM head's device if they differ
            layer_hidden_on_correct_device = layer_hidden
            if layer_hidden.device != lm_head_device:
                try:
                    layer_hidden_on_correct_device = layer_hidden.to(lm_head_device)
                    # print(f"  Moved layer {layer_idx} hidden state from {original_hidden_device} to {lm_head_device}") # Debug
                except Exception as e:
                    print(f"Warning: Failed to move hidden state for layer {layer_idx} to device {lm_head_device}. Skipping layer. Error: {e}")
                    continue # Skip to the next layer if moving fails

            # --- Projection and Softmax ---
            try:
                # Project hidden state to vocabulary logits using the LM head
                # Optional: Cast hidden state dtype if needed, e.g., lm_head expects float32
                # logits = lm_head(layer_hidden_on_correct_device.to(lm_head.dtype))
                logits = lm_head(layer_hidden_on_correct_device)

                # Compute probabilities using softmax
                if use_float32_for_softmax and logits.dtype != torch.float32:
                    # Cast logits to float32 before softmax for numerical stability
                    probs = torch.softmax(logits.float(), dim=-1)
                else:
                    # Compute softmax in the original logits dtype
                    probs = torch.softmax(logits, dim=-1)

            except Exception as e:
                print(f"Warning: Error during LM head projection or softmax for layer {layer_idx}. Skipping layer. Error: {e}")
                # Clean up potentially moved tensor if projection failed
                if layer_hidden_on_correct_device.device != original_hidden_device:
                    del layer_hidden_on_correct_device
                continue # Skip to next layer

            # --- Offload and Store Results ---
            if cpu_offload:
                # Move the computed probabilities to CPU if requested
                if probs.device != torch.device('cpu'):
                    try:
                        probs = probs.cpu()
                    except Exception as e:
                        print(f"Warning: Failed to move probabilities for layer {layer_idx} to CPU. Error: {e}")
                        # Keep tensor on its current device if move fails

            # Store the final probability tensor (on CPU or original device)
            layer_probabilities[layer_idx] = probs

            # --- Memory Management ---
            # Explicitly delete intermediate tensors that are no longer needed, especially on GPU
            # Delete the logits tensor created on the computation device
            if 'logits' in locals() and logits.device == lm_head_device:
                 del logits
            # Delete the hidden state tensor copy if it was moved
            if layer_hidden_on_correct_device.device != original_hidden_device:
                 del layer_hidden_on_correct_device
            # Delete the probs tensor from the computation device *if* it was successfully offloaded to CPU
            # (otherwise it's already stored in layer_probabilities on the compute device)
            if cpu_offload and layer_probabilities[layer_idx].device == torch.device('cpu') and probs.device == lm_head_device:
                 del probs # Delete the GPU copy if CPU offload succeeded

            # Periodically clear CUDA cache if computations are happening on GPU
            # This can help free up fragmented memory but has a small overhead
            if lm_head_device.type == 'cuda' and layer_idx % 5 == 0: # Adjust frequency as needed
                gc.collect()
                torch.cuda.empty_cache()

    print(f"Probability computation complete for {len(layer_probabilities)} layers.")
    return layer_probabilities