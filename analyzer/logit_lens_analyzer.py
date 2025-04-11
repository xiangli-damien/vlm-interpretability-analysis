# -*- coding: utf-8 -*-
"""
Logit Lens analysis implementation (Refactored).

This analyzer now takes pre-computed layer probabilities (projected hidden states)
and extracts probabilities for tracked concept tokens, structuring the output
based on a provided feature mapping (base, patch, newline).
"""

import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Tuple, Union
from torch import nn # Keep nn import for type hinting if needed, though lm_head is no longer passed directly

class LogitLensAnalyzer:
    """
    Analyzer for applying the logit lens technique (Refactored).

    Takes pre-computed vocabulary probabilities for each layer, extracts
    probabilities for tracked concept tokens, and organizes the results
    based on a feature mapping (e.g., distinguishing base, patch, newline features).
    """

    def __init__(self):
        """
        Initialize the LogitLensAnalyzer.
        """
        print("Initialized LogitLensAnalyzer (Refactored).")

    def _validate_concepts(
        self,
        concepts_to_track: Dict[str, List[int]],
        vocab_size: int
    ) -> Dict[str, List[int]]:
        """
        Helper method to validate concept token IDs against vocabulary size.

        Args:
            concepts_to_track (Dict[str, List[int]]): Original concept dictionary.
            vocab_size (int): The model's vocabulary size.

        Returns:
            Dict[str, List[int]]: A dictionary containing only concepts with valid token IDs.
        """
        valid_concepts = {}
        if vocab_size > 0:
            for concept, ids in concepts_to_track.items():
                # Filter token IDs to be within the valid vocabulary range
                valid_ids = [tid for tid in ids if 0 <= tid < vocab_size]
                if valid_ids:
                    valid_concepts[concept] = valid_ids
                else:
                    print(f"  LogitLensAnalyzer Warning: Concept '{concept}' has no valid token IDs within vocab size {vocab_size}. Skipping.")
        else:
            # If vocab_size is unknown or invalid, skip validation
            print("  LogitLensAnalyzer Warning: Invalid vocab_size provided. Skipping concept token ID validation.")
            valid_concepts = concepts_to_track
        return valid_concepts

    def analyze(
        self,
        layer_probabilities: Dict[int, torch.Tensor], # Input: Dict layer_idx -> probability tensor [Batch, SeqLen, Vocab]
        feature_mapping: Dict[str, Any], # Input: Output from engine.create_feature_mapping
        vocab_size: int, # Input: Model's vocabulary size (for validation)
        concepts_to_track: Dict[str, List[int]] # Input: Dict: concept -> token_ids
        # No hidden_states or lm_head needed here anymore
        # cpu_offload parameter removed, assumes probabilities are already on desired device (typically CPU) by the workflow
    ) -> Dict[int, Dict[str, Any]]:
        """
        Performs the logit lens analysis by extracting concept probabilities from pre-computed layer probabilities.

        Args:
            layer_probabilities (Dict[int, torch.Tensor]): Dictionary mapping layer indices to their
                corresponding probability tensors (shape [batch_size, sequence_length, vocab_size]).
                Probabilities should ideally be on CPU for efficient NumPy conversion.
            feature_mapping (Dict[str, Any]): The detailed feature mapping dictionary created by
                the engine, describing base/patch/newline token positions and grids.
            vocab_size (int): The vocabulary size of the language model, used for validation.
            concepts_to_track (Dict[str, List[int]]): Dictionary mapping concept strings
                to lists of their corresponding token IDs.

        Returns:
            Dict[int, Dict[str, Any]]: Nested dictionary:
                {layer_idx: {
                    'base_feature': {concept_str: np.array[grid_h, grid_w]},
                    'patch_feature': {concept_str: np.array[grid_h, grid_w]},
                    'newline_feature': {concept_str: {row_idx: float}}
                    }
                }.
                Returns empty dict if no valid concepts or probabilities are provided.
        """
        print("Analyzing concept probabilities from pre-computed layer probabilities...")
        # --- Input Checks ---
        if not concepts_to_track:
            print("  LogitLensAnalyzer Error: No concepts_to_track provided.")
            return {}
        if not layer_probabilities:
            print("  LogitLensAnalyzer Error: No layer_probabilities provided.")
            return {}
        if not feature_mapping:
            print("  LogitLensAnalyzer Error: Missing 'feature_mapping'. Cannot structure results.")
            return {}

        # --- Validate concept token IDs ---
        valid_concepts = self._validate_concepts(concepts_to_track, vocab_size)
        if not valid_concepts:
            print("  LogitLensAnalyzer Error: No valid concepts left to track after vocabulary check.")
            return {}
        print(f"  LogitLensAnalyzer: Tracking valid concepts: {list(valid_concepts.keys())}")

        # --- Start Processing ---
        # This dictionary will store the final structured results
        structured_probs_by_layer: Dict[int, Dict[str, Any]] = {}

        # Iterate through the layers provided in the layer_probabilities dictionary
        for layer_idx, probs in tqdm(layer_probabilities.items(), desc="Extracting Concept Probs", ncols=100):

            # Ensure the probability tensor is on the CPU for NumPy operations
            if probs.device != torch.device('cpu'):
                print(f"  LogitLensAnalyzer Warning: Probabilities for layer {layer_idx} are on {probs.device}. Moving to CPU.")
                try:
                    probs = probs.cpu()
                except Exception as e:
                    print(f"  LogitLensAnalyzer Error: Failed to move probabilities for layer {layer_idx} to CPU: {e}. Skipping layer.")
                    continue

            # Check tensor shape (expecting [Batch, SeqLen, VocabSize])
            if probs.ndim != 3 or probs.shape[0] != 1:
                 print(f"  LogitLensAnalyzer Warning: Unexpected probability tensor shape {probs.shape} for layer {layer_idx}. Expecting [1, SeqLen, VocabSize]. Skipping layer.")
                 continue

            # --- Extract probabilities based on feature_mapping ---
            layer_results = {"base_feature": {}, "patch_feature": {}, "newline_feature": {}}
            seq_len = probs.shape[1] # Get sequence length from the probability tensor

            # 1. Base Features Extraction
            base_info = feature_mapping.get("base_feature", {})
            # Check if position mapping and grid dimensions are available
            if base_info.get("positions") and base_info.get("grid"):
                grid_h, grid_w = base_info["grid"]
                # Initialize NumPy arrays to store probabilities for each concept
                base_grids = {concept: np.zeros((grid_h, grid_w), dtype=np.float32) for concept in valid_concepts}
                # Iterate through token indices and their corresponding grid positions
                for token_idx, (r, c) in base_info["positions"].items():
                    # Ensure token index is valid and grid indices are within bounds
                    if 0 <= token_idx < seq_len and 0 <= r < grid_h and 0 <= c < grid_w:
                        # For each concept, find the max probability among its associated token IDs
                        for concept, token_ids in valid_concepts.items():
                            # Get probabilities for the concept's token IDs at this sequence position
                            concept_probs_tensor = probs[0, token_idx, token_ids]
                            # Store the maximum probability in the corresponding grid cell
                            base_grids[concept][r, c] = torch.max(concept_probs_tensor).item()
                    # else: print(f"  Debug: Base feature index {token_idx} or pos {(r,c)} out of bounds.") # Optional debug
                layer_results["base_feature"] = base_grids

            # 2. Patch Features Extraction
            patch_info = feature_mapping.get("patch_feature", {})
            # Check if position mapping and unpadded grid dimensions are available
            if patch_info.get("positions") and patch_info.get("grid_unpadded"):
                grid_h, grid_w = patch_info["grid_unpadded"]
                # Initialize NumPy arrays for patch features
                patch_grids = {concept: np.zeros((grid_h, grid_w), dtype=np.float32) for concept in valid_concepts}
                # Iterate through token indices and their corresponding unpadded grid positions
                for token_idx, (r, c) in patch_info["positions"].items():
                    # Ensure indices are valid
                    if 0 <= token_idx < seq_len and 0 <= r < grid_h and 0 <= c < grid_w:
                        # Find max probability for each concept
                        for concept, token_ids in valid_concepts.items():
                            concept_probs_tensor = probs[0, token_idx, token_ids]
                            patch_grids[concept][r, c] = torch.max(concept_probs_tensor).item()
                    # else: print(f"  Debug: Patch feature index {token_idx} or pos {(r,c)} out of bounds.") # Optional debug
                layer_results["patch_feature"] = patch_grids

            # 3. Newline Features Extraction
            newline_info = feature_mapping.get("newline_feature", {})
            # Check if position mapping is available
            if newline_info.get("positions"):
                # Initialize dictionary to store max probability per concept per row index
                newline_dict = {concept: {} for concept in valid_concepts}
                # Iterate through token indices and their corresponding row index (before the newline)
                for token_idx, row_idx in newline_info["positions"].items():
                    # Ensure token index is valid
                    if 0 <= token_idx < seq_len:
                        # Find max probability for each concept
                        for concept, token_ids in valid_concepts.items():
                            concept_probs_tensor = probs[0, token_idx, token_ids]
                            # Store the max probability found for this row index, updating if higher
                            current_max = newline_dict[concept].get(row_idx, 0.0)
                            newline_dict[concept][row_idx] = max(current_max, torch.max(concept_probs_tensor).item())
                    # else: print(f"  Debug: Newline feature index {token_idx} out of bounds.") # Optional debug
                layer_results["newline_feature"] = newline_dict

            # Store the structured results for the current layer
            structured_probs_by_layer[layer_idx] = layer_results

        print("Concept probability extraction complete.")
        return structured_probs_by_layer