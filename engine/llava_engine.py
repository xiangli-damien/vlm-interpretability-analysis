# -*- coding: utf-8 -*-
"""
Encapsulation LLaVANextEngine for interacting with LLaVA-Next models.

Refactored to improve modularity and reusability:
- Image processing utilities moved to 'utils.llava_next_image_utils'.
- Step-wise analysis method generalized using a HookManager interface.
"""
import torch
from torch import nn
import math
import numpy as np
from PIL import Image
import time
import gc
from typing import Dict, Any, Optional, Union, List, Tuple, Callable, Set

# --- Refactoring Imports ---
# Import image processing utilities from the dedicated module
try:
    from utils.llava_next_image_utils import (
        calculate_resized_dimensions,
        resize_image_for_patching,
        pad_image_for_patching,
        compute_llava_spatial_preview # Use the utility version
    )
except ImportError as e:
    print(f"Warning: Could not import LLaVA image utils: {e}. Feature mapping/preview might fail.")
    # Define dummy functions if import fails to avoid crashing later calls
    def calculate_resized_dimensions(*args, **kwargs): return (0,0)
    def resize_image_for_patching(*args, **kwargs): return np.array([])
    def pad_image_for_patching(*args, **kwargs): return np.array([])
    def compute_llava_spatial_preview(*args, **kwargs): return Image.new('RGB', (1,1))

# Import utilities for model loading, token processing, and hooking
try:
    from utils.data_utils import load_image, build_conversation, get_image_token_spans, get_token_indices
    from utils.model_utils import load_model, get_llm_attention_layer_names
    # Import the HookManager protocol and specific implementations if needed directly by engine (unlikely now)
    from utils.hook_utils import HookManager, GradientAttentionCapture, ActivationCache # Import Protocol
except ImportError as e:
    print(f"Warning: Could not import required utils (data, model, hook): {e}")
    # Define dummy classes/functions as needed
    class HookManager: pass
    class GradientAttentionCapture: pass
    class ActivationCache: pass
    def load_image(*args, **kwargs): return Image.new('RGB', (1,1))
    def build_conversation(*args, **kwargs): return []
    def get_image_token_spans(*args, **kwargs): return []
    def get_token_indices(*args, **kwargs): return (torch.tensor([]), torch.tensor([]))
    def load_model(*args, **kwargs): return (None, None)
    def get_llm_attention_layer_names(*args, **kwargs): return []

# Import necessary classes from transformers
try:
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    from transformers.image_processing_utils import select_best_resolution
    # NOTE: Other transformer imports like resize, pad, image_utils are now primarily used
    # within llava_next_image_utils.py, but keep basic ones engine might still need.
    from transformers.image_utils import get_image_size
except ImportError as e:
    print(f"Warning: Could not import required Hugging Face Transformers classes: {e}")
    class LlavaNextProcessor: pass
    class LlavaNextForConditionalGeneration: pass
    def select_best_resolution(*args, **kwargs): return (336, 336) # Default fallback
    def get_image_size(*args, **kwargs): return (0, 0)


class LLaVANextEngine:
    """
    A VLM engine specifically designed for LLaVA-Next models. (Refactored)

    Provides methods for loading models, processing inputs (including LLaVA-Next
    image feature mapping using external utilities), running forward passes,
    generating responses, and running generalized step-wise analysis via HookManagers.
    """
    def __init__(
        self,
        model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        device: Optional[str] = None,
        load_in_4bit: bool = False,
        use_flash_attn: bool = False,
        # **Important**: enable_gradients only affects model loading.
        # The generate_analyze_stepwise method now checks hook_manager.requires_gradient().
        enable_gradients: bool = False
    ):
        """
        Initialize the LLaVA engine.

        Args:
            model_id (str): HuggingFace model ID for a LLaVA-Next model.
            device (Optional[str]): Target device ('cuda', 'cpu', etc.). Auto-detected if None.
            load_in_4bit (bool): Whether to use 4-bit quantization.
            use_flash_attn (bool): Whether to attempt using Flash Attention 2.
            enable_gradients (bool): Whether to enable gradient computation during model loading.
                                     Required if any analysis needs gradients.
        """
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor: Optional[LlavaNextProcessor] = None
        self.model: Optional[LlavaNextForConditionalGeneration] = None
        self.load_in_4bit = load_in_4bit if torch.cuda.is_available() else False
        self.use_flash_attn = use_flash_attn if torch.cuda.is_available() else False
        # Store the initial setting for reference, but analysis methods rely on hook manager
        self._initial_gradients_enabled = enable_gradients

        _effective_enable_gradients = self._initial_gradients_enabled
        if self.load_in_4bit and _effective_enable_gradients:
            print("Warning: Requesting gradients with 4-bit loading. Gradient support is limited.")
        if not _effective_enable_gradients:
             print("Warning: Initializing engine with enable_gradients=False during load. Analyses requiring gradients might fail if model grads are truly disabled.")

        self._load_model(_effective_enable_gradients)
        self._extract_model_configs()

    def _load_model(self, enable_gradients: bool):
        """Internal method to load the model and processor using the utility function."""
        print(f"Engine attempting to load model '{self.model_id}'...")
        try:
            # Pass enable_gradients flag to the loading utility
            self.model, self.processor = load_model(
                model_id=self.model_id,
                use_flash_attn=self.use_flash_attn,
                load_in_4bit=self.load_in_4bit,
                enable_gradients=enable_gradients, # Pass the flag here
                device_map="auto"
            )
            if self.model is not None:
                 self.device = str(self.model.device)
                 print(f"Model loaded. Main device detected: {self.device}")
                 # Verify if gradients are actually enabled on parameters after loading
                 if enable_gradients and not any(p.requires_grad for p in self.model.parameters()):
                      print("Warning: Model loaded, but parameters do not require gradients despite enable_gradients=True during load.")
                 elif not enable_gradients and any(p.requires_grad for p in self.model.parameters()):
                      print("Warning: Model loaded with enable_gradients=False, but some parameters still require gradients.")


        except ImportError as e:
             print(f"ImportError during model loading: {e}. Make sure necessary libraries (like bitsandbytes, flash-attn, accelerate) are installed.")
             raise
        except Exception as e:
            print(f"Error loading model '{self.model_id}' in engine: {e}")
            self.model = None; self.processor = None
            raise

        if self.model is None or self.processor is None:
             raise RuntimeError(f"Failed to initialize model or processor for {self.model_id}")

        print(f"Engine successfully loaded model and processor.")

    def _extract_model_configs(self):
        """Extracts and stores frequently used config parameters from the loaded model/processor."""
        if self.model is None or self.processor is None:
             raise ValueError("Cannot extract configs before model and processor are loaded.")

        self.config = {} # Store engine-level config info
        model_config = self.model.config
        # Attempt to get vocab size
        self.config["vocab_size"] = getattr(model_config, "vocab_size", None)

        # Get processor image processor config if exists
        processor_img_proc_config = {}
        if hasattr(self.processor, "image_processor") and hasattr(self.processor.image_processor, "config"):
             processor_img_proc_config = self.processor.image_processor.config
             if isinstance(processor_img_proc_config, dict): # Should be dict
                  pass # Use it directly
             elif hasattr(processor_img_proc_config, "to_dict"): # If it's a config object
                 processor_img_proc_config = processor_img_proc_config.to_dict()
             else: # Fallback
                 processor_img_proc_config = {}


        # Image Token ID
        self.config["image_token_id"] = getattr(model_config, "image_token_index", None)
        if self.config["image_token_id"] is None:
            try:
                # Handle potential variations in tokenizer access
                tokenizer = getattr(self.processor, "tokenizer", self.processor)
                image_token_str = getattr(tokenizer, "image_token", "<image>") # Default LLaVA token
                self.config["image_token_id"] = tokenizer.convert_tokens_to_ids(image_token_str)
            except Exception:
                print("Warning: Could not determine image_token_id. Using default 32000.")
                self.config["image_token_id"] = 32000

        # LLM Layers
        # Navigate potentially nested config structure (e.g., self.model.language_model.config)
        llm_config = getattr(self.model, "language_model", self.model).config
        self.config["num_llm_layers"] = getattr(llm_config, "num_hidden_layers", 32) # Default 32

        # Vision Tower Patch Info (using model config primarily)
        vision_config = getattr(model_config, "vision_config", None)
        if vision_config:
            base_image_size = getattr(vision_config, "image_size", 336)
            self.config["vision_raw_patch_size"] = getattr(vision_config, "patch_size", 14)
            if self.config["vision_raw_patch_size"] > 0:
                self.config["vision_patch_grid_size"] = base_image_size // self.config["vision_raw_patch_size"]
            else:
                print("Warning: Vision config patch_size invalid. Defaulting grid size to 24.")
                self.config["vision_patch_grid_size"] = 24 # 336 / 14
        else:
            print("Warning: Model config missing 'vision_config'. Using default vision parameters.")
            self.config["vision_raw_patch_size"] = 14
            self.config["vision_patch_grid_size"] = 24

        # Image Processing Config (Prefer model config, fallback to processor img proc config)
        self.config["image_grid_pinpoints"] = getattr(model_config, "image_grid_pinpoints",
                                                      processor_img_proc_config.get("image_grid_pinpoints", None))
        if self.config["image_grid_pinpoints"] is None:
             print("Warning: 'image_grid_pinpoints' not found in model or processor config. Using default [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]. Feature mapping might be incorrect if model differs.")
             self.config["image_grid_pinpoints"] = [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]


        self.config["vision_feature_select"] = getattr(model_config, "vision_feature_select_strategy", "default")

        # LM Head (try standard path)
        self.lm_head = getattr(getattr(self.model, 'language_model', self.model), 'lm_head', None)
        if self.lm_head is None:
             print("Warning: Could not automatically locate LM head module.")


        print("Engine extracted relevant model configurations:")
        print(f"  Vocab Size: {self.config.get('vocab_size', 'N/A')}")
        print(f"  Image Token ID: {self.config.get('image_token_id', 'N/A')}")
        print(f"  LLM Layers: {self.config.get('num_llm_layers', 'N/A')}")
        print(f"  Vision Base Grid: {self.config.get('vision_patch_grid_size', 'N/A')}x{self.config.get('vision_patch_grid_size', 'N/A')}")
        print(f"  Vision Patch Size: {self.config.get('vision_raw_patch_size', 'N/A')}")
        # print(f"  Image Grid Pinpoints: {self.config.get('image_grid_pinpoints', 'N/A')}") # Can be long

    # --------------------------------------------------------------------------
    # LLaVA-Next Specific Image Processing & Feature Mapping Helpers
    # --- THESE ARE NOW REMOVED AND IMPORTED FROM utils.llava_next_image_utils ---
    # _calculate_resized_dimensions, _resize_for_patching, _pad_for_patching
    # --------------------------------------------------------------------------

    def compute_spatial_preview_image(self, original_image: Image.Image) -> Image.Image:
        """
        Computes the 'spatial preview image' using the external utility function.

        Args:
            original_image (Image.Image): The original PIL Image object (RGB format assumed).

        Returns:
            Image.Image: The spatial preview PIL Image.
        """
        print("Computing spatial preview image (resized + padded) via external utility...")
        if not callable(compute_llava_spatial_preview):
             print("Error: compute_llava_spatial_preview utility function not available.")
             return Image.new('RGB', (1,1)) # Return dummy

        image_grid_pinpoints = self.config.get("image_grid_pinpoints")
        raw_patch_size = self.config.get("vision_raw_patch_size")

        if not image_grid_pinpoints or not raw_patch_size:
             raise ValueError("Missing 'image_grid_pinpoints' or 'vision_raw_patch_size' in engine config for spatial preview.")

        try:
            return compute_llava_spatial_preview(
                original_image=original_image,
                image_grid_pinpoints=image_grid_pinpoints,
                raw_patch_size=raw_patch_size
            )
        except Exception as e:
             print(f"Error calling compute_llava_spatial_preview utility: {e}")
             return Image.new('RGB', (1,1)) # Return dummy on error


    def create_feature_mapping(
        self,
        input_ids: torch.Tensor,
        # Renamed argument for clarity: reflects the size passed TO the processor
        image_size_processed_hw: Tuple[int, int]
        ) -> Dict[str, Any]:
        """
        Create mapping from token indices to spatial grid positions.

        Handles LLaVA-Next's base (fixed grid) and spatial patch features based
        on the actual generated input_ids and the dimensions of the image that
        was processed.

        Args:
            input_ids (torch.Tensor): The processed input_ids tensor [1, seq_len]
                                      output by the processor.
            image_size_processed_hw (Tuple[int, int]): Dimensions (Height, Width)
                of the image that was passed into the processor in build_inputs.
                This might be the original size or a pre-resized size.

        Returns:
            Dict[str, Any]: Dictionary with mapping information, including calculated
                            dimensions based on the processed image size.
        """
        image_token_id = self.config.get("image_token_id")
        # (Input validation remains the same)
        if image_token_id is None: raise ValueError("Missing 'image_token_id' in engine config.")
        image_spans = get_image_token_spans(input_ids, image_token_id)
        if not image_spans:
            print("Engine: No image spans found, cannot create feature mapping.")
            return {"error": "No image spans found in input_ids"}

        # (Get config values remains the same)
        base_grid_size = self.config.get("vision_patch_grid_size", 24)
        raw_patch_size = self.config.get("vision_raw_patch_size", 14)
        image_grid_pinpoints = self.config.get("image_grid_pinpoints")
        if not image_grid_pinpoints: raise ValueError("Missing 'image_grid_pinpoints' in engine config.")
        if raw_patch_size <= 0: raise ValueError("Invalid 'vision_raw_patch_size' in engine config.")

        # (Base Feature Mapping logic remains the same)
        span_start = image_spans[0][0]; span_end = image_spans[-1][1]
        expected_base_tokens = base_grid_size * base_grid_size
        actual_base_token_count = min(expected_base_tokens, span_end - span_start + 1)
        base_start_idx = span_start; base_end_idx = span_start + actual_base_token_count - 1
        mapping_base = {base_start_idx + i: (i // base_grid_size, i % base_grid_size) for i in range(actual_base_token_count)}

        # (Spatial Feature Mapping initialization remains the same)
        spatial_start_idx_potential = base_end_idx + 1
        num_spatial_tokens_available = max(0, span_end - spatial_start_idx_potential + 1)
        mapping_spatial, mapping_newline = {}, {}
        unpadded_grid_rows, unpadded_grid_cols = 0, 0
        grid_rows_padded, grid_cols_padded = 0, 0
        target_resolution_wh_calc, resized_dimensions_wh_calc, padded_dimensions_wh_calc = (0, 0), (0, 0), (0, 0)
        actual_spatial_start_idx, actual_spatial_end_idx = -1, -1

        # (Spatial Feature Mapping calculation logic remains the same, using image_size_processed_hw)
        if num_spatial_tokens_available > 0:
            if not callable(calculate_resized_dimensions) or not callable(select_best_resolution):
                 print("Error: Dimension calculation utilities not available. Cannot map spatial features.")
            else:
                try:
                    # Use the size of the image *passed to the processor* here
                    proc_img_h, proc_img_w = image_size_processed_hw
                    # Select best resolution based on the image size processor received
                    target_resolution_hw = select_best_resolution(image_size_processed_hw, image_grid_pinpoints)
                    target_height, target_width = target_resolution_hw
                    target_resolution_wh_calc = (target_width, target_height) # Store calculated W, H

                    # Calculate expected resize/pad dimensions based on this target
                    resized_width, resized_height = calculate_resized_dimensions(image_size_processed_hw, target_resolution_hw)
                    resized_dimensions_wh_calc = (resized_width, resized_height) # Store calculated W, H

                    padded_height = math.ceil(resized_height / raw_patch_size) * raw_patch_size
                    padded_width = math.ceil(resized_width / raw_patch_size) * raw_patch_size
                    padded_dimensions_wh_calc = (padded_width, padded_height) # Store calculated W, H

                    grid_rows_padded = padded_height // raw_patch_size
                    grid_cols_padded = padded_width // raw_patch_size
                    unpadded_grid_rows = math.ceil(resized_height / raw_patch_size)
                    unpadded_grid_cols = math.ceil(resized_width / raw_patch_size)
                    has_newline = unpadded_grid_rows > 1

                    # (Token mapping loop remains the same)
                    current_token_idx = spatial_start_idx_potential
                    processed_spatial_count = 0
                    for r in range(unpadded_grid_rows):
                        for c in range(unpadded_grid_cols):
                            if current_token_idx <= span_end:
                                if processed_spatial_count == 0: actual_spatial_start_idx = current_token_idx
                                mapping_spatial[current_token_idx] = (r, c)
                                actual_spatial_end_idx = current_token_idx
                                current_token_idx += 1; processed_spatial_count += 1
                            else: break
                        if current_token_idx > span_end: break
                        if has_newline and r < (unpadded_grid_rows - 1):
                            if current_token_idx <= span_end:
                                if processed_spatial_count == 0: actual_spatial_start_idx = current_token_idx
                                mapping_newline[current_token_idx] = r
                                actual_spatial_end_idx = current_token_idx
                                current_token_idx += 1; processed_spatial_count += 1
                            else: break
                except Exception as map_err:
                     print(f"Warning: Error during spatial feature mapping calculation: {map_err}")

        # --- Assemble results (MODIFIED KEYS FOR CLARITY) ---
        return {
            # Base features (mapping based on actual tokens)
            "base_feature": {
                "start_idx": base_start_idx, "end_idx": base_end_idx,
                "grid": (base_grid_size, base_grid_size), # Fixed grid size
                "positions": mapping_base,
                "actual_token_count": actual_base_token_count
            },
            # Patch features (mapping based on actual tokens onto calculated grid)
            "patch_feature": {
                "start_idx": actual_spatial_start_idx,
                "end_idx": max(mapping_spatial.keys()) if mapping_spatial else -1,
                "grid_for_visualization": (grid_rows_padded, grid_cols_padded), # Calculated padded grid
                "grid_unpadded": (unpadded_grid_rows, unpadded_grid_cols),       # Calculated unpadded grid
                "positions": mapping_spatial,
                "actual_token_count": len(mapping_spatial)
            },
            # Newline features (mapping based on actual tokens)
            "newline_feature": {
                "start_idx": min(mapping_newline.keys()) if mapping_newline else -1,
                "end_idx": max(mapping_newline.keys()) if mapping_newline else -1,
                "positions": mapping_newline,
                "actual_token_count": len(mapping_newline)
            },
            # Overall info
            "combined_spatial_end_idx": actual_spatial_end_idx, # Last index used by spatial/newline tokens
            "image_spans": image_spans, # Actual token spans found in input_ids
            # Config and calculation details
            "patch_size": raw_patch_size,
            "mapper_input_size_wh": (image_size_processed_hw[1], image_size_processed_hw[0]), # W, H of image passed to this func
            "best_resolution_calculated_wh": target_resolution_wh_calc, # W, H - calculated target
            "padded_dimensions_calculated_wh": padded_dimensions_wh_calc, # W, H - calculated padded size
            "resized_dimensions_calculated_wh": resized_dimensions_wh_calc, # W, H - calculated resized size
        }


    # --------------------------------------------------------------------------
    # Core Engine Methods
    # --------------------------------------------------------------------------

    def build_inputs(
        self,
        image_source: Union[str, Image.Image],
        prompt: str = "Describe this image in detail.",
        conversation_format: bool = True,
        target_image_size: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Any]:
        """
        Build model inputs, including tensors and feature mapping metadata.
        Optionally pre-resizes the image before passing to the processor.

        Args:
            image_source (Union[str, Image.Image]): PIL Image object or path/URL to an image.
            prompt (str): Text prompt to accompany the image.
            conversation_format (bool): Whether to use the LLaVA conversation format.
            target_image_size (Optional[Tuple[int, int]]): If provided (as W, H),
                pre-resizes the image to this size before processor handles it.
                Defaults to None, using the original image size.

        Returns:
            Dict[str, Any]: Dictionary containing inputs, feature mapping, original image,
                            and spatial preview image (based on the image passed to processor).

        Raises:
            ValueError: If model/processor not loaded or config missing.
            RuntimeError: If required utilities are missing.
        """
        if self.model is None or self.processor is None:
            raise ValueError("Model and processor must be loaded before building inputs.")
        if not callable(load_image) or not callable(build_conversation):
             raise RuntimeError("Required data utility functions (load_image, build_conversation) are not available.")

        # --- Load and Optionally Pre-Resize Image ---
        try:
            print(f"Loading image. Target size before processor: {target_image_size or 'Original'}")
            # Load the image using the utility, applying pre-resizing if specified
            img_to_process = load_image(
                image_source,
                resize_to=target_image_size, # Pass target size here
                verbose=False
            )
            image_size_passed_to_processor = img_to_process.size # W, H
            print(f"Image size passed to processor: {image_size_passed_to_processor}")
            # Keep original image separate if resized, otherwise it's the same
            original_image = load_image(image_source, resize_to=None, verbose=False) if target_image_size else img_to_process

        except Exception as img_err:
            print(f"Error loading/resizing image: {img_err}")
            raise ValueError(f"Failed to load or resize image from {image_source}") from img_err
        # ---

        # Build conversation structure
        conversation = build_conversation(prompt, conversation_format=conversation_format)

        # Format prompt text using chat template
        try:
            if hasattr(self.processor, "apply_chat_template") and getattr(self.processor, 'chat_template', None):
                formatted_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            else: # Basic fallback
                image_token = getattr(getattr(self.processor, "tokenizer", self.processor), "image_token", "<image>")
                text_content = prompt
                if conversation_format and isinstance(conversation, list) and conversation and isinstance(conversation[0], dict) and 'content' in conversation[0]:
                    text_items = [item.get('text', '') for item in conversation[0]['content'] if isinstance(item, dict) and item.get('type') == 'text']
                    if text_items: text_content = text_items[0]
                formatted_prompt = f"USER: {image_token}\n{text_content} ASSISTANT:"
        except Exception as e:
            print(f"Warning: Error applying chat template: {e}. Using basic prompt format.")
            image_token = getattr(getattr(self.processor, "tokenizer", self.processor), "image_token", "<image>")
            formatted_prompt = f"USER: {image_token}\n{prompt} ASSISTANT:"

        # --- Process Image and Text ---
        # Pass the potentially pre-resized image to the processor
        try:
            inputs_dict = self.processor(
                images=img_to_process, # Use the (potentially resized) image
                text=formatted_prompt,
                return_tensors="pt"
            )
        except Exception as proc_err:
            print(f"Error during processor call: {proc_err}")
            raise RuntimeError("LLaVA processor failed.") from proc_err
        # ---

        # --- Compute Feature Mapping ---
        # Pass the size of the image *that was given to the processor* (H, W format)
        # This is needed for select_best_resolution inside create_feature_mapping
        image_size_hw_for_mapping = tuple(reversed(image_size_passed_to_processor)) # Convert W,H to H,W
        feature_mapping = self.create_feature_mapping(inputs_dict['input_ids'], image_size_hw_for_mapping)
        # ---

        # Compute spatial preview image using the potentially resized image
        # This reflects the input the processor actually saw for patching/gridding calculation
        spatial_preview_image = self.compute_spatial_preview_image(img_to_process)

        # Move tensors to the correct device
        try:
            # Try model's device first if multi-GPU, else engine's default device
            target_device = self.model.device if hasattr(self.model, 'hf_device_map') else self.device
            inputs_on_device = {k: v.to(target_device) for k, v in inputs_dict.items() if isinstance(v, torch.Tensor)}
        except Exception as e:
            print(f"Warning: Failed to move inputs to device {target_device}. Error: {e}. Inputs may remain on CPU.")
            inputs_on_device = {k: v for k, v in inputs_dict.items() if isinstance(v, torch.Tensor)} # Keep on CPU if move fails

        # Ensure non-tensor items are included (e.g., potentially 'image_sizes' added by processor)
        for k, v in inputs_dict.items():
            if k not in inputs_on_device: inputs_on_device[k] = v

        return {
            "inputs": inputs_on_device,
            "feature_mapping": feature_mapping,
            "original_image": original_image, # The true original image
            "processed_image_input": img_to_process, # The image passed to the processor
            "spatial_preview_image": spatial_preview_image # Preview based on processed_image_input
        }

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> Dict[str, Any]:
        """
        Performs a single forward pass through the model. (Logic remains the same)

        Args:
            inputs (Dict[str, torch.Tensor]): Model inputs (on the correct device).
            output_hidden_states (bool): Whether to return all hidden states.
            output_attentions (bool): Whether to return all attention weights.

        Returns:
            Dict[str, Any]: The output dictionary from the model.
        """
        if self.model is None:
            raise ValueError("Model is not loaded.")

        self.model.eval()
        # Use inference_mode if gradients are not needed *at all* for this specific pass
        # Note: If subsequent gradient computation is needed based on these inputs/outputs,
        # torch.no_grad() might be insufficient. Consider context carefully.
        # For simple forward passes requested by analyzers like LogitLens, this is fine.
        context = torch.inference_mode() if not (output_attentions or output_hidden_states) else torch.no_grad()

        with context:
            try:
                # Ensure only tensors expected by the model are passed
                model_kwargs = {k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                if 'image_sizes' in inputs and inputs['image_sizes'] is not None: # Pass image_sizes if present
                      model_kwargs['image_sizes'] = inputs['image_sizes']

                outputs = self.model(
                    **model_kwargs,
                    output_hidden_states=output_hidden_states,
                    output_attentions=output_attentions,
                    return_dict=True
                )
            except Exception as e:
                 print(f"Error during model forward pass: {e}")
                 raise

        return outputs

    def generate(
        self,
        image: Union[str, Image.Image],
        prompt: str = "Describe this image in detail.",
        max_new_tokens: int = 256,
        num_beams: int = 3,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        return_dict_in_generate: bool = True,
        **generation_kwargs: Any
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Generate a response based on an image and prompt. (Logic remains mostly the same)

        Args:
            image (Union[str, Image.Image]): PIL Image object or path/URL to an image.
            prompt (str): Text prompt to accompany the image.
            ... (other args same)

        Returns:
            Tuple[str, Optional[Dict[str, Any]]]: Cleaned generated text and optional output dict.
        """
        if self.model is None or self.processor is None:
            raise ValueError("Model and processor must be loaded before generation.")

        prepared_data = self.build_inputs(image, prompt)
        inputs_for_generation = prepared_data['inputs']

        print(f"Generating response with max_new_tokens={max_new_tokens}, num_beams={num_beams}...")
        self.model.eval()

        do_sample = False
        if temperature != 1.0 or top_p is not None or top_k is not None:
             if num_beams == 1: do_sample = True
             else: print(f"Warning: Sampling parameters provided but num_beams > 1. Beam search used.")

        # Determine EOS and PAD token IDs safely
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        try: eos_token_id = tokenizer.eos_token_id
        except AttributeError: eos_token_id = None; print("Warning: Cannot find eos_token_id.")
        try: pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
        except AttributeError: pad_token_id = eos_token_id; print("Warning: Cannot find pad_token_id, using EOS.")


        with torch.inference_mode():
            try:
                # Filter inputs to only pass expected tensors/args to generate
                gen_kwargs = {
                    "input_ids": inputs_for_generation["input_ids"],
                    "pixel_values": inputs_for_generation["pixel_values"],
                    "attention_mask": inputs_for_generation.get("attention_mask"),
                    "image_sizes": inputs_for_generation.get("image_sizes"), # Pass if available
                    "max_new_tokens": max_new_tokens,
                    "num_beams": num_beams,
                    "do_sample": do_sample,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "return_dict_in_generate": return_dict_in_generate,
                    "eos_token_id": eos_token_id,
                    "pad_token_id": pad_token_id,
                    **generation_kwargs
                }
                # Remove None values as generate might not handle them gracefully
                gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

                outputs = self.model.generate(**gen_kwargs)
            except Exception as e:
                 print(f"Error during model.generate: {e}")
                 raise

        # Decode and clean the output text (remains same)
        raw_generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        # More robust cleaning: find the last ASSISTANT: separator
        parts = raw_generated_text.rsplit("ASSISTANT:", 1)
        if len(parts) > 1:
             cleaned_text = parts[1].strip()
        else:
             # Fallback if ASSISTANT: isn't present in the output (e.g., only generation)
             # Try to remove the input prompt part if possible
             input_text_decoded = tokenizer.decode(inputs_for_generation["input_ids"][0], skip_special_tokens=True)
             if raw_generated_text.startswith(input_text_decoded):
                  cleaned_text = raw_generated_text[len(input_text_decoded):].strip()
             else: # Final fallback
                  cleaned_text = raw_generated_text
                  print("Warning: Could not reliably separate prompt from response based on 'ASSISTANT:'. Returning full decoded text.")


        output_dict = outputs if return_dict_in_generate else None
        print("Generation complete.")
        return cleaned_text, output_dict


    # --- REFACTORED Step-wise Analysis Method ---
    def generate_analyze_stepwise(
        self,
        inputs: Dict[str, torch.Tensor],
        num_steps: int,
        hook_manager: HookManager, # Accepts any HookManager implementation
        layers_to_hook: List[str], # Layer names needed by the specific hook_manager
        analysis_callback: Callable[[int, int, torch.Tensor, Dict[str, Any]], Any],
        # Callback: step_idx, target_token_pos, generated_token_id, captured_data -> step_result
        layer_batch_size: Optional[int] = None, # Optional: Only relevant for gradient computation
        callback_cpu_offload: bool = True # Offload captured data before calling callback?
    ) -> Tuple[str, List[Any]]:
        """
        Generates text token by token, running an analysis callback at each step
        using a generic HookManager for data capture.

        Handles gradient computation conditionally based on the HookManager.

        Args:
            inputs (Dict[str, torch.Tensor]): Initial model inputs (on the correct device).
                                              Must include 'input_ids', 'pixel_values'.
                                              'attention_mask' and 'image_sizes' recommended.
            num_steps (int): Number of generation steps to perform.
            hook_manager (HookManager): An instance of a HookManager implementation
                                        (e.g., ActivationCache, GradientAttentionCapture).
            layers_to_hook (List[str]): Names of the layers the hook_manager should target.
            analysis_callback (Callable): Function called after each step. Receives:
                - step_idx (int): Current generation step index (0 to num_steps-1).
                - target_token_pos (int): Index of the token position *before* the generated token.
                - generated_token_id (torch.Tensor): ID of the generated token (CPU).
                - captured_data (Dict[str, Any]): Data captured by the hook_manager
                                                  (e.g., {'activations': ..., 'attentions': ...} or
                                                  {'attention_weights': ..., 'attention_grads': ...}).
                                                  Tensors inside are on GPU unless callback_cpu_offload=True.
                Should return results needed for that step.
            layer_batch_size (Optional[int]): If hook_manager requires gradients, specifies how many
                                             layers' gradients to compute per backward pass. Defaults to None (all layers).
            callback_cpu_offload (bool): Move tensors within captured_data to CPU before calling the callback.

        Returns:
            Tuple[str, List[Any]]:
                - The full generated text sequence (excluding the prompt).
                - A list containing the results returned by the analysis_callback for each step.
        """
        if self.model is None or self.processor is None:
            raise ValueError("Model and processor must be loaded.")
        if not isinstance(hook_manager, HookManager):
             raise TypeError("hook_manager must be an instance implementing the HookManager protocol.")

        # Determine if gradient computation is needed based on the hook manager
        requires_grad = hook_manager.requires_gradient()
        print(f"Starting step-by-step generation. Requires gradients: {requires_grad}")

        if requires_grad and not self._initial_gradients_enabled:
            print("Warning: HookManager requires gradients, but engine was loaded with enable_gradients=False. Analysis may fail.")
        if requires_grad and not any(p.requires_grad for p in self.model.parameters()):
            print("Warning: HookManager requires gradients, but model parameters do not require grad. Analysis may fail.")

        # --- Setup ---
        start_time = time.time()
        initial_input_ids = inputs["input_ids"].clone()
        pixel_values = inputs.get("pixel_values")
        if pixel_values is None: raise ValueError("Missing 'pixel_values' in inputs.")
        attention_mask = inputs.get("attention_mask", torch.ones_like(initial_input_ids))
        image_sizes = inputs.get("image_sizes")
        tokenizer = getattr(self.processor, "tokenizer", self.processor)

        current_input_ids = initial_input_ids
        current_attention_mask = attention_mask
        generated_tokens_text = ""
        step_analysis_results = []
        all_hooked_layer_names = layers_to_hook
        num_layers = len(all_hooked_layer_names)

        # --- Generation Loop ---
        for step_idx in range(num_steps):
            print(f"--- Analyzing Step {step_idx+1}/{num_steps} ---")
            step_start_time = time.time()
            step_captured_data: Dict[str, Any] = {}
            next_token_id: Optional[torch.Tensor] = None
            target_token_pos = current_input_ids.shape[1] - 1 # Position being predicted *from*

            # 1. Predict next token (minimal forward pass)
            self.model.eval()
            with torch.inference_mode(): # Always use inference mode for prediction
                try:
                    outputs_pred = self.model(
                        input_ids=current_input_ids,
                        attention_mask=current_attention_mask,
                        pixel_values=pixel_values,
                        image_sizes=image_sizes,
                        use_cache=True # Cache is useful for prediction speed
                    )
                    logits = outputs_pred.logits[:, -1, :] # Logits for the last position
                    next_token_id = torch.argmax(logits, dim=-1) # Greedy decoding
                    del outputs_pred, logits
                    gc.collect(); torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Error during next token prediction at step {step_idx+1}: {e}")
                    break # Stop generation

            if next_token_id is None:
                 print(f"Failed to generate next token at step {step_idx+1}. Stopping.")
                 break

            new_token_text = tokenizer.decode([next_token_id.item()], skip_special_tokens=True)
            print(f"  Step {step_idx+1}: Predicted token '{new_token_text}' (ID: {next_token_id.item()})")

            # --- 2. Run Forward/Backward with Hooks for Analysis ---
            # This section only runs if requires_grad is False (for ActivationCache etc.)
            # OR if requires_grad is True (for GradientAttentionCapture)
            try:
                 # A. Analysis requiring NO gradients (e.g., LogitLens per step)
                 if not requires_grad:
                     self.model.eval() # Ensure eval mode
                     hook_manager.clear() # Clear previous state/hooks
                     hook_manager.register_hooks(self.model, all_hooked_layer_names)
                     with torch.no_grad(): # No gradients needed
                          _ = self.model( # Run forward pass to trigger hooks
                              input_ids=current_input_ids, # Input leading to the prediction
                              attention_mask=current_attention_mask,
                              pixel_values=pixel_values,
                              image_sizes=image_sizes,
                              output_hidden_states=True, # Likely needed for ActivationCache
                              output_attentions=True,   # Likely needed for ActivationCache
                              use_cache=False # Ensure full pass if needed by hooks
                          )
                     step_captured_data = hook_manager.get_captured_data()
                     hook_manager.clear_hooks() # Clean up hooks for this step

                 # B. Analysis REQUIRING gradients (e.g., Saliency)
                 else:
                     self.model.train() # Switch to train mode for gradients

                     # Determine batching strategy for layers
                     effective_batch_size = layer_batch_size if layer_batch_size is not None else num_layers
                     if effective_batch_size <= 0: effective_batch_size = num_layers

                     batched_weights = {}
                     batched_grads = {}

                     for batch_start in range(0, num_layers, effective_batch_size):
                         batch_end = min(batch_start + effective_batch_size, num_layers)
                         current_layer_batch_names = all_hooked_layer_names[batch_start:batch_end]
                         print(f"    Processing gradient batch {batch_start//effective_batch_size + 1} (Layers {batch_start}-{batch_end-1})...")

                         # IMPORTANT: Clear manager state PER BATCH for grads
                         hook_manager.clear()
                         hook_manager.register_hooks(self.model, current_layer_batch_names)

                         # Forward pass with gradients enabled
                         with torch.enable_grad():
                             outputs_grad = self.model(
                                 input_ids=current_input_ids, # Input leading to the prediction
                                 attention_mask=current_attention_mask,
                                 pixel_values=pixel_values,
                                 image_sizes=image_sizes,
                                 output_attentions=True, # Ensure required outputs are enabled
                                 output_hidden_states=False, # Only enable if hook needs it
                                 use_cache=False # Cannot use cache with grad
                             )
                             logits_grad = outputs_grad.logits[:, target_token_pos, :]
                             log_probs_grad = torch.log_softmax(logits_grad.float(), dim=-1)
                             loss = -log_probs_grad[0, next_token_id.item()] # NLL of predicted token

                         # Backward pass
                         self.model.zero_grad(set_to_none=True)
                         loss.backward()

                         # Retrieve data captured FOR THIS BATCH by the hook manager
                         batch_data = hook_manager.get_captured_data() # get_captured_data clears cache internally
                         # Specific handling for GradientAttentionCapture example:
                         if "attention_weights" in batch_data: batched_weights.update(batch_data["attention_weights"])
                         if "attention_grads" in batch_data: batched_grads.update(batch_data["attention_grads"])

                         # Clean up batch resources
                         hook_manager.clear_hooks() # Remove hooks after batch backward
                         del outputs_grad, logits_grad, log_probs_grad, loss, batch_data
                         gc.collect(); torch.cuda.empty_cache()

                     # Consolidate batched results into step_captured_data
                     # Adapt based on what GradientAttentionCapture returns in get_captured_data
                     step_captured_data = {"attention_weights": batched_weights, "attention_grads": batched_grads}
                     # Ensure model is back in eval mode after all grad batches
                     self.model.eval()

            except Exception as analysis_err:
                 print(f"Error during analysis forward/backward at step {step_idx+1}: {analysis_err}")
                 import traceback; traceback.print_exc()
                 # Ensure hooks are cleared even on error
                 hook_manager.clear_hooks()
                 step_captured_data = {"error": str(analysis_err)} # Mark step data as errored

            # 3. Offload captured data if requested
            if callback_cpu_offload and isinstance(step_captured_data, dict):
                 offloaded_data = {}
                 for key, value in step_captured_data.items():
                      if isinstance(value, torch.Tensor):
                           try: offloaded_data[key] = value.detach().cpu()
                           except Exception as cpu_err: print(f"Warn: Failed to CPU offload {key}: {cpu_err}"); offloaded_data[key] = value.detach() # Keep on GPU if fails
                      elif isinstance(value, dict): # Handle nested dicts (e.g., grads/weights)
                           offloaded_data[key] = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in value.items()}
                      else: offloaded_data[key] = value
                 step_captured_data = offloaded_data
            elif isinstance(step_captured_data, dict): # Detach even if staying on GPU
                 detached_data = {}
                 for key, value in step_captured_data.items():
                     if isinstance(value, torch.Tensor): detached_data[key] = value.detach()
                     elif isinstance(value, dict): detached_data[key] = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in value.items()}
                     else: detached_data[key] = value
                 step_captured_data = detached_data


            # 4. Execute the analysis callback
            try:
                callback_result = analysis_callback(
                    step_idx,
                    target_token_pos,
                    next_token_id.detach().cpu(), # Pass token ID on CPU
                    step_captured_data # Pass the (potentially offloaded) captured data dict
                )
                step_analysis_results.append(callback_result)
            except Exception as cb_err:
                print(f"Error executing analysis callback at step {step_idx+1}: {cb_err}")
                import traceback; traceback.print_exc()
                step_analysis_results.append({"error": str(cb_err)})

            # 5. Prepare for next iteration
            current_input_ids = torch.cat([current_input_ids, next_token_id.unsqueeze(0)], dim=1)
            new_mask_entry = torch.ones((1, 1), dtype=current_attention_mask.dtype, device=current_attention_mask.device)
            current_attention_mask = torch.cat([current_attention_mask, new_mask_entry], dim=1)
            generated_tokens_text += new_token_text # Append *decoded* token text

            step_end_time = time.time()
            print(f"  Step {step_idx+1} finished in {step_end_time - step_start_time:.2f}s")

            # Explicit cleanup at end of step loop
            del step_captured_data
            gc.collect(); torch.cuda.empty_cache()

        # --- End of Loop ---
        end_time = time.time()
        print(f"Step-by-step generation and analysis finished in {end_time - start_time:.2f} seconds.")

        # Final cleanup of the hook manager instance?
        hook_manager.clear()

        return generated_tokens_text, step_analysis_results

    # --- Accessor Methods ---

    def get_attention_layer_names(self) -> List[str]:
        """Get the names of the attention modules within the language model."""
        if self.model is None: raise ValueError("Model not loaded.")
        if not callable(get_llm_attention_layer_names): raise RuntimeError("get_llm_attention_layer_names utility not available.")
        # Ensure patterns match the loaded model type if necessary
        return get_llm_attention_layer_names(self.model)

    def get_model(self) -> Optional[LlavaNextForConditionalGeneration]:
        """Returns the loaded model instance."""
        return self.model

    def get_processor(self) -> Optional[LlavaNextProcessor]:
        """Returns the loaded processor instance."""
        return self.processor

    def get_lm_head(self) -> Optional[nn.Module]:
        """Returns the language model head module."""
        lm_head = getattr(self, 'lm_head', None)
        # Try to find it again if not initially stored
        if lm_head is None and self.model:
             lm_head = getattr(getattr(self.model, 'language_model', self.model), 'lm_head', None)
             if lm_head: self.lm_head = lm_head # Store if found now
        return lm_head

    def get_config(self, key: Optional[str] = None) -> Any:
        """Returns the stored engine configuration dictionary or a specific value."""
        config = getattr(self, 'config', None)
        if config is None:
             print("Warning: Engine config not extracted or available.")
             return None
        if key:
             return config.get(key) # Safely get key, returns None if not found
        return config

    def gradients_enabled_on_load(self) -> bool:
        """Returns whether gradients were requested during the initial model loading."""
        return self._initial_gradients_enabled