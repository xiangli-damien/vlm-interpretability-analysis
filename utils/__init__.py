# -*- coding: utf-8 -*-
"""
Initialization file for the VLM Analysis utilities module (utils).

Exposes various helper functions and classes for data handling,
model interaction, visualization, hooking, and LLaVA-specific image processing.
Uses absolute imports assuming the project root is in sys.path.
"""

# --- Data Utilities ---
from utils.data_utils import (
    clean_memory,
    load_image,
    build_conversation,
    get_token_masks,
    get_token_indices,
    get_image_token_spans,
)

# --- Hooking Utilities ---
from utils.hook_utils import (
    ActivationCache,
    GradientAttentionCapture,
)

# --- Model Utilities ---
from utils.model_utils import (
    load_model,
    get_module_by_name,
    get_llm_attention_layer_names,
    # matches_pattern, # Likely internal, commented out from __all__
    analyze_model_architecture,
    print_architecture_summary,
    analyze_image_processing,
)

# --- Visualization Utilities ---
from utils.visual_utils import (
    visualize_information_flow,
    visualize_attention_heatmap,
    visualize_processed_image_input,
    visualize_token_probabilities, # Added based on logit lens workflow
)

# --- LLaVA-Next Specific Image Utilities ---
from utils.llava_next_image_utils import (
    calculate_resized_dimensions,
    resize_image_for_patching,
    pad_image_for_patching,
    compute_llava_spatial_preview
)


# --- Public API Definition (`__all__`) ---
# Lists names to be imported with 'from utils import *'
__all__ = [
    # data_utils
    "clean_memory",
    "load_image",
    "build_conversation",
    "get_token_masks",
    "get_token_indices",
    "get_image_token_spans",

    # hook_utils
    "ActivationCache",
    "GradientAttentionCapture",

    # model_utils
    "load_model",
    "get_module_by_name",
    "get_llm_attention_layer_names",
    # "matches_pattern", # Excluded - likely internal helper
    "analyze_model_architecture",
    "print_architecture_summary",
    "analyze_image_processing",

    # visual_utils
    "visualize_information_flow",
    "visualize_attention_heatmap",
    "visualize_processed_image_input",
    "visualize_token_probabilities",

    # llava_next_image_utils
    "calculate_resized_dimensions",
    "resize_image_for_patching",
    "pad_image_for_patching",
    "compute_llava_spatial_preview",
]