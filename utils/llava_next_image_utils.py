# -*- coding: utf-8 -*-
"""
Image processing utilities specific to LLaVA-Next models.

Includes functions for resizing, padding, and generating spatial preview images
based on LLaVA-Next's multi-resolution strategy.
"""
import math
import numpy as np
from PIL import Image
import torch # Keep torch for potential future tensor ops

# Import necessary components from transformers
from transformers.image_processing_utils import select_best_resolution
from transformers.image_transforms import (
    PaddingMode,
    resize as hf_resize,
    # pad as hf_pad, # Not used directly, numpy pad is used
)
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    to_numpy_array
)
from typing import Tuple

# Function to calculate resized dimensions based on target resolution, preserving aspect ratio
def calculate_resized_dimensions(orig_size_hw: Tuple[int, int], target_res_hw: Tuple[int, int]) -> Tuple[int, int]:
    """
    Calculates aspect-ratio preserved dimensions to fit within a target resolution.

    Args:
        orig_size_hw (Tuple[int, int]): Original image size (Height, Width).
        target_res_hw (Tuple[int, int]): Target resolution (Height, Width).

    Returns:
        Tuple[int, int]: New dimensions (Width, Height).
    """
    orig_h, orig_w = orig_size_hw
    target_h, target_w = target_res_hw
    if orig_w <= 0 or orig_h <= 0: return target_w, target_h # Handle invalid input

    scale_w = target_w / orig_w
    scale_h = target_h / orig_h

    if scale_w < scale_h:
        new_w = target_w
        new_h = min(math.ceil(orig_h * scale_w), target_h)
    else:
        new_h = target_h
        new_w = min(math.ceil(orig_w * scale_h), target_w)
    return int(new_w), int(new_h)

# Function to resize an image array using Hugging Face's resize utility
def resize_image_for_patching(image: np.array, target_resolution_hw: tuple, resample, input_data_format) -> np.array:
    """
    Resizes an image numpy array using HF's resize, preserving aspect ratio.

    Args:
        image (np.array): Input image as a numpy array.
        target_resolution_hw (tuple): Target resolution (Height, Width).
        resample: The resampling filter (e.g., PILImageResampling.BICUBIC).
        input_data_format: The channel dimension format of the input image array.

    Returns:
        np.array: The resized image as a numpy array.
    """
    resized_w, resized_h = calculate_resized_dimensions(
         get_image_size(image, channel_dim=input_data_format), target_resolution_hw
    )
    return hf_resize(image, size=(resized_h, resized_w), resample=resample, input_data_format=input_data_format)

# Function to pad an image array to be divisible by a patch size
def pad_image_for_patching(image: np.array, raw_patch_size: int, input_data_format) -> np.array:
    """
    Pads an image numpy array so its dimensions are divisible by the raw patch size.
    Uses symmetric padding with constant value 0.

    Args:
        image (np.array): Input image as a numpy array.
        raw_patch_size (int): The size of the vision model's patches.
        input_data_format: The channel dimension format of the input image array.

    Returns:
        np.array: The padded image as a numpy array.
    """
    if raw_patch_size <= 0: raise ValueError("raw_patch_size must be positive.")
    resized_height, resized_width = get_image_size(image, channel_dim=input_data_format)
    padded_height = math.ceil(resized_height / raw_patch_size) * raw_patch_size
    padded_width = math.ceil(resized_width / raw_patch_size) * raw_patch_size
    pad_height_total = padded_height - resized_height
    pad_width_total = padded_width - resized_width
    if pad_height_total == 0 and pad_width_total == 0: return image
    pad_top = pad_height_total // 2
    pad_bottom = pad_height_total - pad_top
    pad_left = pad_width_total // 2
    pad_right = pad_width_total - pad_left
    if input_data_format == ChannelDimension.FIRST or input_data_format == "channels_first":
        padding_np = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
    elif input_data_format == ChannelDimension.LAST or input_data_format == "channels_last":
        padding_np = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    else: raise ValueError(f"Unsupported input_data_format for padding: {input_data_format}")
    return np.pad(image, pad_width=padding_np, mode='constant', constant_values=0)

# Function to compute the spatial preview image (resized + padded)
def compute_llava_spatial_preview(
    original_image: Image.Image,
    image_grid_pinpoints: list,
    raw_patch_size: int
    ) -> Image.Image:
    """
    Computes the 'spatial preview image' for LLaVA-Next by resizing and padding.

    Args:
        original_image (Image.Image): The original PIL Image object (RGB).
        image_grid_pinpoints (list): List of possible resolutions from model config.
        raw_patch_size (int): The raw patch size of the vision encoder.

    Returns:
        Image.Image: The spatial preview PIL Image (resized + padded).
    """
    if original_image.mode != "RGB": original_image = original_image.convert("RGB")
    image_np = to_numpy_array(original_image)
    input_df = infer_channel_dimension_format(image_np)
    orig_size_hw = (original_image.height, original_image.width)
    target_resolution_hw = select_best_resolution(orig_size_hw, image_grid_pinpoints)
    resized_image_np = resize_image_for_patching(
        image_np, target_resolution_hw, resample=PILImageResampling.BICUBIC, input_data_format=input_df
    )
    padded_image_np = pad_image_for_patching(
        resized_image_np, raw_patch_size=raw_patch_size, input_data_format=input_df
    )
    if padded_image_np.dtype != np.uint8: padded_image_np = np.clip(padded_image_np, 0, 255).astype(np.uint8)
    if input_df == ChannelDimension.FIRST: padded_image_np = padded_image_np.transpose(1, 2, 0)
    return Image.fromarray(padded_image_np)