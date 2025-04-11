# -*- coding: utf-8 -*-
"""
Model handling and architecture analysis utilities.

Includes functions for:
- Loading VLM models (LLaVA-Next) with configuration options.
- Finding modules within a model by name.
- Identifying attention layers in the language model component.
- Analyzing and summarizing the model's overall architecture.
- Analyzing the specific image processing steps performed by the processor.
"""

import torch
import torch.nn as nn
import time
import os
from typing import Dict, Any, Optional, Union, List, Tuple

# Import necessary components from transformers
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig
)
# Utility for image processing analysis
from transformers.image_processing_utils import select_best_resolution

# Need PIL and requests for analyze_image_processing
from PIL import Image
import requests # Should be handled by data_utils load_image, but keep import here for clarity if analyze_image_processing uses it directly

# Relative import for load_image used in analyze_image_processing
try:
     from utils.data_utils import load_image, build_conversation
except ImportError:
     print("Warning: Could not import 'load_image' from '.data_utils'. Ensure data_utils.py exists.")
     # Define a placeholder if necessary
     def load_image(*args, **kwargs):
         print("Error: load_image is not available. Define it in utils/data_utils.py.")
         raise NotImplementedError


# Model ID mapping (as defined in the notebook)
MODEL_OPTIONS = {
    "mistral_7b": {
        "id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "name": "LLaVA-v1.6-Mistral-7B"
    },
    "vicuna_7b": { # Added Vicuna 7B based on logit lens runner
         "id": "llava-hf/llava-v1.6-vicuna-7b-hf",
         "name": "LLaVA-v1.6-Vicuna-7B"
    },
    "llava_34b": {
        "id": "llava-hf/llava-v1.6-34b-hf",
        "name": "LLaVA-v1.6-34B"
    }
    # Add other models here if needed
}


def load_model(
    model_id: str,
    use_flash_attn: bool = False,
    load_in_4bit: bool = False,
    enable_gradients: bool = False,
    device_map: Optional[str] = "auto" # Added device_map option
) -> Tuple[LlavaNextForConditionalGeneration, LlavaNextProcessor]:
    """
    Loads a LLaVA-Next model and processor with configurable options.

    Handles quantization (4-bit via bitsandbytes), Flash Attention 2,
    gradient enabling, and device mapping.

    Args:
        model_id (str): HuggingFace model ID (e.g., 'llava-hf/llava-v1.6-mistral-7b-hf').
        use_flash_attn (bool): If True, attempts to load the model with Flash Attention 2.
                               Requires compatible hardware and installation. Defaults to False.
        load_in_4bit (bool): If True, loads the model using 4-bit quantization.
                             Requires CUDA, `bitsandbytes`, and `accelerate`. Defaults to False.
                             Note: Gradients are generally not supported well with 4-bit.
        enable_gradients (bool): If True, sets `requires_grad=True` for model parameters *after* loading.
                                 This might have no effect or cause issues if `load_in_4bit` is True. Defaults to False.
        device_map (Optional[str]): The device map strategy for `from_pretrained`.
                                     'auto' distributes model across available devices (GPU/CPU/disk), recommended for large models/quantization.
                                     Set to None to load entirely on the default device (if it fits). Defaults to "auto".

    Returns:
        Tuple[LlavaNextForConditionalGeneration, LlavaNextProcessor]: A tuple containing the loaded model and processor.

    Raises:
        ImportError: If required libraries for quantization or flash attention are missing.
        RuntimeError: If model loading fails for other reasons (e.g., OOM, invalid model ID).
    """
    start_time = time.time()
    print(f"Loading model and processor for: {model_id}...")

    # --- Load Processor ---
    try:
        processor = LlavaNextProcessor.from_pretrained(model_id)
        print("Processor loaded successfully.")
    except Exception as e:
        print(f"Error loading processor for {model_id}: {e}")
        raise RuntimeError(f"Failed to load processor for {model_id}") from e

    # --- Configure Model Loading ---
    attn_implementation = "flash_attention_2" if use_flash_attn else "eager"
    if use_flash_attn:
         print(f"Attempting to use attn_implementation='{attn_implementation}'")
         # Add check for flash-attn installation if possible? transformers might handle this.

    quantization_config = None
    model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32 # Default precision
    effective_device_map = device_map if torch.cuda.is_available() else None # device_map needs accelerate and usually CUDA

    if load_in_4bit:
        if not torch.cuda.is_available():
            print("Warning: load_in_4bit=True requires CUDA. Ignoring quantization.")
        else:
            try:
                # Check if bitsandbytes is installed? Transformers often does this.
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True, # Generally recommended
                    bnb_4bit_quant_type="nf4",      # Recommended quant type
                    bnb_4bit_compute_dtype=torch.float16, # Use float16 for computations
                )
                model_dtype = torch.float16 # Base weights are often loaded as fp16 before quantizing
                print("Configured 4-bit quantization (nf4, float16 compute).")
                if effective_device_map is None:
                     print("Warning: 4-bit quantization typically requires device_map='auto'. Setting device_map='auto'.")
                     effective_device_map = "auto" # Force device_map for 4-bit

            except ImportError:
                print("Error: bitsandbytes library not found. Cannot use load_in_4bit=True.")
                raise
            except Exception as e:
                 print(f"Error configuring BitsAndBytesConfig: {e}")
                 raise

    # --- Load Model ---
    print(f"Loading model with dtype={model_dtype}, device_map='{effective_device_map}', attn='{attn_implementation}'...")
    try:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=model_dtype,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True, # Try to save CPU RAM during load
            device_map=effective_device_map, # Use 'auto' or None
            attn_implementation=attn_implementation,
            trust_remote_code=True # Often needed for custom architectures
        )
        print("Model loaded successfully.")
        print(f"  Model is on device(s): {model.device if effective_device_map is None else 'Multiple (device_map used)'}")

    except ImportError as e:
        print(f"ImportError during model loading: {e}. Ensure 'accelerate' is installed if using device_map or quantization, and 'flash-attn' if using flash_attention_2.")
        raise
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        # Provide more specific advice if possible (e.g., OOM)
        if "out of memory" in str(e).lower():
             print("CUDA Out-of-Memory error detected. Try using 4-bit quantization (load_in_4bit=True) or ensure you have enough GPU RAM.")
        raise RuntimeError(f"Failed to load model {model_id}") from e

    # --- Post-Loading Configuration (Gradients) ---
    if enable_gradients:
        if load_in_4bit:
            print("Warning: Enabling gradients with 4-bit loaded model. This is experimental and may not work as expected or provide meaningful gradients.")
            # Attempt to enable gradients, but results may vary. Peft might be needed for stable 4-bit training/fine-tuning.
            try:
                 # May need specific handling like peft's prepare_model_for_kbit_training
                 print("Attempting to set requires_grad=True on parameters...")
                 model.train() # Put model in train mode
                 for param in model.parameters():
                      param.requires_grad = True
                 # It's often necessary to make only specific parts trainable (e.g., adapters) with 4-bit.
                 # This simple loop might not be sufficient for actual training.
                 print("Note: Full gradient enabling on 4-bit model is complex. Consider using PEFT library for fine-tuning.")
            except Exception as e:
                 print(f"Error enabling gradients on 4-bit model: {e}")
        else:
            print("Enabling gradients for all model parameters...")
            model.train() # Put model in train mode
            for param in model.parameters():
                 param.requires_grad = True
            print("Gradients enabled.")
    else:
         model.eval() # Ensure model is in eval mode if not enabling gradients


    end_time = time.time()
    print(f"Model '{model_id}' and processor loaded in {end_time - start_time:.2f} seconds.")

    return model, processor


def get_module_by_name(model: nn.Module, name: str) -> Optional[nn.Module]:
    """
    Retrieves a submodule from a model using its fully qualified name.

    Handles nested modules separated by dots ('.') and integer indices for
    sequential blocks or list items.

    Args:
        model (nn.Module): The parent model instance.
        name (str): The dot-separated path to the target submodule
                    (e.g., 'language_model.model.layers.5.self_attn').

    Returns:
        Optional[nn.Module]: The submodule if found, otherwise None.
    """
    names = name.split('.')
    module: Union[nn.Module, nn.Sequential, nn.ModuleList] = model

    try:
        for n in names:
            if n.isdigit():
                # Index into ModuleList or Sequential
                module = module[int(n)]
            else:
                # Access attribute (submodule)
                module = getattr(module, n)
        # Ensure the final result is an nn.Module
        if isinstance(module, nn.Module):
             return module
        else:
             # The path led to something other than an nn.Module (e.g., a tensor)
             print(f"Warning: Path '{name}' leads to type {type(module)}, not nn.Module.")
             return None
    except (AttributeError, IndexError, TypeError):
        # Catch errors if path is invalid or indexing fails
        # print(f"Debug: Module not found at path '{name}'.") # Optional debug print
        return None


def matches_pattern(name: str, pattern: str) -> bool:
    """
    Checks if a module name matches a simple pattern with a wildcard '*'.

    The wildcard '*' matches any single component in the dot-separated name,
    typically used for layer numbers.

    Args:
        name (str): The full module name (e.g., 'language_model.model.layers.5.self_attn').
        pattern (str): A pattern containing potentially one or more '*' wildcards
                       (e.g., 'language_model.model.layers.*.self_attn').

    Returns:
        bool: True if the name matches the pattern, False otherwise.
    """
    pattern_parts = pattern.split('.')
    name_parts = name.split('.')

    # Lengths must match for the pattern to apply
    if len(name_parts) != len(pattern_parts):
        return False

    # Compare parts element-wise
    for pattern_part, name_part in zip(pattern_parts, name_parts):
        if pattern_part == '*':
            # Wildcard matches any corresponding part (usually layer index)
            continue
        elif pattern_part != name_part:
            # Literal parts must match exactly
            return False

    # If all parts matched (considering wildcards), the name fits the pattern
    return True


def get_llm_attention_layer_names(model: nn.Module) -> List[str]:
    """
    Extracts the names of likely attention modules within the language model
    component of a VLM.

    Uses predefined patterns common in Transformer architectures. Assumes the
    language model is accessible via `model.language_model`.

    Args:
        model (nn.Module): A VLM model instance, expected to have a `language_model` attribute.

    Returns:
        List[str]: A list of module names identified as attention layers within the language model.
                   Returns an empty list if no language model or attention layers are found.
    """
    attention_layer_names = []

    if not hasattr(model, 'language_model'):
        print("Warning: Model does not have a 'language_model' attribute. Cannot find attention layers.")
        return []

    # Common patterns for attention modules in Hugging Face transformer models
    # (May need adjustment for different model families)
    patterns = [
        'language_model.model.layers.*.self_attn',      # Llama, Mistral, Vicuna style
        'language_model.transformer.h.*.attn',          # GPT-2 style
        'language_model.encoder.layer.*.attention.self',# BERT style (SelfAttention part)
        'language_model.layers.*.attention',            # Some other architectures
        # Add more patterns here if needed
    ]

    print("Searching for language model attention layers using patterns:")
    # for p in patterns: print(f"  - {p}") # Optional: Print patterns being used

    # Iterate through all named modules in the model
    for name, module in model.named_modules():
        # Check if the module name matches any of the defined patterns
        if any(matches_pattern(name, pattern) for pattern in patterns):
             # Basic check: Ensure it looks like an attention mechanism
             # (e.g., has query/key/value projections, though this check is basic)
             is_likely_attention = hasattr(module, 'q_proj') or hasattr(module, 'query') or hasattr(module, 'Wq')
             if is_likely_attention:
                attention_layer_names.append(name)
             # else: # Optional debug print
             #     print(f"  Note: Module '{name}' matched pattern but doesn't look like typical attention. Skipping.")


    if not attention_layer_names:
        print("Warning: No attention layers found matching the known patterns within model.language_model.")
    else:
        print(f"Found {len(attention_layer_names)} potential attention layer names in the language model.")
        # print(f"  Example: {attention_layer_names[0]}") # Optional: Print first found name

    return attention_layer_names


def analyze_model_architecture(model: LlavaNextForConditionalGeneration) -> Dict[str, Any]:
    """
    Analyzes and extracts key architectural information from a LLaVA-Next model instance.

    Args:
        model (LlavaNextForConditionalGeneration): The loaded LLaVA-Next model instance.

    Returns:
        Dict[str, Any]: A dictionary containing structured information about the
                        vision tower, language model, projector, and other relevant configs.
    """
    result: Dict[str, Any] = {"model_type": type(model).__name__}
    print(f"Analyzing architecture for model type: {result['model_type']}")

    # --- Vision Tower Analysis ---
    if hasattr(model, 'vision_tower') and hasattr(model.vision_tower, 'config'):
        vision_tower = model.vision_tower
        vision_config = vision_tower.config
        result["vision_tower"] = {
            "type": getattr(vision_config, "model_type", type(vision_tower).__name__),
            "hidden_size": getattr(vision_config, "hidden_size", "N/A"),
            "num_layers": getattr(vision_config, "num_hidden_layers", "N/A"),
            "num_attention_heads": getattr(vision_config, "num_attention_heads", "N/A"),
            "image_size": getattr(vision_config, "image_size", "N/A"),
            "patch_size": getattr(vision_config, "patch_size", "N/A"),
        }
        # Calculate num_patches if possible
        img_size = result["vision_tower"]["image_size"]
        patch_size = result["vision_tower"]["patch_size"]
        if isinstance(img_size, int) and isinstance(patch_size, int) and patch_size > 0:
             num_patches_per_dim = img_size // patch_size
             result["vision_tower"]["num_patches"] = num_patches_per_dim ** 2
             result["vision_tower"]["patches_per_dim"] = num_patches_per_dim
        else:
             result["vision_tower"]["num_patches"] = "N/A"
             result["vision_tower"]["patches_per_dim"] = "N/A"
        print("  Extracted Vision Tower info.")
    else:
         print("  Warning: Vision Tower or its config not found.")
         result["vision_tower"] = {"status": "Not Found"}

    # --- Language Model Analysis ---
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'config'):
        language_model = model.language_model
        lang_config = language_model.config
        result["language_model"] = {
            "type": getattr(lang_config, "model_type", type(language_model).__name__),
            "hidden_size": getattr(lang_config, "hidden_size", "N/A"),
            "num_layers": getattr(lang_config, "num_hidden_layers", "N/A"),
            "num_attention_heads": getattr(lang_config, "num_attention_heads", "N/A"),
            "vocab_size": getattr(lang_config, "vocab_size", "N/A"),
        }
        print("  Extracted Language Model info.")
    else:
        print("  Warning: Language Model or its config not found.")
        result["language_model"] = {"status": "Not Found"}

    # --- Multi-Modal Projector Analysis ---
    if hasattr(model, 'multi_modal_projector'):
        projector = model.multi_modal_projector
        proj_type = type(projector).__name__
        result["projector"] = {"type": proj_type}
        # Try to infer input/output features (structure might vary)
        in_features, out_features = "N/A", "N/A"
        if isinstance(projector, nn.Sequential):
             # Look at the first and last linear layers if it's a sequence
             first_layer = next((m for m in projector if isinstance(m, nn.Linear)), None)
             last_layer = next((m for m in reversed(projector) if isinstance(m, nn.Linear)), None)
             if first_layer: in_features = getattr(first_layer, "in_features", "N/A")
             if last_layer: out_features = getattr(last_layer, "out_features", "N/A")
        elif isinstance(projector, nn.Linear):
             # Simpler case: projector is a single Linear layer
             in_features = getattr(projector, "in_features", "N/A")
             out_features = getattr(projector, "out_features", "N/A")
        # LLaVA-NeXT often has linear_1, linear_2 structure inside an MLP
        elif hasattr(projector, 'linear_1') and isinstance(projector.linear_1, nn.Linear):
             in_features = getattr(projector.linear_1, "in_features", "N/A")
             if hasattr(projector, 'linear_2') and isinstance(projector.linear_2, nn.Linear):
                   out_features = getattr(projector.linear_2, "out_features", "N/A")
             else: # If only linear_1 exists, it might be the output layer
                  out_features = getattr(projector.linear_1, "out_features", "N/A")


        result["projector"]["in_features"] = in_features
        result["projector"]["out_features"] = out_features
        print(f"  Extracted Projector info (Type: {proj_type}).")
    else:
         print("  Warning: Multi-Modal Projector not found.")
         result["projector"] = {"status": "Not Found"}

    # --- Other Relevant Config from Main Model Config ---
    if hasattr(model, 'config'):
        config = model.config
        result["config"] = {
            "image_token_index": getattr(config, 'image_token_index', None),
            "image_grid_pinpoints": getattr(config, 'image_grid_pinpoints', None),
            "vision_feature_layer": getattr(config, 'vision_feature_layer', None), # Layer index in vision tower to extract features
            "vision_feature_select_strategy": getattr(config, 'vision_feature_select_strategy', None) # e.g., 'patch', 'cls_patch'
        }
        print("  Extracted relevant model config values.")
    else:
         print("  Warning: Main model config not found.")
         result["config"] = {}

    print("Architecture analysis complete.")
    return result


def print_architecture_summary(arch_info: Dict[str, Any]) -> None:
    """
    Prints a formatted summary of the model architecture information.

    Args:
        arch_info (Dict[str, Any]): The dictionary returned by `analyze_model_architecture`.
    """
    model_type = arch_info.get('model_type', 'VLM')
    print(f"\n===== {model_type} Architecture Summary =====")

    # --- Vision Tower ---
    vision = arch_info.get("vision_tower", {})
    if vision and vision.get("status") != "Not Found":
        print("\n--- Vision Encoder (Vision Tower) ---")
        print(f"  Type: {vision.get('type', 'N/A')}")
        print(f"  Hidden Size: {vision.get('hidden_size', 'N/A')}")
        print(f"  Num Layers: {vision.get('num_layers', 'N/A')}")
        print(f"  Num Heads: {vision.get('num_attention_heads', 'N/A')}")
        print(f"  Input Image Size: {vision.get('image_size', 'N/A')}px")
        print(f"  Patch Size: {vision.get('patch_size', 'N/A')}px")
        print(f"  Patches per Dim: {vision.get('patches_per_dim', 'N/A')}")
        print(f"  Total Patches: {vision.get('num_patches', 'N/A')}")
    else:
         print("\n--- Vision Encoder (Vision Tower): Not Found or Info Missing ---")

    # --- Language Model ---
    language = arch_info.get("language_model", {})
    if language and language.get("status") != "Not Found":
        print("\n--- Language Model ---")
        print(f"  Type: {language.get('type', 'N/A')}")
        print(f"  Hidden Size: {language.get('hidden_size', 'N/A')}")
        print(f"  Num Layers: {language.get('num_layers', 'N/A')}")
        print(f"  Num Heads: {language.get('num_attention_heads', 'N/A')}")
        print(f"  Vocab Size: {language.get('vocab_size', 'N/A')}")
    else:
         print("\n--- Language Model: Not Found or Info Missing ---")

    # --- Projector ---
    projector = arch_info.get("projector", {})
    if projector and projector.get("status") != "Not Found":
        print("\n--- Multi-Modal Projector ---")
        print(f"  Type: {projector.get('type', 'N/A')}")
        print(f"  Input Features: {projector.get('in_features', 'N/A')}")
        print(f"  Output Features: {projector.get('out_features', 'N/A')}")
    else:
         print("\n--- Multi-Modal Projector: Not Found or Info Missing ---")

    # --- Other Config ---
    config = arch_info.get("config", {})
    if config:
        print("\n--- Other Relevant Configurations ---")
        print(f"  Image Token Index: {config.get('image_token_index', 'N/A')}")
        print(f"  Vision Feature Layer Index: {config.get('vision_feature_layer', 'N/A')}")
        print(f"  Vision Feature Select Strategy: {config.get('vision_feature_select_strategy', 'N/A')}")
        print(f"  Image Grid Pinpoints (for high-res): {config.get('image_grid_pinpoints', 'N/A')}")

    print("\n============================================")


def analyze_image_processing(
    model: LlavaNextForConditionalGeneration,
    processor: LlavaNextProcessor,
    image_source: Union[Image.Image, str],
    prompt: str = "Describe this image"
) -> Dict[str, Any]:
    """
    Analyzes how a given image is processed by the LLaVA-Next processor
    and prepared as input for the model.

    Loads the image, prepares inputs using the processor, and extracts details about
    image resizing, patching strategy, and tokenization related to the image.

    Args:
        model (LlavaNextForConditionalGeneration): The loaded LLaVA-Next model instance.
        processor (LlavaNextProcessor): The corresponding processor instance.
        image_source (Union[Image.Image, str]): The input image (PIL Image, URL, or file path).
        prompt (str): A sample text prompt to use for generating the full input sequence.

    Returns:
        Dict[str, Any]: A dictionary containing detailed analysis results, including:
                        - 'original_image': Info about the raw input image.
                        - 'processing_params': Key parameters used by the processor.
                        - 'best_resolution': The target resolution selected for high-res processing.
                        - 'processed_image_tensor': Shape and dtype of the final pixel_values tensor.
                        - 'patch_info': Details about ViT patching from its config.
                        - 'token_info': Details about image tokens in the input sequence.
                        - 'inputs_cpu': A copy of the processed model inputs moved to CPU.
    """
    print("\n--- Starting Image Processing Analysis ---")
    try:
        # 1. Load the original image using the utility function
        # Do not resize here; let the processor handle it.
        # Use verbose=False to avoid redundant prints from load_image.
        original_image = load_image(image_source, resize_to=None, verbose=False)
        original_size_wh = original_image.size # (Width, Height)
        original_size_hw = (original_image.height, original_image.width) # (Height, Width)
        print(f"  Original image loaded: Size (WxH) = {original_size_wh}, Mode = {original_image.mode}")

        analysis = {
            "original_image": {"size_wh": original_size_wh, "mode": original_image.mode}
        }

        # 2. Extract relevant parameters from processor and model config
        # Use getattr for safe access
        proc_img_processor = getattr(processor, "image_processor", None)
        model_config = getattr(model, "config", None)
        vision_config = getattr(model_config, "vision_config", None) if model_config else None

        analysis["processing_params"] = {
            "image_token": getattr(processor, "image_token", "<image>"),
            "image_token_id": processor.tokenizer.convert_tokens_to_ids(getattr(processor, "image_token", "<image>")),
            "image_grid_pinpoints": getattr(proc_img_processor, "image_grid_pinpoints", getattr(model_config, "image_grid_pinpoints", "N/A")),
            "vision_feature_select_strategy": getattr(model_config, "vision_feature_select_strategy", "N/A"),
            "vision_feature_layer": getattr(model_config, "vision_feature_layer", "N/A"),
            "raw_patch_size": getattr(vision_config, "patch_size", "N/A") if vision_config else "N/A"
        }
        print(f"  Processor/Model Params:")
        for k, v in analysis["processing_params"].items(): print(f"    {k}: {v}")


        # 3. Determine the 'best' resolution the processor would target
        grid_pinpoints = analysis["processing_params"]["image_grid_pinpoints"]
        if grid_pinpoints != "N/A" and isinstance(grid_pinpoints, list):
             try:
                  analysis["best_resolution_hw"] = select_best_resolution(original_size_hw, grid_pinpoints)
                  print(f"  Selected 'best_resolution' (HxW): {analysis['best_resolution_hw']}")
             except Exception as e:
                  print(f"  Warning: Could not determine best resolution: {e}")
                  analysis["best_resolution_hw"] = "Error"
        else:
             print("  Image grid pinpoints not available; cannot determine best resolution.")
             analysis["best_resolution_hw"] = "N/A"

        # 4. Prepare inputs using the processor to get the actual processed tensors
        # We need a prompt to get the full input_ids structure
        # Use the basic conversation format helper
        conversation = build_conversation(prompt, conversation_format=True)
        try:
            formatted_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        except Exception: # Fallback if template fails
             image_token_plh = analysis["processing_params"]["image_token"]
             formatted_prompt = f"USER: {image_token_plh}\n{prompt} ASSISTANT:"

        # Generate inputs *without* moving to device yet
        # This gives us pixel_values and input_ids as processed
        inputs = processor(images=original_image, text=formatted_prompt, return_tensors="pt")
        print(f"  Processor generated inputs.")

        # 5. Analyze the processed tensors
        if 'pixel_values' in inputs:
            pv_tensor = inputs['pixel_values']
            analysis["processed_image_tensor"] = {"shape": list(pv_tensor.shape), "dtype": str(pv_tensor.dtype)}
            print(f"  Processed pixel_values tensor: Shape={analysis['processed_image_tensor']['shape']}, Dtype={analysis['processed_image_tensor']['dtype']}")
        else:
            analysis["processed_image_tensor"] = None
            print("  Warning: 'pixel_values' not found in processor output.")

        # 6. Analyze ViT patching based on its config
        if vision_config:
             patch_size_cfg = getattr(vision_config, "patch_size", None)
             image_size_cfg = getattr(vision_config, "image_size", None) # This is the size ViT expects (e.g., 336)
             if patch_size_cfg and image_size_cfg:
                 analysis["patch_info"] = {
                     "vit_input_size": image_size_cfg,
                     "patch_size": patch_size_cfg,
                     "patches_per_dim": image_size_cfg // patch_size_cfg if patch_size_cfg > 0 else "N/A",
                     "total_patches_base": (image_size_cfg // patch_size_cfg) ** 2 if patch_size_cfg > 0 else "N/A",
                 }
                 print(f"  ViT Config Patch Info: Input Size={image_size_cfg}px, Patch Size={patch_size_cfg}px, Base Patches={analysis['patch_info']['total_patches_base']}")
             else:
                 analysis["patch_info"] = {"status": "Config missing size info"}
                 print("  ViT config missing image_size or patch_size for patch analysis.")
        else:
             analysis["patch_info"] = {"status": "Vision config not found"}


        # 7. Analyze token sequence related to the image
        # Compare input_ids with text-only tokenization to infer image token count
        text_only_inputs = processor.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs.get("input_ids")
        image_token_id = analysis["processing_params"]["image_token_id"]

        if input_ids is not None:
             num_combined_tokens = input_ids.shape[1]
             num_text_tokens = text_only_inputs.input_ids.shape[1]
             # The processor replaces the single <image> placeholder with the actual number of image tokens
             num_image_tokens_in_sequence = input_ids[0].eq(image_token_id).sum().item()

             analysis["token_info"] = {
                 "image_token_string": analysis["processing_params"]["image_token"],
                 "image_token_id": image_token_id,
                 "text_only_token_length": num_text_tokens,
                 "combined_token_length": num_combined_tokens,
                 "num_image_tokens_found": num_image_tokens_in_sequence,
             }
             print(f"  Token Analysis: Image Token ID={image_token_id}, Text Tokens={num_text_tokens}, Combined Tokens={num_combined_tokens}, Found Image Tokens={num_image_tokens_in_sequence}")
        else:
            analysis["token_info"] = {"status": "input_ids not found"}
            print("  Warning: 'input_ids' not found in processor output for token analysis.")

        # 8. Store CPU copy of inputs for potential visualization later
        analysis["inputs_cpu"] = {k: v.detach().cpu() for k, v in inputs.items() if torch.is_tensor(v)}
        print("  Stored CPU copy of processed inputs.")

    except Exception as e:
         print(f"Error during image processing analysis: {e}")
         import traceback
         traceback.print_exc()
         analysis["error"] = str(e)

    print("--- Image Processing Analysis Complete ---")
    return analysis