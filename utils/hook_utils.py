# -*- coding: utf-8 -*-
"""
PyTorch Hook Utilities for Model Analysis.

Provides classes to capture intermediate activations, attention weights,
and gradients during model forward and backward passes using hooks.
Includes methods required for generic step-wise analysis engine integration.
"""

import torch
import torch.nn as nn
import gc
# Add runtime_checkable to typing imports
from typing import Dict, Any, Optional, List, Tuple, Callable, Set, Protocol, runtime_checkable

# Import necessary utility from model_utils (using absolute path based on structure)
try:
    # Assuming utils and engine are siblings under the project root
    from utils.model_utils import get_module_by_name
except ImportError:
    print("Warning: Could not import 'get_module_by_name' from 'utils.model_utils'. Ensure model_utils.py exists in the utils directory.")
    # Define a placeholder if necessary
    def get_module_by_name(model, name):
        print(f"Error: get_module_by_name placeholder called for '{name}'. Define in utils/model_utils.py.")
        return None


# --- HookManager Interface (Conceptual via Protocol) ---
# Defines the methods expected by the engine's stepwise analysis function.
# *** ADD @runtime_checkable decorator ***
@runtime_checkable
class HookManager(Protocol):
    """
    Protocol defining the interface for hook manager classes used by the engine.
    Requires @runtime_checkable to support isinstance() checks.
    """
    def register_hooks(self, model: nn.Module, layer_names: List[str]):
        """Registers hooks on the specified layers of the model."""
        ... # Ellipsis indicates abstract method in Protocol

    def clear_hooks(self):
        """Removes all registered hooks."""
        ...

    def clear_cache(self):
        """Clears any captured data from the cache."""
        ...

    def clear(self):
        """Clears both hooks and cached data."""
        ...

    def get_captured_data(self) -> Dict[str, Any]:
        """Returns the data captured by the hooks since the last clear_cache."""
        ...

    def requires_gradient(self) -> bool:
        """Returns True if this hook manager requires gradient computation."""
        ...


# --- Activation Cache (Forward Only) ---
class ActivationCache(HookManager):
    """
    Captures activations and optionally attention weights using forward hooks.
    Provides a clean way to register hooks and store results.
    Implements the HookManager protocol.
    """
    def __init__(self):
        self.activations: Dict[str, torch.Tensor] = {}  # Stores hidden states from layers
        self.attentions: Dict[str, torch.Tensor] = {}   # Stores attention weights
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []  # Tracks registered hooks

    def _hook_fn(self, layer_name: str, capture_attention: bool = False):
        """Creates a hook function that captures layer outputs."""
        def hook(module, input, output):
            # Capture hidden states
            activation_tensor = None
            if isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                activation_tensor = output[0]
            elif isinstance(output, torch.Tensor):
                activation_tensor = output

            if activation_tensor is not None:
                self.activations[layer_name] = activation_tensor.detach()

            # Optionally capture attention weights
            if capture_attention:
                attn_weights = None
                # Common patterns for attention weights in Hugging Face models
                if isinstance(output, tuple):
                    # Pattern 1: (hidden_state, present_key_value, attention_weights)
                    if len(output) > 2 and isinstance(output[2], torch.Tensor) and output[2].ndim >= 4:
                       # Basic check: Often [batch, heads, seq, seq]
                       if output[2].shape[-1] == output[2].shape[-2]:
                           attn_weights = output[2]
                    # Pattern 2: (hidden_state, attention_weights, present_key_value)
                    elif len(output) > 1 and isinstance(output[1], torch.Tensor) and output[1].ndim >= 4:
                         if output[1].shape[-1] == output[1].shape[-2]:
                             attn_weights = output[1]
                    # Pattern 3: (hidden_state, maybe_pooled_output, attentions) - Less common direct output
                    # Add more patterns specific to the models you use if needed

                if attn_weights is not None:
                    self.attentions[layer_name] = attn_weights.detach()
        return hook

    def register_hooks(self, model: torch.nn.Module, layer_names: List[str], attention_layer_names: List[str] = None) -> None:
        """
        Registers forward hooks on specified layers.

        Args:
            model (torch.nn.Module): The model to hook into.
            layer_names (List[str]): List of layer names to capture hidden states from.
            attention_layer_names (List[str], optional): List of layer names to also capture attention from. Defaults to None.
        """
        self.clear() # Clear previous hooks and data
        _attention_layers = set(attention_layer_names or [])
        all_layers_to_hook = set(layer_names) | _attention_layers

        for name, module in model.named_modules():
            if name in all_layers_to_hook:
                capture_attn = name in _attention_layers
                handle = module.register_forward_hook(self._hook_fn(name, capture_attention=capture_attn))
                self._hooks.append(handle)
        print(f"ActivationCache: Registered {len(self._hooks)} forward hooks.")

    def get_captured_data(self) -> Dict[str, Any]:
        """Returns captured activations and attentions, then clears storage."""
        data = {"activations": self.activations.copy(), "attentions": self.attentions.copy()}
        self.activations.clear()
        self.attentions.clear()
        return data

    def clear_hooks(self) -> None:
        """Removes all registered hooks."""
        # print(f"ActivationCache: Removing {len(self._hooks)} hooks.")
        for handle in self._hooks:
            handle.remove()
        self._hooks = []

    def clear(self) -> None:
        """Clears captured data and removes hooks."""
        self.clear_hooks()
        self.activations = {}
        self.attentions = {}
        # print("ActivationCache: Cleared data and hooks.")

    def requires_gradient(self) -> bool:
        """ActivationCache does not require gradients."""
        return False

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_value, traceback): self.clear()


# --- Gradient Attention Capture (Forward + Backward) ---
class GradientAttentionCapture(HookManager):
    """
    Captures attention weights and gradients.
    Provides a compute_saliency method similar to implied old working version.
    """
    def __init__(self, cpu_offload: bool = True): # Only cpu_offload argument
        """
        Initializes the gradient capturer.

        Args:
            cpu_offload (bool): If True, computed saliency scores are moved to CPU.
        """
        self.attention_weights: Dict[str, torch.Tensor] = {}
        self.attention_grads: Dict[str, torch.Tensor] = {}
        self._forward_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._backward_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._tensor_grad_hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        self._hooked_layers: Set[str] = set()
        self.cpu_offload = cpu_offload # Assume this applies to saliency scores now
        print(f"[DEBUG] Initialized OldStyle GradientAttentionCapture (CPU Offload Saliency: {self.cpu_offload})")

    # --- _create_forward_hook_fn (Use robust version) ---
    def _create_forward_hook_fn(self, layer_name: str) -> Callable:
        def hook(module, input, output):
            # print(f"[DEBUG] Forward Hook Fired: {layer_name}")
            attn_weights = None
            if isinstance(output, tuple): # Find weights logic
                if len(output) > 2 and isinstance(output[2], torch.Tensor) and output[2].ndim >= 4 and output[2].shape[-1] == output[2].shape[-2]: attn_weights = output[2]
                elif len(output) > 1 and isinstance(output[1], torch.Tensor) and output[1].ndim >= 4 and output[1].shape[-1] == output[1].shape[-2]: attn_weights = output[1]
            if attn_weights is not None and attn_weights.requires_grad:
                # print(f"  [DEBUG] {layer_name}: Storing weights {attn_weights.shape}")
                self.attention_weights[layer_name] = attn_weights
        return hook

    # --- _create_backward_hook_fn (Use robust version) ---
    def _create_backward_hook_fn(self, layer_name: str) -> Callable:
        def module_backward_hook(module, grad_input, grad_output):
            # print(f"[DEBUG] Backward Hook Fired: {layer_name}")
            if layer_name in self.attention_weights:
                attn_weights_tensor = self.attention_weights[layer_name]
                if attn_weights_tensor.requires_grad:
                    def _capture_tensor_grad(grad):
                        # print(f"[DEBUG] Tensor Grad Hook Called: {layer_name}")
                        if grad is not None:
                            # print(f"  [DEBUG] {layer_name}: Storing grad {grad.shape}")
                            # Store grad on original device, offload handled in compute_saliency
                            self.attention_grads[layer_name] = grad.detach()

                    if layer_name not in self._tensor_grad_hooks: # Avoid re-registering
                         handle = attn_weights_tensor.register_hook(_capture_tensor_grad)
                         self._tensor_grad_hooks[layer_name] = handle
                         # print(f"  [DEBUG] {layer_name}: Registered tensor hook.")
        return module_backward_hook

    # --- register_hooks (Use robust version) ---
    def register_hooks(self, model: torch.nn.Module, layer_names: List[str]):
        self.clear_hooks() # Clear only hooks
        self._hooked_layers = set(layer_names)
        print(f"[DEBUG] OldStyle Registering hooks for layers: {list(self._hooked_layers)}")
        # (Loop and register forward/backward hooks as before)
        registered_fwd = 0; registered_bwd = 0
        for layer_name in layer_names:
             module = None # Get module
             try: names = layer_name.split('.'); module = model; # ... (rest of get module) ...
             except Exception: module = None
             if module is not None:
                 f_handle = module.register_forward_hook(self._create_forward_hook_fn(layer_name)); self._forward_hooks.append(f_handle); registered_fwd+=1
                 b_handle = module.register_full_backward_hook(self._create_backward_hook_fn(layer_name)); self._backward_hooks.append(b_handle); registered_bwd+=1
             # else: print(...) # Optional warning
        # print(f"Registered {registered_fwd} fwd / {registered_bwd} bwd hooks.")
        return self # Return self as expected by old workflow

    # *** ADDED: compute_saliency method as called by old workflow ***
    def compute_saliency(self) -> Dict[str, torch.Tensor]:
        """
        Computes saliency = |attention * gradient| for layers where both were captured.
        Does NOT automatically clear the captured weights/grads here.
        """
        saliency_scores: Dict[str, torch.Tensor] = {}
        # Process layers that have BOTH weights and grads stored from the *last* backward
        processable_layers = list(self.attention_weights.keys() & self.attention_grads.keys())

        # Filter based on layers hooked in the *current* cycle for safety
        processable_layers = [ln for ln in processable_layers if ln in self._hooked_layers]

        # print(f"[DEBUG] compute_saliency: Will compute for {len(processable_layers)} layers.") # Debug
        if not processable_layers:
             # Don't clear anything here, let the workflow manage clearing after the step
             return {}

        calculated_count = 0
        for layer_name in processable_layers:
            attn_weights = self.attention_weights[layer_name]
            grad = self.attention_grads[layer_name]

            # (Device/Shape checks as before)
            if attn_weights.device != grad.device: 
                try: grad=grad.to(attn_weights.device) 
                except Exception: continue
            if attn_weights.shape != grad.shape: continue

            try:
                saliency = torch.abs(attn_weights.float() * grad.float()).detach()
                if self.cpu_offload: # Use the single offload flag
                    saliency = saliency.cpu()
                saliency_scores[layer_name] = saliency
                calculated_count += 1
            except Exception as e: print(f"Calc error {layer_name}: {e}")

        # print(f"Computed saliency for {calculated_count} layers.") # Debug
        # *** DO NOT CLEAR self.attention_weights or self.attention_grads here ***
        # Let the caller manage clearing the overall object state if needed.
        return saliency_scores

    # --- clear_hooks (remains the same) ---
    def clear_hooks(self):
        # print(f"Removing {len(self._forward_hooks)} Fwd, {len(self._backward_hooks)} Bwd, {len(self._tensor_grad_hooks)} Tensor hooks...")
        for handle in self._forward_hooks: handle.remove()
        for handle in self._backward_hooks: handle.remove()
        for handle in self._tensor_grad_hooks.values(): handle.remove()
        self._forward_hooks = []; self._backward_hooks = []; self._tensor_grad_hooks = {}
        self._hooked_layers = set()

    # Add explicit clear_cache if needed by workflow
    def clear_cache(self):
        self.attention_weights = {}
        self.attention_grads = {}
        gc.collect()

    # Add clear method if workflow really needs it
    def clear(self):
         self.clear_hooks()
         self.clear_cache()

    def requires_gradient(self) -> bool: return True
    def __del__(self): self.clear_hooks()
    def __enter__(self): return self
    def __exit__(self,et,ev,tb): self.clear_hooks()