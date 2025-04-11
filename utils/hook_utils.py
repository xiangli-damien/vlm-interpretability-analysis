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
    Captures gradients flowing through attention modules for saliency analysis.
    Includes the saliency computation logic directly, similar to original notebook.
    Implements the HookManager protocol.
    """
    def __init__(self, cpu_offload_grads: bool = False, cpu_offload_saliency: bool = True):
        """
        Initializes the gradient capturer.

        Args:
            cpu_offload_grads (bool): If True, move captured gradients to CPU immediately
                                      after capture. Defaults to False.
            cpu_offload_saliency (bool): If True, move computed saliency scores to CPU
                                         before returning. Defaults to True.
        """
        self.attention_weights: Dict[str, torch.Tensor] = {} # Stores weights from forward
        self.attention_grads: Dict[str, torch.Tensor] = {}   # Stores grads from backward
        self._forward_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._backward_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._tensor_grad_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.cpu_offload_grads = cpu_offload_grads
        self.cpu_offload_saliency = cpu_offload_saliency # <<< STORE THE ARGUMENT
        self._currently_hooked_layers: set[str] = set()
        # Make the init print reflect both options
        print(f"Initialized GradientAttentionCapture (Offload Grads: {self.cpu_offload_grads}, Offload Saliency: {self.cpu_offload_saliency})")



    def _forward_hook_fn(self, layer_name: str):
        """Creates a forward hook to STORE the attention weights tensor."""
        # (Implementation remains the same as previous corrected version)
        def hook(module, input, output):
            attn_weights = None
            if isinstance(output, tuple):
                if len(output) > 2 and isinstance(output[2], torch.Tensor) and output[2].ndim >= 4 and output[2].shape[-1] == output[2].shape[-2]: attn_weights = output[2]
                elif len(output) > 1 and isinstance(output[1], torch.Tensor) and output[1].ndim >= 4 and output[1].shape[-1] == output[1].shape[-2]: attn_weights = output[1]
            if attn_weights is not None and attn_weights.requires_grad:
                self.attention_weights[layer_name] = attn_weights
        return hook

    def _backward_hook_fn(self, layer_name: str):
        """Creates a backward hook (for module) that registers a *tensor* hook."""
        # (Implementation remains the same as previous corrected version)
        def hook(module, grad_input, grad_output):
            if layer_name in self.attention_weights:
                attn_weights_tensor = self.attention_weights[layer_name]
                def _capture_grad(grad):
                    if grad is not None:
                        processed_grad = grad.detach()
                        if self.cpu_offload_grads: processed_grad = processed_grad.cpu()
                        self.attention_grads[layer_name] = processed_grad
                if attn_weights_tensor.requires_grad:
                   handle = attn_weights_tensor.register_hook(_capture_grad)
                   self._tensor_grad_hooks.append(handle)
        return hook

    def register_hooks(self, model: torch.nn.Module, layer_names: List[str]) -> None:
        """Registers forward and backward hooks on specified layers."""
        # Don't clear data here, only hooks from previous registrations
        self.clear_hooks()
        self._currently_hooked_layers = set(layer_names) # Track layers for this cycle

        if not layer_names:
            print("GradientAttentionCapture: No layer names provided to register hooks.")
            return
        # (Rest of the registration logic remains the same as previous corrected version)
        registered_fwd = 0
        registered_bwd = 0
        for layer_name in layer_names:
            module = None
            # ... (try getting module as before) ...
            try: # Simplified get_module_by_name logic
                names = layer_name.split('.'); module = model
                for n in names: module = module[int(n)] if n.isdigit() else getattr(module, n)
            except Exception: module = None

            if module is not None:
                f_handle = module.register_forward_hook(self._forward_hook_fn(layer_name))
                self._forward_hooks.append(f_handle); registered_fwd += 1
                b_handle = module.register_full_backward_hook(self._backward_hook_fn(layer_name))
                self._backward_hooks.append(b_handle); registered_bwd += 1
            else:
                print(f"Warning: Module '{layer_name}' not found. Cannot register hooks.")
        print(f"GradientAttentionCapture: Registered {registered_fwd} fwd / {registered_bwd} bwd hooks for {len(layer_names)} layers.")

    # *** NEW METHOD: Based on original compute_saliency_scores ***
    def compute_saliency_scores_and_clear(self) -> Dict[str, torch.Tensor]:
        """
        Computes saliency = |attention * gradient| for the layers hooked in the
        *current registration cycle* where both weights and grads were captured.
        Clears the captured weights and grads for the processed layers afterwards.

        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping layer names to computed saliency tensors.
                                      Tensors are detached and potentially on CPU.
        """
        saliency_scores: Dict[str, torch.Tensor] = {}
        # Process only the layers that were intended to be hooked in this cycle
        # And for which we actually captured both weights and grads
        processable_layers = list(self._currently_hooked_layers & self.attention_weights.keys() & self.attention_grads.keys())

        if not processable_layers:
            print("Compute Saliency: No layers found with both weights and grads captured for the current cycle.")
            # Still clear any potentially captured data for the hooked layers
            self._clear_data_for_layers(self._currently_hooked_layers)
            self._currently_hooked_layers.clear()
            return {}

        print(f"Compute Saliency: Calculating for {len(processable_layers)} layers...")
        calculated_count = 0
        for layer_name in processable_layers:
            attn_weights = self.attention_weights[layer_name]
            grad = self.attention_grads[layer_name]

            # Device check/move (same as in standalone calculate_saliency_scores)
            if attn_weights.device != grad.device:
                print(f"  Saliency Calc Warning: Device mismatch layer '{layer_name}'! W:{attn_weights.device}, G:{grad.device}. Moving grad.")
                try: grad = grad.to(attn_weights.device)
                except Exception as e: print(f"  Error moving grad: {e}. Skip."); continue
            # Shape check
            if attn_weights.shape != grad.shape:
                 print(f"  Saliency Calc Warning: Shape mismatch layer '{layer_name}'! W:{attn_weights.shape}, G:{grad.shape}. Skip."); continue

            try:
                saliency = torch.abs(attn_weights.float() * grad.float()).detach()
                if self.cpu_offload_saliency:
                    saliency = saliency.cpu()
                saliency_scores[layer_name] = saliency
                calculated_count += 1
            except Exception as e:
                print(f"  Saliency Calc Error: Failed for layer '{layer_name}': {e}. Skip.")

            # *** IMPORTANT: Clean up data for this layer *after* processing ***
            del self.attention_weights[layer_name]
            del self.attention_grads[layer_name]

        print(f"Compute Saliency: Calculated scores for {calculated_count} layers.")
        # Clear any remaining data for layers that were hooked but maybe didn't get grads etc.
        self._clear_data_for_layers(self._currently_hooked_layers - set(processable_layers))
        self._currently_hooked_layers.clear() # Clear the tracking set for the next cycle
        gc.collect() # Add garbage collection here
        return saliency_scores

    def _clear_data_for_layers(self, layer_names_to_clear: set[str]):
        """Helper to remove captured data for specific layers."""
        for layer_name in layer_names_to_clear:
            self.attention_weights.pop(layer_name, None)
            self.attention_grads.pop(layer_name, None)

    # Keep get_captured_data for potential other uses, but it doesn't clear hooks
    def get_captured_data(self) -> Dict[str, Any]:
        """Returns captured attention weights and gradients *without* clearing hooks."""
        # Return copies
        data = {
            "attention_weights": self.attention_weights.copy(),
            "attention_grads": self.attention_grads.copy()
        }
        # DO NOT CLEAR DATA OR HOOKS HERE in this version
        return data

    def _clear_tensor_grad_hooks(self) -> None:
        # (Implementation remains the same)
        for handle in self._tensor_grad_hooks: handle.remove()
        self._tensor_grad_hooks = []

    def clear_hooks(self) -> None:
        # (Implementation remains the same)
        for handle in self._forward_hooks: handle.remove()
        self._forward_hooks = []
        for handle in self._backward_hooks: handle.remove()
        self._backward_hooks = []
        self._clear_tensor_grad_hooks()
        self._currently_hooked_layers.clear() # Also clear the tracking set

    def clear(self) -> None:
        """Clears captured data AND removes all hooks."""
        self.clear_hooks() # Removes hooks and clears _currently_hooked_layers
        self.attention_weights = {}
        self.attention_grads = {}
        gc.collect()
        print("GradientAttentionCapture: Cleared ALL data and hooks.")

    def requires_gradient(self) -> bool: return True
    def __del__(self): self.clear_hooks()
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_value, traceback): self.clear_hooks() # Use clear_hooks on exit
