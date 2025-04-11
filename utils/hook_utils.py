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
    Implements the HookManager protocol.
    Uses forward hooks to store attention weights and backward hooks to capture gradients.
    """
    def __init__(self, cpu_offload_grads: bool = False):
        """
        Initializes the gradient capturer.

        Args:
            cpu_offload_grads (bool): If True, move captured gradients to CPU immediately
                                      after capture to save GPU memory. Defaults to False.
        """
        self.attention_weights: Dict[str, torch.Tensor] = {} # Stores attention weights from forward pass
        self.attention_grads: Dict[str, torch.Tensor] = {}   # Stores gradients w.r.t attention weights
        self._forward_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._backward_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._tensor_grad_hooks: List[torch.utils.hooks.RemovableHandle] = [] # Track tensor-specific grad hooks
        self.cpu_offload_grads = cpu_offload_grads
        print(f"Initialized GradientAttentionCapture (Offload Grads: {self.cpu_offload_grads})")

    def _forward_hook_fn(self, layer_name: str):
        """Creates a forward hook to STORE the attention weights tensor."""
        def hook(module, input, output):
            attn_weights = None
            # Common patterns for attention weights in Hugging Face models
            if isinstance(output, tuple):
                # Pattern 1: (hidden_state, present_key_value, attention_weights)
                if len(output) > 2 and isinstance(output[2], torch.Tensor) and output[2].ndim >= 4:
                   if output[2].shape[-1] == output[2].shape[-2]: attn_weights = output[2]
                # Pattern 2: (hidden_state, attention_weights, present_key_value)
                elif len(output) > 1 and isinstance(output[1], torch.Tensor) and output[1].ndim >= 4:
                     if output[1].shape[-1] == output[1].shape[-2]: attn_weights = output[1]

            # Add other checks if needed (e.g., for different model output structures)

            # Store the tensor IF found AND it requires grad (needed for backward)
            if attn_weights is not None and attn_weights.requires_grad:
                # Store the *original* tensor, do not detach here!
                self.attention_weights[layer_name] = attn_weights
                # print(f"DEBUG Forward Hook {layer_name}: Stored attn weights {attn_weights.shape}") # DEBUG
            # elif attn_weights is not None:
            #     print(f"DEBUG Forward Hook {layer_name}: Found attn weights {attn_weights.shape} but requires_grad=False") # DEBUG
            # else:
            #     print(f"DEBUG Forward Hook {layer_name}: Could not find suitable attn weights in output.") # DEBUG

        return hook

    def _backward_hook_fn(self, layer_name: str):
        """Creates a backward hook (for the *module*) that registers a *tensor* hook."""
        def hook(module, grad_input, grad_output):
            # Check if we stored attention weights for this layer in the forward pass
            if layer_name in self.attention_weights:
                attn_weights_tensor = self.attention_weights[layer_name]

                # Define the function that captures the gradient for the *tensor*
                def _capture_grad(grad):
                    if grad is not None:
                        # print(f"DEBUG Tensor Grad Hook {layer_name}: Captured grad {grad.shape}") # DEBUG
                        # Detach and optionally move to CPU
                        processed_grad = grad.detach()
                        if self.cpu_offload_grads:
                            processed_grad = processed_grad.cpu()
                        self.attention_grads[layer_name] = processed_grad
                    # else:
                        # print(f"DEBUG Tensor Grad Hook {layer_name}: Received None gradient.") # DEBUG

                # CRITICAL: Register the capture function directly on the stored attention tensor
                # This hook will execute when the gradient for *this specific tensor* is computed
                if attn_weights_tensor.requires_grad:
                   handle = attn_weights_tensor.register_hook(_capture_grad)
                   self._tensor_grad_hooks.append(handle) # Track this tensor hook
                # else:
                #    print(f"DEBUG Backward Hook {layer_name}: attn_weights tensor does not require grad, cannot register tensor hook.") # DEBUG

            # else:
            #     print(f"DEBUG Backward Hook {layer_name}: No attn weights stored, skipping tensor hook registration.") # DEBUG

        return hook

    def register_hooks(self, model: torch.nn.Module, layer_names: List[str]) -> None:
        """Registers forward and backward hooks on specified layers."""
        self.clear() # Ensure clean state before registering new hooks
        if not layer_names:
            print("GradientAttentionCapture: No layer names provided to register hooks.")
            return

        for layer_name in layer_names:
            module = None
            try:
                # Helper to get module by name (assumed to exist in model_utils)
                from utils.model_utils import get_module_by_name
                module = get_module_by_name(model, layer_name)
            except ImportError:
                 print("Warning: Cannot import get_module_by_name from utils.model_utils")
                 # Basic fallback (might not work for nested modules)
                 try: module = dict(model.named_modules())[layer_name]
                 except KeyError: module = None
            except Exception as e:
                 print(f"Error getting module '{layer_name}': {e}")
                 module = None

            if module is not None:
                # Register forward hook to store attention weights
                f_handle = module.register_forward_hook(self._forward_hook_fn(layer_name))
                self._forward_hooks.append(f_handle)
                # Register backward hook to attach gradient hook to the tensor
                b_handle = module.register_full_backward_hook(self._backward_hook_fn(layer_name))
                self._backward_hooks.append(b_handle)
            else:
                print(f"Warning: Module '{layer_name}' not found in model. Cannot register hooks.")
        print(f"GradientAttentionCapture: Registered {len(self._forward_hooks)} forward and {len(self._backward_hooks)} backward hooks.")


    def get_captured_data(self) -> Dict[str, Any]:
        """Returns captured attention weights and gradients, then clears internal storage."""
        # Return copies to prevent external modification
        data = {
            "attention_weights": self.attention_weights.copy(),
            "attention_grads": self.attention_grads.copy()
        }
        # print(f"DEBUG get_captured_data: Returning {len(data['attention_weights'])} weights, {len(data['attention_grads'])} grads.") # DEBUG
        # Clear internal storage after retrieval
        self.attention_weights.clear()
        self.attention_grads.clear()
        # Also clear tensor grad hooks as they are tied to the specific backward pass
        self._clear_tensor_grad_hooks()
        return data

    def _clear_tensor_grad_hooks(self) -> None:
        """Removes only the tensor-specific gradient hooks."""
        # print(f"GradientAttentionCapture: Removing {len(self._tensor_grad_hooks)} tensor grad hooks.") # DEBUG
        for handle in self._tensor_grad_hooks:
            handle.remove()
        self._tensor_grad_hooks = []


    def clear_hooks(self) -> None:
        """Removes all registered forward, backward, and tensor hooks."""
        # print(f"GradientAttentionCapture: Removing {len(self._forward_hooks)} forward hooks.") # DEBUG
        for handle in self._forward_hooks: handle.remove()
        self._forward_hooks = []

        # print(f"GradientAttentionCapture: Removing {len(self._backward_hooks)} backward hooks.") # DEBUG
        for handle in self._backward_hooks: handle.remove()
        self._backward_hooks = []

        # Ensure tensor hooks are also cleared
        self._clear_tensor_grad_hooks()

    def clear(self) -> None:
        """Clears captured data and removes all hooks."""
        self.clear_hooks()
        self.attention_weights = {}
        self.attention_grads = {}
        gc.collect() # Add garbage collection on full clear
        # print("GradientAttentionCapture: Cleared data and all hooks.")


    def requires_gradient(self) -> bool:
        """GradientAttentionCapture requires gradients."""
        return True

    def __del__(self):
        """Ensure hooks are removed when the object is deleted."""
        self.clear_hooks()

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_value, traceback): self.clear()