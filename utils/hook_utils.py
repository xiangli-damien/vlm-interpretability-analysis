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


# --- Activation Cache Implementation ---
# Implicitly implements HookManager if all methods are defined
class ActivationCache:
    """
    Captures activations (hidden states) and optionally attention weights
    using forward hooks. Does NOT require gradients.
    Implements the HookManager interface implicitly.
    """
    def __init__(self, cpu_offload: bool = True, capture_activations: bool = True, capture_attentions: bool = False):
        """
        Initializes the ActivationCache.

        Args:
            cpu_offload (bool): Move captured tensors to CPU and detach. Defaults to True.
            capture_activations (bool): Whether to capture hidden states/activations. Defaults to True.
            capture_attentions (bool): Whether to capture attention weights. Defaults to False.
        """
        self.activations: Dict[str, torch.Tensor] = {}
        self.attentions: Dict[str, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.cpu_offload = cpu_offload
        self.capture_activations = capture_activations
        self.capture_attentions = capture_attentions
        self._layers_hooked: Set[str] = set()
        print(f"Initialized ActivationCache (Capture Activations: {self.capture_activations}, Capture Attentions: {self.capture_attentions}, CPU Offload: {self.cpu_offload})")

    def _create_hook_fn(self, layer_name: str) -> Callable:
        """Factory function to create the specific forward hook callback for a layer."""
        def hook(module: nn.Module, input_args: Tuple[Any, ...], output: Any):
            # --- Capture Activations (Hidden State) ---
            if self.capture_activations:
                hidden_state: Optional[torch.Tensor] = None
                # Try to extract the primary tensor output (often the hidden state)
                if isinstance(output, torch.Tensor):
                    hidden_state = output
                # Handle cases where output is a tuple (e.g., (hidden_state, ...))
                elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                    hidden_state = output[0]
                # Add more checks here if models have different output structures

                if hidden_state is not None:
                    # Process the captured hidden state
                    processed_hs = hidden_state.detach() # Detach from computation graph
                    if self.cpu_offload:
                        processed_hs = processed_hs.cpu() # Move to CPU if requested
                    self.activations[layer_name] = processed_hs
                # else: # Reduce verbosity for missing states
                #     print(f"ActivationCache Warn: No hidden state tensor found for layer '{layer_name}' output type {type(output)}.")

            # --- Capture Attention Weights ---
            if self.capture_attentions:
                attn_weights: Optional[torch.Tensor] = None
                # Common patterns for attention weights in Hugging Face model outputs:
                # (hidden_state, attn_weights, ...)
                # (hidden_state, present_key_value, attn_weights, ...)
                if isinstance(output, tuple):
                    # Check second element if it looks like attention weights (typically 4D: B, H, S, S)
                    if len(output) > 1 and isinstance(output[1], torch.Tensor) and output[1].ndim == 4:
                        attn_weights = output[1]
                    # Check third element if second element might be KV cache
                    elif len(output) > 2 and isinstance(output[2], torch.Tensor) and output[2].ndim == 4:
                        attn_weights = output[2]
                    # Add more checks for different model families if needed

                if attn_weights is not None:
                    # Process the captured attention weights
                    processed_attn = attn_weights.detach() # Detach from graph
                    if self.cpu_offload:
                        processed_attn = processed_attn.cpu() # Move to CPU if requested
                    self.attentions[layer_name] = processed_attn
                # else: # Reduce verbosity for missing attention
                #     print(f"ActivationCache Warn: No attention weights tensor found for layer '{layer_name}' output type {type(output)}.")
        return hook

    # --- HookManager Interface Methods ---

    def register_hooks(self, model: nn.Module, layer_names: List[str]):
        """Registers forward hooks on the specified layers."""
        self.clear() # Clear previous hooks and cache before registering new ones
        self._layers_hooked = set(layer_names)
        # print(f"ActivationCache: Registering forward hooks for {len(layer_names)} layers...")

        hook_count = 0
        not_found_count = 0
        for name in layer_names:
            module = get_module_by_name(model, name)
            if module:
                # Create and register the hook for this specific module
                hook_fn = self._create_hook_fn(name)
                handle = module.register_forward_hook(hook_fn)
                self._hooks.append(handle) # Store the handle to remove later
                hook_count += 1
            else:
                print(f"Warning (ActivationCache): Module '{name}' not found in model.")
                not_found_count += 1

        # print(f"ActivationCache: Registered {hook_count} hooks.")
        if not_found_count > 0:
            print(f"Warning (ActivationCache): Could not find {not_found_count} out of {len(layer_names)} requested modules.")
        return self # Allow chaining

    def get_captured_data(self) -> Dict[str, Any]:
        """Returns captured activations and/or attentions based on initialization config."""
        data = {}
        if self.capture_activations:
            data["activations"] = self.activations
        if self.capture_attentions:
            data["attentions"] = self.attentions
        # NOTE: We do *not* clear the cache here automatically.
        # The calling function (e.g., engine's stepwise loop) should call clear_cache()
        # when appropriate for the analysis step.
        return data

    def requires_gradient(self) -> bool:
        """ActivationCache does not require gradients."""
        return False

    def clear_hooks(self):
        """Removes all registered forward hooks."""
        if not self._hooks: return # Nothing to remove
        # print(f"ActivationCache: Removing {len(self._hooks)} hooks...")
        for handle in self._hooks:
            handle.remove() # Use the handle to remove the hook
        self._hooks = [] # Clear the list of handles
        self._layers_hooked = set() # Clear the set of hooked layers

    def clear_cache(self):
        """Clears stored activation and attention data from internal dictionaries."""
        # print("ActivationCache: Clearing cached data...")
        self.activations = {}
        self.attentions = {}
        # Optionally trigger garbage collection, but might be overkill here
        # gc.collect()

    def clear(self):
        """Clears both hooks and cached data. Recommended before reuse."""
        self.clear_hooks()
        self.clear_cache()

    # --- Context Manager Methods (Optional but convenient) ---
    # Allows using 'with ActivationCache(...) as cache:' syntax
    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit context manager: automatically removes hooks."""
        self.clear_hooks()


# --- Gradient Attention Capture Implementation ---
class GradientAttentionCapture:
    """
    This class captures the gradients flowing through attention modules using forward
    and full backward hooks. It stores both the attention weights and their gradients,
    and it can be configured to offload gradients to CPU.
    
    Modification:
    In the forward hook, if the attention weights do not have grad enabled, we force them
    to require gradients (by calling `requires_grad_()`). This ensures that when we call
    backward(), the gradients are computed so that the hook can capture them.
    """
    def __init__(self, cpu_offload_grads: bool = False):
        self.attention_weights = {}  # Mapping: layer name -> attention weight tensor
        self.attention_grads = {}    # Mapping: layer name -> gradient tensor (captured)
        self.forward_hooks = []      # Store forward hook handles
        self.backward_hooks = []     # Store backward hook handles
        self.cpu_offload_grads = cpu_offload_grads

    def _forward_hook_fn(self, layer_name: str):
        """
        Creates and returns a forward hook function that attempts to extract the attention weights.
        If the tensor does not have gradients enabled, it forces it (via requires_grad_()).
        """
        def hook(module, input, output):
            # Try to extract the attention weights from typical output positions.
            attn_weights = None
            if isinstance(output, tuple):
                # Some models return attention as the second element (e.g., output[1])
                if len(output) >= 2 and isinstance(output[1], torch.Tensor):
                    attn_weights = output[1]
                # Alternatively, check the third element.
                elif len(output) >= 3 and isinstance(output[2], torch.Tensor):
                    attn_weights = output[2]
            elif isinstance(output, torch.Tensor):
                attn_weights = output

            if attn_weights is not None:
                # IMPORTANT: Ensure that the attention weights require gradients
                if not attn_weights.requires_grad:
                    # Force the tensor to require gradients so that backward() can compute them.
                    attn_weights.requires_grad_()
                    # Optionally, call retain_grad() so that even if the tensor is not a leaf it will be retained.
                    attn_weights.retain_grad()
                # Store a detached copy of the attention weights (for reference)
                self.attention_weights[layer_name] = attn_weights.detach()
        return hook

    def _backward_hook_fn(self, layer_name: str):
        """
        Creates and returns a backward hook function that, when the gradient is computed,
        captures it and stores it in the attention_grads dictionary.
        """
        def hook(module, grad_input, grad_output):
            if layer_name not in self.attention_weights:
                return
            attn_weights = self.attention_weights[layer_name]
            def _capture_grad(grad):
                grad_detached = grad.detach()
                if self.cpu_offload_grads:
                    grad_detached = grad_detached.cpu()
                self.attention_grads[layer_name] = grad_detached
            # Register a hook on the attention weights so that when they are backpropagated,
            # the gradient is captured.
            attn_weights.register_hook(_capture_grad)
        return hook

    def register_hooks(self, model: torch.nn.Module, layer_names: list):
        """
        Registers forward and backward hooks on the modules whose names match those in layer_names.
        """
        self.clear_hooks()
        for name, module in model.named_modules():
            if name in layer_names:
                f_hook = module.register_forward_hook(self._forward_hook_fn(name))
                # Use register_full_backward_hook() so that gradients from non-leaf nodes are captured.
                b_hook = module.register_full_backward_hook(self._backward_hook_fn(name))
                self.forward_hooks.append(f_hook)
                self.backward_hooks.append(b_hook)
        return self

    def get_captured_data(self) -> dict:
        """
        Returns a dictionary containing the captured attention weights and gradients.
        It then clears the internal stored hooks and data.
        """
        data = {
            "attention_weights": self.attention_weights.copy(),
            "attention_grads": self.attention_grads.copy()
        }
        self.clear_hooks()
        self.attention_weights.clear()
        self.attention_grads.clear()
        return data

    def clear_hooks(self):
        """
        Removes all registered forward and backward hooks.
        """
        for hook in self.forward_hooks:
            hook.remove()
        for hook in self.backward_hooks:
            hook.remove()
        self.forward_hooks = []
        self.backward_hooks = []