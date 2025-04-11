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
# Implicitly implements HookManager if all methods are defined
class GradientAttentionCapture:
    """
    Captures attention weights (forward pass) and their gradients (backward pass).
    REQUIRES gradients to be computed during the model execution.
    Implements the HookManager interface implicitly.
    """
    def __init__(self, cpu_offload_grads: bool = True):
        """
        Initializes the GradientAttentionCapture.

        Args:
            cpu_offload_grads (bool): If True, move captured gradient tensors to CPU
                                      immediately after capture. Defaults to True.
                                      Attention weights stay on original device until get_captured_data.
        """
        # Stores attention weights tensor (requires_grad=True ideally) from forward pass
        self.attention_weights: Dict[str, torch.Tensor] = {}
        # Stores detached gradient tensors captured during backward pass
        self.attention_grads: Dict[str, torch.Tensor] = {}
        self._forward_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._backward_hooks: List[torch.utils.hooks.RemovableHandle] = []
        # Stores handles for hooks attached *directly to tensors* during backward pass
        self._tensor_grad_hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        self._hooked_layers: Set[str] = set()
        self.cpu_offload_grads = cpu_offload_grads
        print(f"Initialized GradientAttentionCapture (Offload Grads: {self.cpu_offload_grads})")

    def _create_forward_hook_fn(self, layer_name: str) -> Callable:
        """Factory for the forward hook: captures attention weights *without detaching*."""
        def forward_hook(module: nn.Module, input_args: Tuple[Any, ...], output: Any):
            attn_weights: Optional[torch.Tensor] = None
            # Try to find attention weights in output tuple (common HF patterns)
            if isinstance(output, tuple):
                if len(output) > 1 and isinstance(output[1], torch.Tensor) and output[1].ndim == 4:
                    attn_weights = output[1]
                elif len(output) > 2 and isinstance(output[2], torch.Tensor) and output[2].ndim == 4:
                    attn_weights = output[2]

            if attn_weights is not None:
                # Warn if the captured attention tensor doesn't require gradients
                if not attn_weights.requires_grad:
                    print(f"Warning (GradCapture): Attention weights tensor for layer '{layer_name}' does not require grad. Gradients may not flow.")
                # Store the original tensor (still attached to graph)
                self.attention_weights[layer_name] = attn_weights
            # else: # Reduce verbosity
            #     print(f"GradCapture Warn: No attention weights found for grad capture layer '{layer_name}'.")
        return forward_hook

    def _create_backward_hook_fn(self, layer_name: str) -> Callable:
        """Factory for the *module* backward hook: registers a *tensor* grad hook."""
        def module_backward_hook(module: nn.Module, grad_input: Tuple[torch.Tensor, ...], grad_output: Tuple[torch.Tensor, ...]):
            # Check if we captured weights for this layer in the forward pass
            if layer_name in self.attention_weights:
                attn_weights_tensor = self.attention_weights[layer_name]

                # Check if tensor requires grad and if we haven't already registered a hook for it in *this* backward pass
                if attn_weights_tensor.requires_grad and layer_name not in self._tensor_grad_hooks:

                    # Define the function that will be called when the gradient for *this tensor* is computed
                    def _capture_tensor_grad(grad: torch.Tensor):
                        # Process and store the gradient
                        processed_grad = grad.detach()
                        if self.cpu_offload_grads:
                            processed_grad = processed_grad.cpu()
                        self.attention_grads[layer_name] = processed_grad

                        # --- Crucial Cleanup ---
                        # Remove the handle for *this specific tensor hook* after it fires.
                        # This prevents it from firing again if backward() is called multiple times
                        # on the same graph for some reason, or storing outdated grads.
                        if layer_name in self._tensor_grad_hooks:
                            # self._tensor_grad_hooks[layer_name].remove() # Potentially problematic if hooks list changes?
                            del self._tensor_grad_hooks[layer_name] # Safer: just remove our reference
                            # print(f"Debug: Tensor hook fired and removed for {layer_name}") # Optional debug

                    # Register the hook directly on the attention weights tensor
                    # This hook fires when the gradient *for this tensor* is computed
                    handle = attn_weights_tensor.register_hook(_capture_tensor_grad)
                    # Store the handle so we can potentially remove it later if backward isn't called
                    self._tensor_grad_hooks[layer_name] = handle
                    # print(f"Debug: Tensor hook registered for {layer_name}") # Optional debug
        return module_backward_hook

    # --- HookManager Interface Methods ---

    def register_hooks(self, model: nn.Module, layer_names: List[str]):
        """Registers forward and backward hooks on specified module layers."""
        self.clear_hooks() # Clear previous hooks (module and tensor)
        # Do NOT clear cache here, attention_weights are needed for the backward pass
        self._hooked_layers = set(layer_names)
        # print(f"GradientAttentionCapture: Registering hooks for {len(layer_names)} layers...")

        hook_count = 0
        not_found_count = 0
        for name in layer_names:
            module = get_module_by_name(model, name)
            if module and isinstance(module, nn.Module):
                # Register forward hook (captures weights)
                f_handle = module.register_forward_hook(self._create_forward_hook_fn(name))
                self._forward_hooks.append(f_handle)
                # Register *module* backward hook (which in turn registers tensor hook)
                # Use register_full_backward_hook for compatibility and access to grad_input/grad_output
                b_handle = module.register_full_backward_hook(self._create_backward_hook_fn(name))
                self._backward_hooks.append(b_handle)
                hook_count += 1
            else:
                print(f"Warning (GradCapture): Module '{name}' not found.")
                not_found_count += 1

        # print(f"GradientAttentionCapture: Registered {hook_count * 2} module hooks.")
        if not_found_count > 0:
            print(f"Warning (GradCapture): Could not find {not_found_count} out of {len(layer_names)} requested modules.")
        return self # Allow chaining

    def requires_gradient(self) -> bool:
        """GradientAttentionCapture requires gradients to function."""
        return True

    def get_captured_data(self) -> Dict[str, Any]:
        """
        Returns captured attention weights (detached) and gradients.
        Gradients are typically available only after `loss.backward()` has been called.

        IMPORTANT: This method now also clears the internal caches (`attention_weights`, `attention_grads`)
                   after retrieving the data, making it suitable for step-wise analysis where
                   each step requires fresh capture.
        """
        # Return detached weights for safety, grads are already detached during capture
        data = {
            "attention_weights": {k: v.detach() for k, v in self.attention_weights.items()},
            "attention_grads": self.attention_grads # Gradients are already detached
        }
        # --- Crucial for Step-wise ---
        # Clear internal caches after retrieving data to prepare for the next step.
        self.clear_cache()
        # ---
        return data

    def clear_hooks(self):
        """Removes all registered module forward/backward hooks AND any pending tensor grad hooks."""
        hooks_removed = 0
        # Remove module forward hooks
        for handle in self._forward_hooks: handle.remove(); hooks_removed += 1
        # Remove module backward hooks
        for handle in self._backward_hooks: handle.remove(); hooks_removed += 1
        # Remove any tensor grad hooks that might not have fired (e.g., if backward wasn't called)
        tensor_hooks_removed = 0
        for handle in self._tensor_grad_hooks.values(): handle.remove(); tensor_hooks_removed += 1

        # Clear internal lists/dictionaries
        self._forward_hooks, self._backward_hooks, self._tensor_grad_hooks = [], [], {}
        self._hooked_layers = set()
        # if hooks_removed > 0 or tensor_hooks_removed > 0:
        #     print(f"GradientAttentionCapture: Removed {hooks_removed} module hooks and {tensor_hooks_removed} pending tensor hooks.")

    def clear_cache(self):
        """Clears stored attention weights and gradients."""
        # print("GradientAttentionCapture: Clearing cached weights and gradients...")
        self.attention_weights, self.attention_grads = {}, {}
        # gc.collect() # Optional GC

    def clear(self):
        """Clears both hooks and cached data."""
        self.clear_hooks()
        self.clear_cache()

    # --- Context Manager Methods ---
    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit context manager: automatically removes hooks."""
        self.clear_hooks()

    # Destructor to ensure hooks are removed when object is garbage collected
    def __del__(self):
        """Attempt to clear hooks upon object deletion."""
        self.clear_hooks()