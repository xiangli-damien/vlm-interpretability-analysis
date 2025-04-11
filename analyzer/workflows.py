# -*- coding: utf-8 -*-
"""
End-to-end analysis workflow functions for Logit Lens and Saliency Analysis.
Includes a memory-optimized saliency workflow.
"""

import torch
import os
import gc
import time
import json
import pickle
from PIL import Image
from typing import Dict, Any, Optional, Union, List, Tuple
from tqdm import tqdm # Added tqdm for progress

# --- Imports from VLM Analysis Project Modules (Absolute Paths) ---
from engine import LLaVANextEngine
from engine.representations import get_logits_from_hidden_states # Needed for LogitLens
from analyzer.logit_lens_analyzer import LogitLensAnalyzer
from analyzer.saliency_analyzer import (
    calculate_saliency_scores,
    analyze_layerwise_saliency_flow
)
try:
    from utils.data_utils import get_token_indices # Keep original import attempt
    from utils.visual_utils import visualize_token_probabilities, visualize_information_flow
    from utils.hook_utils import GradientAttentionCapture
    from utils.model_utils import get_llm_attention_layer_names # Needed here now
except ImportError as e:
    print(f"Warning: Could not import required components in workflows.py: {e}")
    def get_token_indices(*args, **kwargs): return (torch.tensor([]), torch.tensor([]))
    def visualize_token_probabilities(*args, **kwargs): return []
    def visualize_information_flow(*args, **kwargs): pass
    class GradientAttentionCapture: pass
    def get_llm_attention_layer_names(*args, **kwargs): return []


# --- Workflow 1: Logit Lens (Refactored) ---

def run_logit_lens_workflow(
    engine: LLaVANextEngine,                 # Input: Initialized LLaVA-Next engine
    image_source: Union[str, Image.Image],   # Input: Image path, URL, or PIL object
    prompt_text: str,                        # Input: Text prompt for the model
    concepts_to_track: Optional[List[str]] = None, # Input: List of concept strings (e.g., "cat", "sign")
    selected_layers: Optional[List[int]] = None,   # Input: Specific layer indices (0-based) to analyze/visualize. None for all.
    output_dir: str = "logit_lens_analysis", # Input: Directory path to save analysis outputs
    cpu_offload: bool = True                 # Input: Controls CPU offload in get_logits_from_hidden_states
) -> Dict[str, Any]:
    """
    Performs the complete logit lens analysis pipeline (Refactored Workflow).

    1. Prepares inputs using the LLaVANextEngine.
    2. Runs a forward pass via the engine to get all hidden states.
    3. Retrieves the Language Model (LM) head from the engine.
    4. Calls `representations.get_logits_from_hidden_states` to compute probabilities for each layer.
    5. Calls `LogitLensAnalyzer.analyze` (passing the computed probabilities) to extract and structure
       probabilities for tracked concepts based on the feature mapping.
    6. Visualizes the structured concept probabilities.
    7. Saves a summary file.

    Args:
        engine (LLaVANextEngine): The initialized LLaVA-Next engine instance.
        image_source (Union[str, Image.Image]): PIL image, URL string, or local file path.
        prompt_text (str): The text prompt for the model.
        concepts_to_track (Optional[List[str]]): List of concept strings to track probabilities for.
        selected_layers (Optional[List[int]]): Specific layer indices (0-based, relative to hidden_states tuple)
                                               to compute probabilities for and visualize. If None, processes all layers.
        output_dir (str): Directory path to save analysis outputs (visualizations, summary file).
        cpu_offload (bool): If True, attempts to move intermediate probability tensors to CPU during
                            the `get_logits_from_hidden_states` computation to save GPU memory.

    Returns:
        Dict[str, Any]: A dictionary containing analysis results:
                        - 'structured_concept_probabilities': Output from LogitLensAnalyzer.
                        - 'feature_mapping': Mapping from token indices to features (base, patch, etc.).
                        - 'concepts_tracked_ids': Dictionary mapping tracked concept strings to their token IDs.
                        - 'visualization_paths': List of paths to saved visualization files.
                        - 'summary_path': Path to the saved summary text file.
                        Includes an 'error' key with an error message string on failure.
    """
    print("\n--- Starting LLaVA-Next Logit Lens Workflow (Refactored) ---")
    print(f"  Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    # Initialize results dictionary with a default error state
    results = {"error": "Workflow did not initialize properly."}

    # --- Basic Validation ---
    if engine.model is None or engine.processor is None:
        results["error"] = "Engine is not properly initialized (model or processor is None)."
        print(f"Error: {results['error']}")
        return results
    if not concepts_to_track:
         print("Warning: No concepts_to_track provided. Logit Lens analysis will not extract specific concept probabilities.")
         # Allow workflow to proceed, but analyzer step might be skipped or return empty.


    try:
        # --- 1. Prepare Inputs & Feature Mapping using the Engine ---
        print("  Step 1: Preparing inputs and feature mapping...")
        prepared_data = engine.build_inputs(image_source, prompt_text)
        # Validate prepared data structure
        if not prepared_data or "inputs" not in prepared_data or "feature_mapping" not in prepared_data:
            results["error"] = "Input preparation via engine failed or returned unexpected structure."
            print(f"Error: {results['error']}")
            return results
        # Store feature mapping for later use by analyzer and visualization
        results["feature_mapping"] = prepared_data["feature_mapping"]
        print("  Inputs prepared successfully.")

        # --- 2. Run Forward Pass to Get Hidden States ---
        print("  Step 2: Running forward pass to get hidden states...")
        # Get expected number of layers from engine config
        num_layers = engine.get_config("num_llm_layers")
        if not isinstance(num_layers, int) or num_layers <= 0:
            results["error"] = f"Could not determine a valid number of LLM layers from engine config ({num_layers})."
            print(f"Error: {results['error']}")
            return results

        # Perform forward pass requesting hidden states
        forward_outputs = engine.forward(prepared_data['inputs'], output_hidden_states=True)
        hidden_states = forward_outputs.get("hidden_states")

        # Validate hidden states structure (expecting num_layers + 1 states including embeddings)
        if not hidden_states or not isinstance(hidden_states, (list, tuple)) or len(hidden_states) != num_layers + 1:
            results["error"] = f"Failed to get expected number of hidden states ({num_layers+1}). Got: {len(hidden_states) if hidden_states else 0}"
            print(f"Error: {results['error']}")
            # Clean up potentially large tensor tuple/list
            del hidden_states
            if 'forward_outputs' in locals(): del forward_outputs
            gc.collect(); torch.cuda.empty_cache()
            return results
        print(f"  Successfully retrieved {len(hidden_states)} hidden states.")

        # --- 3. Prepare Concepts Dictionary (Token IDs) ---
        print("  Step 3: Preparing concept token IDs...")
        concept_token_ids = {}
        concepts_actually_tracked = []
        vocab_size = engine.get_config("vocab_size")
        if vocab_size is None or vocab_size <= 0:
            print("Warning: Could not get valid vocab size from engine. Concept token ID validation may be skipped.")

        if concepts_to_track:
            tokenizer = engine.get_processor().tokenizer
            for concept in concepts_to_track:
                try:
                    # Encode concept string to token IDs
                    token_ids = tokenizer.encode(concept, add_special_tokens=False)
                    # Validate IDs against vocab size if available
                    valid_ids = [tid for tid in token_ids if 0 <= tid < vocab_size] if vocab_size > 0 else token_ids
                    if valid_ids:
                        concept_token_ids[concept] = valid_ids
                        concepts_actually_tracked.append(concept)
                    else:
                        print(f"  Warning: Concept '{concept}' yielded no valid token IDs after encoding and validation.")
                except Exception as e:
                    print(f"  Warning: Error encoding concept '{concept}': {e}")
        # Store the dictionary mapping concept strings to their valid token IDs
        results["concepts_tracked_ids"] = concept_token_ids
        print(f"  Prepared token IDs for concepts: {concepts_actually_tracked}")

        # --- 4. Compute Layer Probabilities using Representations Module ---
        print("  Step 4: Computing layer probabilities via representations module...")
        # Get the language model head required for projection
        lm_head = engine.get_lm_head()
        if not lm_head:
            results["error"] = "Could not get LM head from engine. Cannot compute probabilities."
            print(f"Error: {results['error']}")
            del hidden_states # Cleanup
            if 'forward_outputs' in locals(): del forward_outputs
            gc.collect(); torch.cuda.empty_cache()
            return results

        # Call the dedicated function to project hidden states and get probabilities
        layer_probabilities = get_logits_from_hidden_states(
            hidden_states=hidden_states,
            lm_head=lm_head,
            layers_to_process=selected_layers, # Pass user-selected layers (or None for all)
            cpu_offload=cpu_offload,           # Control offloading during this step
            use_float32_for_softmax=True       # Ensure stable softmax
        )

        # Validate the output
        if not layer_probabilities:
             print("Warning: `get_logits_from_hidden_states` returned empty results. LogitLens analysis might be incomplete.")
             # Proceed, but analyzer step will likely do nothing

        # Explicitly delete hidden_states now as they are no longer needed
        del hidden_states
        if 'forward_outputs' in locals(): del forward_outputs
        gc.collect(); torch.cuda.empty_cache()
        print(f"  Computed probabilities for {len(layer_probabilities)} layers.")


        # --- 5. Analyze Concept Probabilities using LogitLensAnalyzer ---
        structured_concept_probs = {} # Initialize
        if not concept_token_ids:
            print("  Step 5: Skipping concept probability extraction (no valid concepts to track).")
        elif not layer_probabilities:
            print("  Step 5: Skipping concept probability extraction (no layer probabilities computed).")
        else:
            print("  Step 5: Extracting and structuring concept probabilities via LogitLensAnalyzer...")
            analyzer = LogitLensAnalyzer()
            # Pass the pre-computed probabilities to the refactored analyzer
            structured_concept_probs = analyzer.analyze(
                layer_probabilities=layer_probabilities, # Pass the computed probabilities
                feature_mapping=results["feature_mapping"],
                vocab_size=vocab_size if vocab_size > 0 else -1,
                concepts_to_track=concept_token_ids,
                # No cpu_offload needed here as analyzer expects data on target device (CPU)
            )
            print("  Concept probability extraction complete.")

        # Store the structured results from the analyzer
        results["structured_concept_probabilities"] = structured_concept_probs

        # Explicitly delete layer_probabilities dictionary to free memory
        del layer_probabilities
        gc.collect()


        # --- 6. Visualize Probabilities ---
        viz_paths = [] # Initialize
        if not structured_concept_probs:
            print("  Step 6: Skipping visualization (no structured concept probabilities available).")
        else:
            print("  Step 6: Calling visualization function...")
            # Prepare data needed specifically for visualization
            viz_input_data = {
                "feature_mapping": results["feature_mapping"],
                "original_image": prepared_data["original_image"],
                "spatial_preview_image": prepared_data["spatial_preview_image"],
                "prompt_text": prompt_text
            }
            try:
                viz_paths = visualize_token_probabilities(
                    token_probs=structured_concept_probs, # Pass the structured results
                    input_data=viz_input_data,
                    # Pass selected_layers to control which layers are plotted
                    # If selected_layers was None, visualization might plot all available layers in structured_concept_probs
                    selected_layers=selected_layers,
                    output_dir=output_dir
                )
                print(f"  Visualization complete. {len(viz_paths)} plots generated.")
            except Exception as viz_err:
                 print(f"  Error during visualization: {viz_err}")
                 # Continue workflow, but report visualization failure
        results["visualization_paths"] = viz_paths


        # --- 7. Save Summary File ---
        summary_path = os.path.join(output_dir, "analysis_summary.txt")
        print(f"  Step 7: Saving analysis summary to: {summary_path}")
        try:
            # Write summary details to a text file
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("LLaVA-Next Logit Lens Analysis Summary (Refactored Workflow)\n" + "=" * 60 + "\n\n")
                # Model and Input Info
                f.write(f"Model ID: {engine.model_id}\n")
                img_src_repr = image_source if isinstance(image_source, str) else f"PIL Image ({prepared_data['original_image'].size} {prepared_data['original_image'].mode})"
                f.write(f"Image Source: {img_src_repr}\n")
                f.write(f"Prompt: {prompt_text}\n\n")

                # Tracked Concepts
                f.write(f"Concepts Tracked: {concepts_actually_tracked}\n")
                tracked_ids = results.get("concepts_tracked_ids", {})
                if tracked_ids:
                    for concept, ids in tracked_ids.items():
                        f.write(f"  - '{concept}': {ids}\n")
                else:
                    f.write("  (No concepts requested or none had valid token IDs)\n")

                # Image Processing Info (from feature mapping)
                f.write("\nImage Processing Information (from feature_mapping):\n")
                fm = results.get("feature_mapping", {})
                if fm:
                    orig_w, orig_h = fm.get('original_size', ('N/A', 'N/A'))
                    f.write(f"  Original Size (WxH): ({orig_w}, {orig_h})\n")
                    resized_w, resized_h = fm.get('resized_dimensions', ('N/A','N/A'))
                    f.write(f"  Resized Size (WxH): ({resized_w}, {resized_h})\n")
                    padded_w, padded_h = fm.get('padded_dimensions', ('N/A', 'N/A'))
                    f.write(f"  Padded Preview Size (WxH): ({padded_w}, {padded_h})\n")
                    f.write(f"  Best Resolution Target (WxH): {fm.get('best_resolution', 'N/A')}\n")
                    f.write(f"  Raw Patch Size: {fm.get('patch_size', 'N/A')}\n")
                    if fm.get("base_feature"):
                        f.write(f"  Base Feature Grid: {fm['base_feature'].get('grid', 'N/A')}\n")
                    if fm.get("patch_feature"):
                        f.write(f"  Patch Feature Unpadded Grid: {fm['patch_feature'].get('grid_unpadded', 'N/A')}\n")
                        f.write(f"  Patch Feature Padded Grid (Vis): {fm['patch_feature'].get('grid_for_visualization', 'N/A')}\n")
                else:
                    f.write("  Feature mapping information not available.\n")

                # Analysis Details
                analyzed_layer_indices = list(structured_concept_probs.keys()) if structured_concept_probs else []
                layers_info = "All Available" if selected_layers is None else str(selected_layers)
                f.write(f"\nLayers Requested for Analysis: {layers_info}\n")
                f.write(f"Layers Successfully Analyzed (Indices): {analyzed_layer_indices}\n")

                # Output Files
                f.write(f"\nVisualizations saved to subdirectories in: {output_dir}\n")
                viz_paths_list = results.get('visualization_paths', [])
                f.write(f"Number of visualization files generated: {len(viz_paths_list)}\n")

            results["summary_path"] = summary_path
            results["error"] = None # Clear initial error if workflow completes successfully
        except Exception as e:
            print(f"    Error writing summary file: {e}")
            results["summary_path"] = None # Indicate summary saving failed
            results["error"] = f"Workflow completed but failed to save summary: {e}" # Keep error info

        print(f"--- Logit Lens Workflow Complete ---")

    except Exception as e:
        # Catch unexpected errors during the workflow
        print(f"Error during Logit Lens workflow: {e}")
        import traceback
        traceback.print_exc()
        results["error"] = f"Workflow failed with exception: {e}" # Store the error message

    # --- Final Cleanup ---
    # Ensure intermediate data is deleted regardless of success/failure
    if 'forward_outputs' in locals(): del forward_outputs
    if 'hidden_states' in locals(): del hidden_states
    if 'prepared_data' in locals(): del prepared_data
    if 'layer_probabilities' in locals(): del layer_probabilities
    if 'structured_concept_probs' in locals(): del structured_concept_probs
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return results


# --- Workflow 2: Saliency Analysis (Existing - Assumed Correct, No Changes) ---
# This function remains the same as the one you provided previously,
# as its structure using the engine's stepwise analysis and the saliency analyzer
# functions seems appropriate and doesn't duplicate logic from representations.py.

def run_saliency_workflow(
    engine: LLaVANextEngine,                 # Input: Initialized Engine (must have enable_gradients=True)
    image_source: Union[str, Image.Image],   # Input: Image path, URL, or PIL object
    prompt_text: str,                        # Input: Text prompt
    num_tokens: int = 5,                     # Input: Number of tokens to generate and analyze
    output_dir: str = "saliency_analysis",   # Input: Directory for outputs
    layer_batch_size: Optional[int] = 4,     # Input: Batch size for layer gradients (passed to engine)
    save_plots: bool = True                  # Input: Whether to save saliency flow plots
) -> Dict[str, Any]:
    """
    Performs token-by-token generation with gradient-based saliency analysis workflow.

    Leverages the engine's `generate_analyze_stepwise` method with `GradientAttentionCapture`.
    The analysis callback computes saliency scores and flow metrics using `saliency_analyzer` functions.

    Args:
        engine (LLaVANextEngine): Initialized engine instance. Must have been loaded with `enable_gradients=True`.
        image_source (Union[str, Image.Image]): Source of the image.
        prompt_text (str): The text prompt.
        num_tokens (int): Number of generation steps to perform and analyze.
        output_dir (str): Directory to save results and plots.
        layer_batch_size (Optional[int]): Batch size for computing layer gradients per backward pass
                                          within the engine's stepwise method. None means compute all at once.
        save_plots (bool): If True, generates and saves saliency flow plots for each step.

    Returns:
        Dict[str, Any]: Dictionary containing the generated text, step-wise analysis results,
                        timing information, configuration, and error status.
    """
    print(f"\n--- Starting Gradient-Based Saliency Workflow ---")
    print(f" Config: NumTokens={num_tokens}, LayerBatch={layer_batch_size}")
    # Create output directory if saving plots
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        print(f" Plots will be saved to: {output_dir}")

    overall_start_time = time.time()
    # Initialize results with default error
    final_results = {"error": "Workflow did not complete."}

    # --- Input Validations ---
    if engine.model is None or engine.processor is None:
        final_results["error"] = "Engine is not properly initialized."
        print(f"Error: {final_results['error']}")
        return final_results
    # Saliency requires gradients during model loading
    if not engine.gradients_enabled_on_load(): # Check the flag used during engine init
        final_results["error"] = "Engine must be initialized with enable_gradients=True for saliency workflow."
        print(f"Error: {final_results['error']}")
        return final_results
    # Double-check if parameters actually require grad (might fail if loading 4-bit even with flag)
    if not any(p.requires_grad for p in engine.model.parameters()):
        print("Warning: Engine flag requested gradients, but no model parameters actually require grad (possibly due to quantization). Saliency analysis will likely fail or produce zero results.")
        # Allow proceeding but expect failure/zeros


    try:
        # --- 1. Prepare Initial Inputs ---
        print(" Step 1: Preparing initial inputs...")
        prepared_data = engine.build_inputs(image_source, prompt_text)
        initial_inputs = prepared_data['inputs']
        # Ensure necessary tensors are present
        if not all(k in initial_inputs for k in ["input_ids", "pixel_values"]):
            final_results["error"] = "Initial inputs missing 'input_ids' or 'pixel_values'."
            print(f"Error: {final_results['error']}")
            return final_results

        # Get token indices needed for flow metric calculation
        image_token_id = engine.get_config("image_token_id")
        if image_token_id is None:
             final_results["error"] = "Could not get image_token_id from engine config."
             print(f"Error: {final_results['error']}")
             return final_results
        # Perform index finding on CPU version of input_ids
        text_indices_t, image_indices_t = get_token_indices(initial_inputs['input_ids'].cpu(), image_token_id)
        print(f" Initial token counts: {len(text_indices_t)} Text, {len(image_indices_t)} Image.")

        # Get names of attention layers to hook for gradient capture
        attn_layer_names = engine.get_attention_layer_names()
        if not attn_layer_names:
            final_results["error"] = "Could not identify attention layer names via engine."
            print(f"Error: {final_results['error']}")
            return final_results
        print(f" Identified {len(attn_layer_names)} attention layers to hook for gradients.")

        # --- 2. Define the Analysis Callback for Each Step ---
        # This callback will be executed by the engine after each generation step.
        # It receives captured attention weights and gradients.
        # NOTE: This list needs to be defined *outside* the callback to persist across steps.
        stepwise_analysis_storage = []

        def saliency_analysis_step_callback(
            step_idx: int,                          # Current generation step index (0-based)
            target_token_pos: int,                  # Index of the token *before* the one being generated
            generated_token_id: torch.Tensor,       # ID of the generated token (on CPU)
            captured_data: Dict[str, Any]           # Data from GradientAttentionCapture (weights, grads)
        ) -> Dict[str, Any]:
            """Callback function executed at each step of generate_analyze_stepwise."""
            step_result = {"step": step_idx, "error": None} # Initialize result dict for this step
            try:
                 # Decode generated token for logging/plotting
                 token_text = engine.processor.decode([generated_token_id.item()], skip_special_tokens=True)
                 step_result["token_text"] = token_text
                 step_result["token_id"] = generated_token_id.item()
                 print(f"  Callback Step {step_idx+1}: Analyzing saliency for token '{token_text}' (ID: {generated_token_id.item()})")

                 # Extract weights and gradients from captured data
                 # Note: These tensors might be on GPU or CPU depending on engine's callback_cpu_offload setting
                 attention_weights = captured_data.get("attention_weights", {})
                 attention_grads = captured_data.get("attention_grads", {})

                 # Check if necessary data was captured
                 if not attention_grads:
                     print(f"    Callback Step {step_idx+1}: No gradients captured.")
                     step_result["saliency_metrics"] = {"error": "No gradients captured"}
                     step_result["error"] = "No gradients captured"
                     stepwise_analysis_storage.append(step_result) # Store error result
                     return step_result # Return early for this step

                 if not attention_weights:
                     print(f"    Callback Step {step_idx+1}: No attention weights captured (required for saliency).")
                     # Proceed with grads only? Saliency calculation will fail. Maybe report zero flow?
                     step_result["saliency_metrics"] = {"error": "No attention weights captured"}
                     step_result["error"] = "No attention weights captured"
                     stepwise_analysis_storage.append(step_result) # Store error result
                     return step_result

                 # Calculate Saliency Scores using the analyzer function
                 saliency_scores = calculate_saliency_scores(attention_weights, attention_grads)
                 if not saliency_scores:
                     print(f"    Callback Step {step_idx+1}: Saliency score calculation returned empty.")
                     step_result["saliency_metrics"] = {"error": "Saliency calculation failed"}
                     step_result["error"] = "Saliency calculation failed"
                     stepwise_analysis_storage.append(step_result) # Store error result
                     return step_result

                 # Analyze Layer-wise Flow using the analyzer function
                 # Pass the CPU versions of token indices
                 flow_metrics = analyze_layerwise_saliency_flow(
                     saliency_scores=saliency_scores,
                     text_indices=text_indices_t.cpu(), # Ensure indices are passed on CPU
                     image_indices=image_indices_t.cpu(),# Ensure indices are passed on CPU
                     target_token_idx=target_token_pos, # Target is the position *before* generation
                     cpu_offload=True                   # Request analyzer to work on CPU
                 )
                 step_result["saliency_metrics"] = flow_metrics

                 # Generate and save plot if requested and metrics are valid
                 if save_plots and flow_metrics and 'error' not in flow_metrics :
                     try:
                         # Create a filename based on step, token, and model
                         model_name_short = engine.model_id.split('/')[-1]
                         safe_token_text = "".join(c if c.isalnum() else "_" for c in token_text.strip())
                         if not safe_token_text: safe_token_text = f"id{generated_token_id.item()}"
                         plot_filename = f"token_{(step_idx+1):02d}_{safe_token_text}_saliency_flow.png"
                         step_plot_path = os.path.join(output_dir, plot_filename)
                         # Create plot title
                         plot_title = f"{model_name_short} - Saliency Flow for Token {step_idx+1} ('{token_text}') -> Pos {target_token_pos}"
                         # Call visualization utility
                         visualize_information_flow(flow_metrics, plot_title, step_plot_path)
                         step_result["plot_path"] = step_plot_path # Store path in results
                     except Exception as plot_err:
                         print(f"    Callback Step {step_idx+1}: Error generating plot: {plot_err}")
                         step_result["plot_path"] = f"Error: {plot_err}" # Record plot error

            except Exception as cb_err:
                print(f"    Callback Step {step_idx+1}: Unhandled error in callback: {cb_err}")
                import traceback; traceback.print_exc()
                step_result["error"] = f"Callback failed: {cb_err}"

            # Append the results of this step to the persistent list
            stepwise_analysis_storage.append(step_result)
            # Clean up potentially large tensors from this step's analysis
            if 'saliency_scores' in locals(): del saliency_scores
            if 'flow_metrics' in locals(): del flow_metrics
            gc.collect() # Trigger garbage collection
            return step_result # Return step result (engine might use it)

        # --- 3. Run Step-wise Generation and Analysis via Engine ---
        print(" Step 3: Running step-wise generation and analysis...")
        # The hook manager will be created internally by the engine based on requires_grad=True logic,
        # or we could instantiate GradientAttentionCapture explicitly and pass it.
        # The engine's current implementation implies it handles hook manager creation based on `requires_grad`.

        # The analysis_results_list returned by the engine might be redundant if callback stores results itself.
        generated_text, _ = engine.generate_analyze_stepwise(
            inputs=initial_inputs,
            num_steps=num_tokens,
            hook_manager=GradientAttentionCapture(cpu_offload_grads=True), # Explicitly pass manager instance
            layers_to_hook=attn_layer_names,
            analysis_callback=saliency_analysis_step_callback,
            # requires_grad=True, # No longer needed if hook_manager is passed
            layer_batch_size=layer_batch_size,
            callback_cpu_offload=True # Offload captured weights/grads to CPU before calling callback
        )

        # --- 4. Collate and Return Final Results ---
        overall_end_time = time.time()
        final_results = {
            "sequence_text": generated_text,
            "step_results": stepwise_analysis_storage, # Use the list populated by the callback
            "total_time": overall_end_time - overall_start_time,
            "model_name": engine.model_id,
            "config": {
                "num_tokens": num_tokens,
                "layer_batch_size": layer_batch_size,
                "prompt": prompt_text,
                # Handle non-string image source representation
                "image_source": image_source if isinstance(image_source, str) else "PIL Image Input",
                "engine_grads_enabled_on_load": engine.gradients_enabled_on_load()
            },
            "error": None # Clear default error if reached here without fatal exceptions
        }
        print(f"\n--- Saliency Workflow Finished ({final_results['total_time']:.2f} seconds) ---")

    except Exception as e:
        # Catch errors occurring outside the callback but within the workflow
        print(f"Error during Saliency workflow execution: {e}")
        import traceback; traceback.print_exc()
        # Store the exception message in the final results
        final_results["error"] = f"Workflow failed with exception: {e}"

    # --- Final Cleanup ---
    if 'prepared_data' in locals(): del prepared_data
    if 'initial_inputs' in locals(): del initial_inputs
    # Other large objects should be cleaned up within loops/callbacks or by GC
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return final_results




def run_saliency_workflow_memory_optimized(
    engine: LLaVANextEngine,
    image_source: Union[str, Image.Image],
    prompt_text: str,
    num_tokens: int = 5,
    output_dir: str = "saliency_analysis_mem_opt",
    layer_batch_size: int = 1, # How many layers to process per backward pass
    save_plots: bool = True,
    target_image_size: Optional[Tuple[int, int]] = None,
    # *** NEW: Option to control saliency offload within hook manager ***
    cpu_offload_saliency: bool = True
) -> Dict[str, Any]:
    """
    Performs saliency analysis using batched forward/backward passes.
    Uses GradientAttentionCapture.compute_saliency_scores_and_clear internally.
    """
    print(f"\n--- Starting Memory-Optimized Saliency Workflow (Original Logic Style) ---")
    print(f" Config: NumTokens={num_tokens}, LayerBatch={layer_batch_size}, TargetImageSize(WxH)={target_image_size or 'Default'}, OffloadSaliency={cpu_offload_saliency}")
    if save_plots: os.makedirs(output_dir, exist_ok=True)
    final_results = {"error": "Workflow did not complete."}
    start_time = time.time()

    model = engine.get_model()
    processor = engine.get_processor()
    if model is None or processor is None: return {"error": "Engine model/processor not available."}
    if not engine.gradients_enabled_on_load(): return {"error": "Engine must have gradients enabled on load."}
    if not any(p.requires_grad for p in model.parameters()): print("Warning: No model parameters require grad.")

    try:
        # --- 1. Initial Setup (remains the same) ---
        print(" Step 1: Preparing initial inputs...")
        # ... (input prep, token index finding, layer name getting logic as before) ...
        prepared_data = engine.build_inputs(image_source, prompt_text, target_image_size=target_image_size)
        initial_inputs = prepared_data['inputs']; pixel_values = initial_inputs["pixel_values"]; image_sizes = initial_inputs.get("image_sizes")
        initial_input_ids = initial_inputs["input_ids"]
        image_token_id = engine.get_config("image_token_id")
        text_indices, image_indices = get_token_indices(initial_input_ids.cpu(), image_token_id)
        all_attn_layer_names = engine.get_attention_layer_names()
        if not all_attn_layer_names: raise ValueError("Could not find attention layers.")
        print(f" Setup complete. Text={len(text_indices)}, Image={len(image_indices)}, Layers={len(all_attn_layer_names)}")

        # --- Main Loop ---
        token_results = {}
        generated_sequence = ""
        current_input_ids = initial_input_ids.clone()
        current_attention_mask = torch.ones_like(current_input_ids, device=model.device)

        print("\n Step 2: Starting token-by-token analysis loop...")
        for step_idx in tqdm(range(num_tokens), desc="Analyzing Tokens", ncols=100):
            step_start_inner = time.time()
            current_step_results = {}
            # *** MODIFICATION: Saliency scores accumulated per step ***
            all_saliency_scores_step = {}
            next_token_id = None
            loss_val = None

            # A. Predict next token ID (remains the same)
            model.eval()
            try:
                with torch.inference_mode():
                    outputs_pred = model(input_ids=current_input_ids, attention_mask=current_attention_mask, pixel_values=pixel_values, image_sizes=image_sizes, use_cache=True)
                    logits = outputs_pred.logits[:, -1, :]; next_token_id = torch.argmax(logits, dim=-1)
                    log_probs = torch.log_softmax(logits.float(), dim=-1); loss_val = -log_probs[0, next_token_id.item()].item()
                    del outputs_pred, logits, log_probs; gc.collect(); torch.cuda.empty_cache()
            except Exception as pred_err: print(f"\nPred Err step {step_idx+1}: {pred_err}"); final_results["error"] = "..."; break
            if next_token_id is None: print(f"\nToken gen failed step {step_idx+1}."); break
            new_token_text = processor.tokenizer.decode([next_token_id.item()])
            print(f"\n Step {step_idx+1}/{num_tokens}: Gen='{new_token_text}' (ID:{next_token_id.item()}), Loss={loss_val:.4f}")

            # B. Compute Gradients and Saliency (Batched Layer Loop)
            model.train()
            # Instantiate Hook Manager *once* per step
            # Pass the saliency offload option
            grad_capture = GradientAttentionCapture(cpu_offload_grads=True, cpu_offload_saliency=cpu_offload_saliency)
            num_layers = len(all_attn_layer_names)
            try:
                print(f"  Calculating gradients & saliency for step {step_idx+1}...")
                for batch_start in range(0, num_layers, layer_batch_size):
                    batch_end = min(batch_start + layer_batch_size, num_layers)
                    current_layer_batch_names = all_attn_layer_names[batch_start:batch_end]

                    model.zero_grad(set_to_none=True)
                    # Register hooks ONLY for the current batch
                    grad_capture.register_hooks(model, current_layer_batch_names)
                    loss = None
                    try: # Inner try for forward/backward
                        with torch.enable_grad():
                            outputs_grad = model(input_ids=current_input_ids, attention_mask=current_attention_mask, pixel_values=pixel_values, image_sizes=image_sizes, use_cache=False, output_attentions=True)
                            logits_grad = outputs_grad.logits[:, -1, :]; log_probs_grad = torch.log_softmax(logits_grad.float(), dim=-1)
                            loss = -log_probs_grad[0, next_token_id.item()]

                        del outputs_grad, logits_grad, log_probs_grad # Cleanup before backward
                        gc.collect(); torch.cuda.empty_cache()

                        loss.backward() # Compute gradients for batch

                        # *** MODIFICATION: Compute saliency using the hook manager's method ***
                        # This call now computes saliency AND clears data for the processed layers
                        batch_saliency = grad_capture.compute_saliency_scores_and_clear()
                        all_saliency_scores_step.update(batch_saliency)
                        # ***********************************************************************

                    except Exception as backward_err: raise backward_err # Re-raise
                    finally:
                        # Ensure hooks are cleared and loss is deleted for the batch
                        grad_capture.clear_hooks() # <<< IMPORTANT: Clear hooks after batch backward/saliency compute
                        if 'loss' in locals() and loss is not None: del loss
                        gc.collect(); torch.cuda.empty_cache()

                # C. Analyze flow (using step's saliency scores)
                model.eval()
                target_idx = current_input_ids.shape[1] - 1
                if not all_saliency_scores_step:
                     print(f"  Warning: No saliency scores generated step {step_idx+1}."); flow_metrics = {"error": "No saliency scores"}
                else:
                    print(f"  Analyzing saliency flow for step {step_idx+1}...")
                    flow_metrics = analyze_layerwise_saliency_flow(all_saliency_scores_step, text_indices.cpu(), image_indices.cpu(), target_idx, cpu_offload=True)

                # D. Save plots (remains the same)
                if save_plots and isinstance(flow_metrics, dict) and 'error' not in flow_metrics :
                    # ... (plotting logic as before) ...
                    try:
                        model_name_short = engine.model_id.split('/')[-1]; safe_token_text = "".join(c if c.isalnum() else "_" for c in new_token_text.strip())
                        if not safe_token_text: safe_token_text = f"id{next_token_id.item()}"
                        plot_filename = f"token_{(step_idx+1):02d}_{safe_token_text}_saliency_flow.png"; step_plot_path = os.path.join(output_dir, plot_filename)
                        plot_title = f"{model_name_short} - Saliency Flow Token {step_idx+1} ('{new_token_text}') -> Pos {target_idx}"
                        visualize_information_flow(flow_metrics, plot_title, step_plot_path); current_step_results["plot_path"] = step_plot_path
                    except Exception as plot_err: print(f" Plot Err: {plot_err}"); current_step_results["plot_path"] = f"Error: {plot_err}"


                # E. Store results (remains the same)
                current_step_results.update({"token_text": new_token_text,"token_id": next_token_id.item(),"loss": loss_val,"metrics": flow_metrics})
                token_results[f"token_{step_idx+1}"] = current_step_results

                # F. Prepare for next iteration (remains the same)
                current_input_ids = torch.cat([current_input_ids, next_token_id.unsqueeze(0)], dim=1)
                new_mask_entry = torch.ones((1, 1), dtype=current_attention_mask.dtype, device=model.device)
                current_attention_mask = torch.cat([current_attention_mask, new_mask_entry], dim=1)
                generated_sequence += new_token_text

            except Exception as step_err: # Catch errors from B, C, D, E, F
                print(f"\nError analyzing step {step_idx+1}: {step_err}")
                import traceback; traceback.print_exc()
                token_results[f"token_{step_idx+1}"] = {"error": f"Analysis failed: {step_err}"}
                final_results["error"] = f"Analysis failed step {step_idx+1}: {step_err}"
                if 'grad_capture' in locals(): grad_capture.clear_hooks() # Ensure hooks cleared on error
                break # Stop the loop
            finally:
                # Final cleanup for the step
                # grad_capture should be out of scope or hooks cleared already
                if 'all_saliency_scores_step' in locals(): del all_saliency_scores_step
                gc.collect(); torch.cuda.empty_cache()
                print(f"  Step {step_idx+1} analysis finished in {time.time() - step_start_inner:.2f}s")

        # --- End of Loop & Final Results (fix AttributeError location) ---
        loop_end_time = time.time()
        final_results = { # Collate results
            "token_results": token_results, "sequence_text": generated_sequence,
            "total_time": loop_end_time - start_time, "model_name": engine.model_id,
            "config": { "num_tokens": num_tokens, "layer_batch_size": layer_batch_size, "prompt": prompt_text, "image_source": image_source if isinstance(image_source, str) else "PIL Input", "target_image_size": target_image_size, "cpu_offload_saliency": cpu_offload_saliency },
            "error": final_results.get("error") # Preserve error if loop broke early
        }
        if final_results.get("error") is None and step_idx == num_tokens - 1: final_results["error"] = None
        elif final_results.get("error") is None: final_results["error"] = f"Loop stopped early step {step_idx+1}"

        print(f"\n--- Memory-Optimized Saliency Workflow Finished ({final_results['total_time']:.2f} seconds) ---")

    except Exception as e:
        print(f"Error during Workflow setup/execution: {e}")
        import traceback; traceback.print_exc()
        final_results["error"] = f"Workflow setup/exec failed: {e}"
    finally:
        # Final Cleanup - No grad_capture expected here anymore
        if 'prepared_data' in locals(): del prepared_data
        if 'initial_inputs' in locals(): del initial_inputs
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    return final_results