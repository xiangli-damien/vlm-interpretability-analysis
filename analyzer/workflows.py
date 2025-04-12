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
    from utils.data_utils import load_image, find_token_indices, build_conversation
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



def run_saliency_workflow(
    model: torch.nn.Module, # Expects model with grads enabled
    processor: Any,
    image_source: Union[str, Image.Image],
    prompt_text: str,
    num_tokens: int = 5,
    output_dir: str = "saliency_analysis",
    image_size: Optional[Tuple[int, int]] = (336, 336), # Allow None
    layer_batch_size: int = 2,
    save_plots: bool = True,
    cpu_offload_saliency: bool = True
) -> Dict[str, Any]:
    """
    Performs token-by-token generation with gradient-based saliency analysis.
    Reflects the logic structure of the previously provided 'working older' workflow.
    Calls grad_capture.compute_saliency() internally.
    CORRECTED: Ensures input tensors are moved to the model's primary device.
    """
    print(f"\n--- Starting Saliency Workflow (Old Logic Style - Corrected Device Handling) ---")
    print(f" Config: NumTokens={num_tokens}, ImgSize={image_size}, LayerBatch={layer_batch_size}, OffloadSaliency={cpu_offload_saliency}")
    if save_plots: os.makedirs(output_dir, exist_ok=True); print(f" Plots will be saved to: {output_dir}")
    final_results = {"error": "Workflow did not complete."}
    start_time = time.time()

    try:
        # --- 1. Initial Setup ---
        if not any(p.requires_grad for p in model.parameters()):
             raise RuntimeError("Gradients are not enabled on the model.")
        # Load image - pass image_size which might be None or a tuple
        image = load_image(image_source, resize_to=image_size, verbose=False)
        conversation = build_conversation(prompt_text)
        formatted_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        # Process inputs (will likely be on CPU initially)
        inputs_dict = processor(images=image, text=formatted_prompt, return_tensors="pt")

        # *** CORRECTED DEVICE PLACEMENT ***
        try:
            # Determine the target device (usually model.device for embedding layer)
            # model.device should point to the primary device even with device_map
            target_device = model.device
            print(f" Moving input tensors to target device: {target_device}")
            inputs = {k: v.to(target_device) for k, v in inputs_dict.items() if torch.is_tensor(v)}
             # Ensure non-tensor items are included if processor adds them
            for k, v in inputs_dict.items():
                if k not in inputs: inputs[k] = v
        except Exception as e:
             print(f"Warning: Failed to move inputs to device {target_device}. Error: {e}. Inputs may remain on CPU.")
             inputs = inputs_dict # Fallback to original dict if move fails
        # **********************************

        initial_input_ids = inputs["input_ids"]; pixel_values = inputs["pixel_values"]; image_sizes = inputs.get("image_sizes")
        image_token_id = getattr(model.config, "image_token_index", 32000)
        text_indices, image_indices = find_token_indices(initial_input_ids.cpu(), image_token_id) # Find indices on CPU copy
        all_attn_layer_names = get_llm_attention_layer_names(model)
        if not all_attn_layer_names: raise ValueError("Could not find attention layers.")
        print(f" Found {len(all_attn_layer_names)} attention layers.")

        # --- 2. Helper for Batched Gradient Computation (remains the same internally) ---
        def generate_next_token_and_compute_saliency(current_input_ids_inner):
            print(f"[Saliency] ===== Begin step with input shape: {current_input_ids_inner.shape} =====")

            all_saliency_scores_batch = {}
            next_token_id_inner = None
            loss_val_inner = None
            model.eval()

            current_mask = torch.ones_like(current_input_ids_inner)
            with torch.no_grad():
                outputs_pred = model(
                    input_ids=current_input_ids_inner,
                    attention_mask=current_mask,
                    pixel_values=pixel_values,
                    image_sizes=image_sizes,
                    use_cache=True
                )
                logits = outputs_pred.logits[:, -1, :]
                next_token_id_inner = torch.argmax(logits, dim=-1)
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                loss_val_inner = -log_probs[0, next_token_id_inner.item()].item()

                print(f"[Saliency] Forward prediction done. Token: {next_token_id_inner.item()}, Loss: {loss_val_inner:.4f}")
                del outputs_pred, logits, log_probs
                gc.collect()
                torch.cuda.empty_cache()

            model.train()
            print("[Saliency] Switched model to train mode.")
            grad_capture = GradientAttentionCapture(cpu_offload=cpu_offload_saliency)
            print("[Saliency] GradientAttentionCapture initialized.")

            num_layers = len(all_attn_layer_names)
            for batch_start in range(0, num_layers, layer_batch_size):
                batch_end = min(batch_start + layer_batch_size, num_layers)
                current_layer_batch = all_attn_layer_names[batch_start:batch_end]
                print(f"[Saliency] Registering hooks for layers {batch_start} to {batch_end - 1}...")

                model.zero_grad(set_to_none=True)
                grad_capture.register_hooks(model, current_layer_batch)

                try:
                    with torch.enable_grad():
                        outputs_grad = model(
                            input_ids=current_input_ids_inner,
                            attention_mask=current_mask,
                            pixel_values=pixel_values,
                            image_sizes=image_sizes,
                            use_cache=False,
                            output_attentions=True
                        )
                        logits_grad = outputs_grad.logits[:, -1, :]
                        log_probs_grad = torch.log_softmax(logits_grad.float(), dim=-1)
                        loss_inner = -log_probs_grad[0, next_token_id_inner.item()]
                        print(f"[Saliency] Computed loss: {loss_inner.item():.4f} for token ID {next_token_id_inner.item()}")

                    del outputs_grad, logits_grad, log_probs_grad
                    gc.collect()
                    torch.cuda.empty_cache()

                    print("[Saliency] Running backward pass...")
                    loss_inner.backward()
                    print("[Saliency] Backward pass completed.")

                    print("[Saliency] Computing saliency scores...")
                    batch_saliency = grad_capture.compute_saliency()
                    print(f"[Saliency] Saliency computed for layers: {list(batch_saliency.keys())}")
                    all_saliency_scores_batch.update(batch_saliency)

                except Exception as bk_err:
                    grad_capture.clear_hooks()
                    print(f"[Saliency] Exception during backward: {bk_err}")
                    raise bk_err
                finally:
                    grad_capture.clear_hooks()

                if 'loss_inner' in locals() and loss_inner is not None:
                    del loss_inner
                gc.collect()
                torch.cuda.empty_cache()

            model.eval()
            grad_capture.clear_cache()
            print("[Saliency] Cleared gradient cache.")
            print(f"[Saliency] ===== End of step =====\n")

            return next_token_id_inner, all_saliency_scores_batch, loss_val_inner

        # --- 3. Token-by-Token Generation and Analysis Loop (remains the same) ---
        token_results = {}; generated_sequence = ""; current_input_ids = initial_input_ids.clone()
        loop_start_time = time.time()
        for step_idx in range(num_tokens):
            print(f"--- Analyzing Saliency Token {step_idx+1}/{num_tokens} ---")
            step_saliency_scores = {}
            try:
                # (Call helper, analyze flow, visualize, store results, update input_ids - logic remains the same)
                next_token_id, step_saliency_scores, loss_val = generate_next_token_and_compute_saliency(current_input_ids)
                if next_token_id is None: break
                new_token_text = processor.tokenizer.decode([next_token_id.item()])
                print(f"  Generated token: '{new_token_text}' (ID: {next_token_id.item()}, Loss: {loss_val:.4f})")
                target_idx = current_input_ids.shape[1] - 1
                if not step_saliency_scores: 
                    print("  Warning: No saliency scores computed for this token.")
                    flow_metrics = {"error": "No saliency scores computed"}
                else:
                    print(f"  [DEBUG Workflow] Saliency scores generated for step {step_idx+1}. Keys: {list(step_saliency_scores.keys())}") 
                    flow_metrics = analyze_layerwise_saliency_flow(step_saliency_scores, text_indices.cpu(), image_indices.cpu(), target_idx, cpu_offload=True)
                    print(f"  [DEBUG Workflow] Flow metrics computed for step {step_idx+1}. Keys: {list(flow_metrics.keys()) if isinstance(flow_metrics, dict) else 'Not a dict'}")
                if save_plots and isinstance(flow_metrics, dict) and 'error' not in flow_metrics :
                     try:
                         model_name_short = model.config._name_or_path.split('/')[-1]; safe_token_text = "".join(c if c.isalnum() else "_" for c in new_token_text.strip()) or f"id{next_token_id.item()}"
                         plot_filename = f"token_{(step_idx+1):02d}_{safe_token_text}_saliency_flow.png"; plot_path = os.path.join(output_dir, plot_filename)
                         plot_title = f"{model_name_short} - Saliency Flow Token {step_idx+1} ('{new_token_text}') -> Pos {target_idx}"
                         visualize_information_flow(flow_metrics, plot_title, plot_path)
                     except Exception as plot_err: print(f" Plot Err: {plot_err}")
                token_results[f"token_{step_idx+1}"] = {"token_text": new_token_text,"token_id": next_token_id.item(),"loss": loss_val,"metrics": flow_metrics}
                current_input_ids = torch.cat([current_input_ids, next_token_id.unsqueeze(0)], dim=1)
                generated_sequence += new_token_text
                del step_saliency_scores, flow_metrics; gc.collect(); torch.cuda.empty_cache()

            except Exception as e: # Catch errors in the loop
                print(f"Error analyzing token {step_idx+1}: {e}")
                import traceback; traceback.print_exc()
                token_results[f"token_{step_idx+1}"] = {"error": f"Analysis failed: {e}"}
                final_results["error"] = f"Analysis failed at step {step_idx+1}: {e}"
                break
        loop_end_time = time.time()

        # --- Collate results (remains the same) ---
        final_results = { # Collate results
            "token_results": token_results, "sequence_text": generated_sequence,
            "total_time": loop_end_time - start_time, "model_name": model.config._name_or_path,
            "config": {"num_tokens": num_tokens,"image_size": image_size,"layer_batch_size": layer_batch_size,"prompt": prompt_text,"image_source": image_source if isinstance(image_source, str) else "PIL Input", "cpu_offload_saliency": cpu_offload_saliency},
            "error": final_results.get("error")
        }
        # ... (setting final error state logic) ...
        if final_results.get("error") is None and step_idx == num_tokens - 1: final_results["error"] = None
        elif final_results.get("error") is None: final_results["error"] = f"Loop stopped early step {step_idx+1}"

        print(f"\n--- Saliency Workflow (Old Logic) Finished ({final_results['total_time']:.2f} seconds) ---")

    except Exception as e: # Catch setup errors
        print(f"Error during Saliency workflow setup: {e}")
        import traceback; traceback.print_exc()
        final_results["error"] = f"Workflow setup failed: {e}"

    # No saving logic here
    return final_results