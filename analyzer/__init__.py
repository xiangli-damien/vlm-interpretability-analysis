# -*- coding: utf-8 -*-
"""
Initialization file for the VLM Analysis analyzer module.

Exposes core analyzer classes and workflow functions.
Uses absolute imports assuming the project root is in sys.path.
"""

# --- Core Analyzer Implementations ---
from analyzer.logit_lens_analyzer import LogitLensAnalyzer
from analyzer.saliency_analyzer import (
    calculate_saliency_scores,
    analyze_layerwise_saliency_flow,
    compute_flow_metrics_optimized  # Utility for saliency flow calculation
)

# --- Workflow Functions ---
from analyzer.workflows import (
    run_logit_lens_workflow,
    run_saliency_workflow,
    run_saliency_workflow_memory_optimized # <-- ADD THE NEW FUNCTION HERE
)

# --- Public API Definition (`__all__`) ---
# Lists names to be imported with 'from analyzer import *'
__all__ = [
    # Analyzers & Components
    "LogitLensAnalyzer",
    "calculate_saliency_scores",
    "analyze_layerwise_saliency_flow",
    "compute_flow_metrics_optimized",
    # Workflows
    "run_logit_lens_workflow",
    "run_saliency_workflow",
    "run_saliency_workflow_memory_optimized", # <-- ADD TO __all__
]