# -*- coding: utf-8 -*-
"""
Initialization file for the VLM Analysis engine module.
Makes the core engine class and extraction functions easily importable.
"""

from engine.llava_engine import LLaVANextEngine
from engine.representations import get_hidden_states_from_forward_pass, get_logits_from_hidden_states

__all__ = [
    "LLaVANextEngine",
    "get_hidden_states_from_forward_pass",
    "get_logits_from_hidden_states",
]