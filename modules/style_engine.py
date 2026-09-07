"""
Style Engine Module — Fooocus Designer 2.0
Loads Fooocus, SAI, and MRE style definitions from JSON templates and applies
multi-style {prompt} interpolation and negative prompt stacking.
"""

import os
import glob
import json
from typing import Dict, List, Tuple, Optional

_styles_cache: Optional[Dict[str, Dict[str, str]]] = None


def get_styles_dir() -> str:
    """Return the absolute path to the style JSON templates directory."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root_dir, "config", "sdxl_styles")


def load_styles(force_reload: bool = False) -> Dict[str, Dict[str, str]]:
    """
    Load and cache all style templates from config/sdxl_styles/*.json.

    Returns:
        Dictionary mapping style names to {"name": ..., "prompt": ..., "negative_prompt": ...}.
    """
    global _styles_cache
    if _styles_cache is not None and not force_reload:
        return _styles_cache

    styles_dir = get_styles_dir()
    styles: Dict[str, Dict[str, str]] = {}

    if not os.path.isdir(styles_dir):
        _styles_cache = styles
        return styles

    for fpath in sorted(glob.glob(os.path.join(styles_dir, "*.json"))):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "name" in item:
                            styles[item["name"]] = {
                                "name": item["name"],
                                "prompt": item.get("prompt", "{prompt}"),
                                "negative_prompt": item.get("negative_prompt", ""),
                            }
        except Exception as e:
            print(f"[Style Engine] Error reading style file {fpath}: {e}")

    _styles_cache = styles
    return _styles_cache


def get_available_styles() -> List[str]:
    """Return an alphabetically sorted list of available style names."""
    styles = load_styles()
    return sorted(list(styles.keys()))


def get_style(name: str) -> Optional[Dict[str, str]]:
    """Retrieve configuration for a specific style name."""
    styles = load_styles()
    return styles.get(name)


def apply_styles(
    prompt: str,
    negative_prompt: str = "",
    style_names: Optional[List[str]] = None,
) -> Tuple[str, str]:
    """
    Apply one or more Fooocus styles to a prompt and negative prompt.

    Args:
        prompt: Raw user or category-enhanced prompt.
        negative_prompt: Existing negative prompt.
        style_names: List of style names to apply sequentially.

    Returns:
        Tuple of (styled_positive_prompt, styled_negative_prompt).
    """
    if not style_names:
        return prompt, negative_prompt

    styles = load_styles()
    current_positive = prompt.strip()
    cleaned_neg = negative_prompt.strip().rstrip(",").strip()
    negative_parts = [cleaned_neg] if cleaned_neg else []

    for name in style_names:
        if name in styles:
            style_cfg = styles[name]
            p_template = style_cfg.get("prompt", "{prompt}")
            n_addition = style_cfg.get("negative_prompt", "").strip().rstrip(",").strip()

            # Substitute {prompt} token
            if "{prompt}" in p_template:
                current_positive = p_template.replace("{prompt}", current_positive).strip(", ")
            elif current_positive:
                current_positive = f"{current_positive}, {p_template}".strip(", ")
            else:
                current_positive = p_template

            if n_addition and n_addition not in negative_parts:
                negative_parts.append(n_addition)

    final_negative = ", ".join(negative_parts).strip(", ")
    return current_positive, final_negative
