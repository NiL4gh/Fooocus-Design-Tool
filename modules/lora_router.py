"""
LoRA Router Module — Fooocus Designer 2.0
Manages dynamic loading, switching, and caching of PEFT LoRA adapters on SDXL pipelines.
Provides zero-reload switching and automatic trigger word extraction.
"""

import os
import re
from typing import Optional, Dict, Any, List

_loaded_adapters: Dict[str, str] = {}
_active_adapter: Optional[str] = None


def _sanitize_name(name: str) -> str:
    """Convert category name to a clean alphanumeric adapter ID."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name.strip().lower()).strip("_")


def is_lora_available(source: str) -> bool:
    """Check if a LoRA source exists locally or is a valid HuggingFace repo ID."""
    if not source or not isinstance(source, str):
        return False
    if os.path.exists(source):
        return True
    
    # Reject strings ending with known file extensions if they do not exist locally
    known_extensions = (".safetensors", ".bin", ".pt", ".ckpt")
    if source.lower().endswith(known_extensions):
        return False

    # Validate HF repo IDs with regex ^[a-zA-Z0-9_\-\.]+/[a-zA-Z0-9_\-\.]+$
    if re.match(r"^[a-zA-Z0-9_\-\.]+/[a-zA-Z0-9_\-\.]+$", source):
        return True

    return False


def get_active_adapter() -> Optional[str]:
    """Return the name of the currently active adapter, if any."""
    return _active_adapter


def clear_adapters(
    pipeline,
    base_adapters: Optional[List[str]] = None,
    base_weights: Optional[List[float]] = None,
) -> None:
    """
    Disable active LoRA adapters on the pipeline or restore base adapters.

    Args:
        pipeline: Diffusers SDXL pipeline instance or dummy test pipeline.
        base_adapters: Optional list of adapter names to retain (e.g. ['lightning_fast']).
        base_weights: Optional list of adapter weights corresponding to base_adapters.
    """
    global _active_adapter
    if pipeline is not None:
        if base_adapters:
            if hasattr(pipeline, "enable_lora"):
                try:
                    pipeline.enable_lora()
                except Exception as e:
                    print(f"[LoRA Router] Warning enabling LoRA: {e}")
            if hasattr(pipeline, "set_adapters"):
                try:
                    weights = list(base_weights) if base_weights is not None else [1.0] * len(base_adapters)
                    pipeline.set_adapters(list(base_adapters), adapter_weights=weights)
                except Exception as e:
                    print(f"[LoRA Router] Warning setting base adapters: {e}")
        elif hasattr(pipeline, "disable_lora"):
            try:
                pipeline.disable_lora()
            except Exception as e:
                print(f"[LoRA Router] Warning disabling LoRA: {e}")
    _active_adapter = None


def apply_category_lora(
    pipeline,
    category_cfg: Dict[str, Any],
    force_active: bool = True,
    base_adapters: Optional[List[str]] = None,
    base_weights: Optional[List[float]] = None,
) -> str:
    """
    Apply a category's baked LoRA to the pipeline.

    Args:
        pipeline: Diffusers SDXL pipeline instance or dummy test pipeline.
        category_cfg: Dictionary containing category configuration and optional "lora" block.
        force_active: Whether to activate the adapter immediately.
        base_adapters: Optional list of base adapter names to preserve (e.g. ['lightning_fast']).
        base_weights: Optional list of base adapter weights.

    Returns:
        The trigger words string to prepend/append to the generation prompt.
    """
    global _active_adapter, _loaded_adapters

    if not category_cfg or "lora" not in category_cfg or not category_cfg["lora"]:
        clear_adapters(pipeline, base_adapters=base_adapters, base_weights=base_weights)
        return ""

    lora_cfg = category_cfg["lora"]
    source = lora_cfg.get("source") or lora_cfg.get("repo_or_file")
    weight = float(lora_cfg.get("weight", 0.85))
    trigger_words = lora_cfg.get("trigger_words", "").strip()

    if not source:
        clear_adapters(pipeline, base_adapters=base_adapters, base_weights=base_weights)
        return trigger_words

    category_name = category_cfg.get("name", "custom")
    adapter_name = _sanitize_name(category_name)

    # Compute target adapter names and weights atomically
    if base_adapters:
        active_names = list(base_adapters) + [adapter_name]
        default_base_weights = [1.0] * len(base_adapters)
        active_weights = list(base_weights if base_weights is not None else default_base_weights) + [weight]
    else:
        active_names = [adapter_name]
        active_weights = [weight]

    # In mock mode, update internal state without touching torch weights
    if os.environ.get("MOCK_IMAGE_GEN") == "1" or pipeline == "mock_pipeline":
        if adapter_name not in _loaded_adapters:
            if hasattr(pipeline, "load_lora_weights"):
                weight_name = lora_cfg.get("weight_name")
                kwargs = {"adapter_name": adapter_name}
                if weight_name:
                    kwargs["weight_name"] = weight_name
                pipeline.load_lora_weights(source, **kwargs)
            _loaded_adapters[adapter_name] = source

        if force_active:
            if hasattr(pipeline, "enable_lora"):
                pipeline.enable_lora()
            if hasattr(pipeline, "set_adapters"):
                pipeline.set_adapters(active_names, adapter_weights=active_weights)
            _active_adapter = adapter_name
        return trigger_words

    if pipeline is None:
        return trigger_words

    try:
        # Check if adapter is already recognized by PEFT/diffusers
        already_in_peft = False
        if hasattr(pipeline, "peft_config") and isinstance(pipeline.peft_config, dict):
            if adapter_name in pipeline.peft_config:
                already_in_peft = True
        elif hasattr(pipeline, "get_active_adapters"):
            try:
                if adapter_name in (pipeline.get_active_adapters() or []):
                    already_in_peft = True
            except Exception:
                pass

        # Load adapter if not already in cache and not in PEFT
        if adapter_name not in _loaded_adapters and not already_in_peft:
            print(f"[LoRA Router] Loading adapter '{adapter_name}' from: {source}")
            weight_name = lora_cfg.get("weight_name")
            kwargs = {"adapter_name": adapter_name}
            if weight_name:
                kwargs["weight_name"] = weight_name
            
            try:
                pipeline.load_lora_weights(source, **kwargs)
                _loaded_adapters[adapter_name] = source
            except Exception as load_err:
                err_str = str(load_err).lower()
                if "already in use" in err_str:
                    print(f"[LoRA Router] Adapter '{adapter_name}' already registered in PEFT, activating directly.")
                    _loaded_adapters[adapter_name] = source
                else:
                    raise load_err
        else:
            _loaded_adapters[adapter_name] = source

        # Set active adapter and scale weight
        if force_active:
            if hasattr(pipeline, "enable_lora"):
                pipeline.enable_lora()
            if hasattr(pipeline, "set_adapters"):
                pipeline.set_adapters(active_names, adapter_weights=active_weights)
            _active_adapter = adapter_name
            print(f"[LoRA Router] Activated adapter '{adapter_name}' with weight {weight}")

    except Exception as e:
        print(f"[LoRA Router] Failed to load/set LoRA '{adapter_name}': {e}. Continuing without LoRA.")
        clear_adapters(pipeline, base_adapters=base_adapters, base_weights=base_weights)

    return trigger_words
