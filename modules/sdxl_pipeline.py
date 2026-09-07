"""
SDXL Pipeline Module — Fooocus Designer 2.0
Primary raster generation engine using RunDiffusion/Juggernaut-XL-v9 via HuggingFace diffusers.
Optimized for Google Colab T4 and 12GB+ GPUs with FP16, SDPA, VAE tiling, and baked LoRAs.
"""

import os
import gc
import random
import torch
from typing import Optional, Tuple, List, Dict, Any
from PIL import Image

from modules.lora_router import apply_category_lora, clear_adapters, get_active_adapter

_pipeline = None
_device = None
_current_speed_mode = None

# Default Base Model and Lightning Scheduler / LoRA
DEFAULT_SDXL_MODEL = "RunDiffusion/Juggernaut-XL-v9"
LIGHTNING_LORA_REPO = "ByteDance/SDXL-Lightning"
LIGHTNING_LORA_WEIGHT = "sdxl_lightning_4step_lora.safetensors"


def get_device() -> str:
    """Detect available compute device."""
    global _device
    if _device is not None:
        return _device
    if torch.cuda.is_available():
        _device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _device = "mps"
    else:
        _device = "cpu"
    return _device


def load_pipeline(speed_mode: str = "fast", progress_callback=None):
    """
    Load SDXL pipeline with PyTorch SDPA, VAE tiling/slicing, and memory safeguards.

    Args:
        speed_mode: "fast" (SDXL-Lightning 5-step) or "master" (Juggernaut-XL 25-step).
        progress_callback: Optional callable(message) for UI progress updates.

    Returns:
        The loaded StableDiffusionXLPipeline.
    """
    global _pipeline, _current_speed_mode

    if os.environ.get("MOCK_IMAGE_GEN") == "1":
        if _pipeline is None:
            _pipeline = "mock_pipeline"
        _current_speed_mode = speed_mode
        return _pipeline

    if _pipeline is not None:
        if _current_speed_mode == speed_mode:
            return _pipeline

    if progress_callback:
        progress_callback(f"Warming up Juggernaut XL ({speed_mode} mode)...")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

        if _pipeline is None:
            if progress_callback:
                progress_callback(f"Loading {DEFAULT_SDXL_MODEL} in FP16...")

            _pipeline = StableDiffusionXLPipeline.from_pretrained(
                DEFAULT_SDXL_MODEL,
                torch_dtype=dtype,
                variant="fp16" if device == "cuda" else None,
                use_safetensors=True,
                low_cpu_mem_usage=True,
            )

            if device == "cuda":
                _pipeline.to("cuda")
                # Colab zero-OOM memory optimizations
                _pipeline.vae.enable_tiling()
                _pipeline.vae.enable_slicing()
                try:
                    _pipeline.enable_attention_slicing()
                except Exception:
                    pass
            elif device == "mps":
                _pipeline.to("mps")

        # Configure scheduler for speed mode
        if speed_mode == "fast":
            # Load 4-step Lightning configuration
            _pipeline.scheduler = EulerDiscreteScheduler.from_config(
                _pipeline.scheduler.config,
                timestep_spacing="trailing"
            )
            # Load ByteDance SDXL-Lightning LoRA if not already applied
            try:
                _pipeline.load_lora_weights(
                    LIGHTNING_LORA_REPO,
                    weight_name=LIGHTNING_LORA_WEIGHT,
                    adapter_name="lightning_fast"
                )
            except Exception as e:
                print(f"[SDXL Pipeline] Lightning adapter notice: {e}")

            if hasattr(_pipeline, "enable_lora"):
                _pipeline.enable_lora()
            if hasattr(_pipeline, "set_adapters"):
                _pipeline.set_adapters(["lightning_fast"], adapter_weights=[1.0])
        else:
            # Master mode: standard DPM++ 2M Karras or Euler
            from diffusers import DPMSolverMultistepScheduler
            _pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                _pipeline.scheduler.config,
                use_karras_sigmas=True,
                algorithm_type="sde-dpmsolver++"
            )
            if hasattr(_pipeline, "disable_lora"):
                try:
                    _pipeline.disable_lora()
                except Exception:
                    pass

        _current_speed_mode = speed_mode
        if progress_callback:
            progress_callback(f"Pipeline ready on {device} ({speed_mode} mode)")

    except Exception as e:
        print(f"[SDXL Pipeline] Failed to load pipeline: {e}")
        _pipeline = None
        _current_speed_mode = None
        raise

    return _pipeline


def get_current_speed_mode() -> Optional[str]:
    """Return currently active speed mode ('fast', 'master', or None)."""
    return _current_speed_mode


def generate(
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    seed: int = -1,
    speed_mode: str = "fast",
    category_cfg: Optional[Dict[str, Any]] = None,
    progress_callback=None,
) -> Tuple[Image.Image, int]:
    """
    Generate an image using Juggernaut XL with auto-baked LoRAs.

    Args:
        prompt: Raw user or enhanced prompt text.
        negative_prompt: Negative prompt.
        width: Output width (default 1024).
        height: Output height (default 1024).
        seed: Random seed (-1 for random).
        speed_mode: "fast" (5-6 steps) or "master" (25-28 steps).
        category_cfg: Category configuration containing optional baked LoRA.
        progress_callback: Progress reporting callable.

    Returns:
        Tuple of (PIL.Image, used_seed).
    """
    if seed == -1 or seed is None:
        seed = random.randint(0, 2**32 - 1)

    pipe = load_pipeline(speed_mode=speed_mode, progress_callback=progress_callback)

    # Configure adapters on pipe
    trigger_words = ""
    has_lora = bool(category_cfg and category_cfg.get("lora"))
    if has_lora:
        trigger_words = apply_category_lora(pipe, category_cfg)
        adapter_name = get_active_adapter()
        lora_weight = float(category_cfg["lora"].get("weight", 0.85))
        if adapter_name:
            if speed_mode == "fast":
                if hasattr(pipe, "enable_lora"):
                    pipe.enable_lora()
                if hasattr(pipe, "set_adapters"):
                    pipe.set_adapters(["lightning_fast", adapter_name], adapter_weights=[1.0, lora_weight])
            else:
                if hasattr(pipe, "enable_lora"):
                    pipe.enable_lora()
                if hasattr(pipe, "set_adapters"):
                    pipe.set_adapters([adapter_name], adapter_weights=[lora_weight])
        else:
            if speed_mode == "fast":
                if hasattr(pipe, "enable_lora"):
                    pipe.enable_lora()
                if hasattr(pipe, "set_adapters"):
                    pipe.set_adapters(["lightning_fast"], adapter_weights=[1.0])
    else:
        clear_adapters(pipe)
        if speed_mode == "fast":
            if hasattr(pipe, "enable_lora"):
                pipe.enable_lora()
            if hasattr(pipe, "set_adapters"):
                pipe.set_adapters(["lightning_fast"], adapter_weights=[1.0])

    final_prompt = f"{trigger_words}, {prompt}".strip(", ") if trigger_words else prompt

    # In mock mode, synthesize design image for fast headless testing
    if os.environ.get("MOCK_IMAGE_GEN") == "1":
        from PIL import ImageDraw
        random.seed(seed)

        if progress_callback:
            progress_callback(f"[Mock SDXL] Generating ({speed_mode} mode, seed: {seed})...")

        img = Image.new("RGB", (width, height), color="#0f172a")
        draw = ImageDraw.Draw(img)

        # Draw decorative grid and center motif
        draw.rectangle([int(width * 0.1), int(height * 0.1), int(width * 0.9), int(height * 0.9)], outline="#38bdf8", width=3)
        draw.ellipse([int(width * 0.25), int(height * 0.25), int(width * 0.75), int(height * 0.75)], fill="#4f46e5")
        draw.text((int(width * 0.15), int(height * 0.85)), f"SDXL Juggernaut [{speed_mode}] | Seed: {seed}", fill="#ffffff")

        if progress_callback:
            progress_callback("Mock generation complete!")
        return img, seed

    # Configure steps and CFG according to speed mode
    if speed_mode == "fast":
        steps = 6
        cfg_scale = 1.8
    else:
        steps = 28
        cfg_scale = 6.0

    generator = torch.Generator(device="cpu").manual_seed(seed)

    if progress_callback:
        progress_callback(f"Rendering asset ({steps} steps, CFG {cfg_scale}, seed {seed})...")

    try:
        result = pipe(
            prompt=final_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            generator=generator,
        )
        image = result.images[0]

        if progress_callback:
            progress_callback("Generation complete!")

        return image, seed

    except Exception as e:
        print(f"[SDXL Pipeline] Generation failed: {e}")
        raise


def generate_variations(
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    base_seed: Optional[int] = None,
    count: int = 2,
    speed_mode: str = "fast",
    category_cfg: Optional[Dict[str, Any]] = None,
    progress_callback=None,
) -> List[Tuple[Image.Image, int]]:
    """Generate multiple variations of a design asset with seed spread."""
    if base_seed is None or base_seed == -1:
        base_seed = random.randint(0, 2**32 - 1)

    results = []
    for i in range(count):
        varied_seed = (base_seed + (i + 1) * 7919) % (2**32)
        if progress_callback:
            progress_callback(f"Generating variation {i + 1}/{count}...")

        img, s = generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            seed=varied_seed,
            speed_mode=speed_mode,
            category_cfg=category_cfg,
            progress_callback=None,
        )
        results.append((img, s))

    return results


def unload_pipeline() -> None:
    """Free VRAM by deleting pipeline and releasing CUDA cache."""
    global _pipeline, _current_speed_mode
    if _pipeline is not None:
        clear_adapters(_pipeline)
        del _pipeline
        _pipeline = None
        _current_speed_mode = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("[SDXL Pipeline] Unloaded pipeline and cleared GPU memory")


def is_loaded() -> bool:
    """Return whether the pipeline is loaded in memory."""
    return _pipeline is not None
