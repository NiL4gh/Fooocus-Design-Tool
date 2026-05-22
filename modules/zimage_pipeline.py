"""
Z-Image-Turbo Pipeline Module
Primary raster generation engine using Tongyi-MAI/Z-Image-Turbo via HuggingFace diffusers.
Optimized for T4 GPU (15GB VRAM) with FP16 and CPU offloading.
"""

import torch
import gc
from PIL import Image

_pipeline = None
_device = None


def get_device():
    """Detect available device."""
    global _device
    if _device is not None:
        return _device
    if torch.cuda.is_available():
        _device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        _device = "mps"
    else:
        _device = "cpu"
    return _device


def load_pipeline(progress_callback=None):
    """
    Load Z-Image-Turbo pipeline with FP16 quantization and CPU offloading.

    Args:
        progress_callback: Optional callable(message) for UI progress updates.

    Returns:
        The loaded diffusion pipeline.
    """
    global _pipeline

    if _pipeline is not None:
        return _pipeline

    if progress_callback:
        progress_callback("Loading Z-Image-Turbo model (first launch may take a few minutes)...")

    try:
        from diffusers import AutoPipelineForText2Image

        device = get_device()
        dtype = torch.float16 if device == "cuda" else torch.float32

        if progress_callback:
            progress_callback("Downloading model from HuggingFace...")

        _pipeline = AutoPipelineForText2Image.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=dtype,
        )

        if device == "cuda":
            _pipeline.enable_model_cpu_offload()
            if progress_callback:
                progress_callback("Model loaded with GPU acceleration + CPU offload")
        else:
            _pipeline = _pipeline.to(device)
            if progress_callback:
                progress_callback(f"Model loaded on {device}")

        # Memory optimizations
        if device == "cuda":
            try:
                _pipeline.enable_xformers_memory_efficient_attention()
            except Exception:
                pass  # xformers not available, skip

        print(f"[Z-Image-Turbo] Pipeline loaded on {device} with dtype {dtype}")

    except Exception as e:
        print(f"[Z-Image-Turbo] Failed to load pipeline: {e}")
        _pipeline = None
        raise

    return _pipeline


def generate(
    prompt,
    negative_prompt="",
    width=1024,
    height=1024,
    seed=-1,
    num_inference_steps=4,
    guidance_scale=0.0,
    progress_callback=None,
):
    """
    Generate an image using Z-Image-Turbo.

    Args:
        prompt: Text prompt for generation.
        negative_prompt: Negative prompt text.
        width: Output width in pixels.
        height: Output height in pixels.
        seed: Random seed (-1 for random).
        num_inference_steps: Number of diffusion steps (default 4 for turbo).
        guidance_scale: CFG scale (0.0 for turbo models).
        progress_callback: Optional callable(message) for progress updates.

    Returns:
        PIL Image object.
    """
    import random

    pipe = load_pipeline(progress_callback)

    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    generator = torch.Generator(device="cpu").manual_seed(seed)

    if progress_callback:
        progress_callback(f"Generating image (seed: {seed})...")

    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        image = result.images[0]

        if progress_callback:
            progress_callback("Generation complete!")

        return image, seed

    except Exception as e:
        print(f"[Z-Image-Turbo] Generation failed: {e}")
        raise


def generate_variations(
    prompt,
    negative_prompt="",
    width=1024,
    height=1024,
    base_seed=None,
    count=4,
    variation_strength=50,
    progress_callback=None,
):
    """
    Generate multiple variations by mixing seeds.

    Args:
        prompt: Text prompt.
        negative_prompt: Negative prompt.
        width: Output width.
        height: Output height.
        base_seed: Base seed to vary from.
        count: Number of variations (2-4).
        variation_strength: How different the variations should be (1-100).
        progress_callback: Optional progress callback.

    Returns:
        List of (PIL Image, seed) tuples.
    """
    import random

    if base_seed is None:
        base_seed = random.randint(0, 2**32 - 1)

    results = []
    for i in range(count):
        # Create varied seeds by adding offsets proportional to variation_strength
        offset = int((i + 1) * variation_strength * 137)  # Prime multiplier for spread
        varied_seed = (base_seed + offset) % (2**32)

        if progress_callback:
            progress_callback(f"Generating variation {i + 1}/{count}...")

        image, used_seed = generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            seed=varied_seed,
            progress_callback=None,
        )
        results.append((image, used_seed))

    return results


def unload_pipeline():
    """Free VRAM by unloading the pipeline."""
    global _pipeline
    if _pipeline is not None:
        del _pipeline
        _pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("[Z-Image-Turbo] Pipeline unloaded")


def is_loaded():
    """Check if the pipeline is currently loaded."""
    return _pipeline is not None
