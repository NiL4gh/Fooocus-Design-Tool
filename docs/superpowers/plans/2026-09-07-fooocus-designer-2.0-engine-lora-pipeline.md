# Fooocus Designer 2.0: Engine & LoRA Pipeline Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the legacy 4-bit Flux / Z-Image generation engine with a native SDXL diffusers engine powered by Juggernaut XL, integrating a zero-reload PEFT multi-LoRA router and baking category LoRAs into the design presets for Colab and 12GB+ GPUs.

**Architecture:** A centralized `modules/sdxl_pipeline.py` wraps `diffusers.StableDiffusionXLPipeline` with native SDPA, VAE tiling/slicing, and dual-speed mode (ByteDance SDXL-Lightning 4-step/8-step scheduler vs full Juggernaut XL 25-step). A dedicated `modules/lora_router.py` manages loading, caching, scaling, and auto-trigger word injection for category presets defined in `config/design_categories.json`.

**Tech Stack:** Python 3.10+, PyTorch 2.1+, HuggingFace `diffusers` (>=0.25.0), `transformers`, `accelerate`, `safetensors`, Gradio 4.x, PIL, `unittest`.

## Global Constraints
- Target environment: Google Colab (T4 15GB GPU / 12.7GB CPU RAM) and local 12GB+ GPUs.
- All tensors and weights in FP16 on CUDA (`torch.float16`, `low_cpu_mem_usage=True`).
- Never perform large array or model loading without `gc.collect()` and `torch.cuda.empty_cache()`.
- VAE tiling (`pipe.vae.enable_tiling()`) and slicing (`pipe.vae.enable_slicing()`) must be enabled on CUDA to prevent 1024×1024 decode OOM.
- In test environments (`MOCK_IMAGE_GEN=1`), all pipeline and LoRA operations must run deterministically in memory without downloading checkpoints or requiring a GPU.
- Windows PowerShell compatibility: all scripts and commands must work cleanly in `pwsh`.

---

### Task 1: Create the Zero-Reload PEFT LoRA Router (`modules/lora_router.py`)

**Files:**
- Create: `modules/lora_router.py`
- Test: `tests/test_lora_router.py`

**Interfaces:**
- Consumes: `diffusers.StableDiffusionXLPipeline` instance, category LoRA configuration dictionaries.
- Produces: 
  - `apply_category_lora(pipeline, category_cfg, force_active=True) -> str`: Activates the category's LoRA adapter on the pipeline and returns the trigger words to append to the prompt.
  - `get_active_adapter() -> str | None`: Returns the currently active adapter name.
  - `clear_adapters(pipeline)`: Unloads or disables currently active LoRA adapters.

- [ ] **Step 1: Write the failing unit tests for LoRA Router**

Create `tests/test_lora_router.py`:
```python
import os
import unittest
os.environ["MOCK_IMAGE_GEN"] = "1"

from modules.lora_router import (
    apply_category_lora,
    get_active_adapter,
    clear_adapters,
    is_lora_available,
)

class DummyPipeline:
    def __init__(self):
        self.loaded_adapters = {}
        self.active_adapters = []

    def load_lora_weights(self, source, weight_name=None, adapter_name=None):
        self.loaded_adapters[adapter_name] = source

    def set_adapters(self, adapter_names, adapter_weights=None):
        self.active_adapters = adapter_names

    def disable_lora(self):
        self.active_adapters = []

class TestLoraRouter(unittest.TestCase):
    def setUp(self):
        self.pipe = DummyPipeline()
        clear_adapters(self.pipe)

    def test_apply_category_lora_with_valid_config(self):
        cfg = {
            "name": "Adobe Stock Silhouette",
            "lora": {
                "source": "models/loras/silhouette.safetensors",
                "weight": 0.85,
                "trigger_words": "solid black silhouette, vector style"
            }
        }
        trigger = apply_category_lora(self.pipe, cfg)
        self.assertEqual(trigger, "solid black silhouette, vector style")
        self.assertEqual(get_active_adapter(), "adobe_stock_silhouette")
        self.assertIn("adobe_stock_silhouette", self.pipe.active_adapters)

    def test_apply_category_lora_without_lora(self):
        cfg = {
            "name": "Artwork",
            "lora": None
        }
        trigger = apply_category_lora(self.pipe, cfg)
        self.assertEqual(trigger, "")
        self.assertIsNone(get_active_adapter())

    def test_clear_adapters(self):
        cfg = {
            "name": "Flat Vector",
            "lora": {
                "source": "models/loras/flat_vector.safetensors",
                "weight": 0.8,
                "trigger_words": "flat vector"
            }
        }
        apply_category_lora(self.pipe, cfg)
        self.assertIsNotNone(get_active_adapter())
        clear_adapters(self.pipe)
        self.assertIsNone(get_active_adapter())
        self.assertEqual(len(self.pipe.active_adapters), 0)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_lora_router.py`  
Expected: FAIL with `ModuleNotFoundError: No module named 'modules.lora_router'`

- [ ] **Step 3: Implement `modules/lora_router.py`**

Create `modules/lora_router.py`:
```python
"""
LoRA Router Module — Fooocus Designer 2.0
Manages dynamic loading, switching, and caching of PEFT LoRA adapters on SDXL pipelines.
Provides zero-reload switching and automatic trigger word extraction.
"""

import os
import re
from typing import Optional, Dict, Any

_loaded_adapters: Dict[str, str] = {}
_active_adapter: Optional[str] = None


def _sanitize_name(name: str) -> str:
    """Convert category name to a clean alphanumeric adapter ID."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name.strip().lower()).strip("_")


def is_lora_available(source: str) -> bool:
    """Check if a LoRA source exists locally or is a valid HuggingFace repo ID."""
    if not source:
        return False
    if os.path.exists(source):
        return True
    # If source looks like a HuggingFace repo (e.g. "org/model")
    if "/" in source and not source.startswith("/"):
        return True
    return False


def get_active_adapter() -> Optional[str]:
    """Return the name of the currently active adapter, if any."""
    return _active_adapter


def clear_adapters(pipeline) -> None:
    """Disable active LoRA adapters on the pipeline."""
    global _active_adapter
    if pipeline is not None and hasattr(pipeline, "disable_lora"):
        try:
            pipeline.disable_lora()
        except Exception as e:
            print(f"[LoRA Router] Warning disabling LoRA: {e}")
    _active_adapter = None


def apply_category_lora(pipeline, category_cfg: Dict[str, Any], force_active: bool = True) -> str:
    """
    Apply a category's baked LoRA to the pipeline.

    Args:
        pipeline: Diffusers SDXL pipeline instance or dummy test pipeline.
        category_cfg: Dictionary containing category configuration and optional "lora" block.
        force_active: Whether to activate the adapter immediately.

    Returns:
        The trigger words string to prepend/append to the generation prompt.
    """
    global _active_adapter, _loaded_adapters

    if not category_cfg or "lora" not in category_cfg or not category_cfg["lora"]:
        clear_adapters(pipeline)
        return ""

    lora_cfg = category_cfg["lora"]
    source = lora_cfg.get("source") or lora_cfg.get("repo_or_file")
    weight = float(lora_cfg.get("weight", 0.85))
    trigger_words = lora_cfg.get("trigger_words", "").strip()

    if not source:
        clear_adapters(pipeline)
        return trigger_words

    category_name = category_cfg.get("name", "custom")
    adapter_name = _sanitize_name(category_name)

    # In mock mode, update internal state without touching torch weights
    if os.environ.get("MOCK_IMAGE_GEN") == "1" or pipeline == "mock_pipeline":
        if hasattr(pipeline, "load_lora_weights"):
            pipeline.load_lora_weights(source, adapter_name=adapter_name)
        if hasattr(pipeline, "set_adapters"):
            pipeline.set_adapters([adapter_name], adapter_weights=[weight])
        _active_adapter = adapter_name
        return trigger_words

    if pipeline is None:
        return trigger_words

    try:
        # Load adapter if not already in cache
        if adapter_name not in _loaded_adapters:
            print(f"[LoRA Router] Loading adapter '{adapter_name}' from: {source}")
            weight_name = lora_cfg.get("weight_name")
            kwargs = {"adapter_name": adapter_name}
            if weight_name:
                kwargs["weight_name"] = weight_name
            
            pipeline.load_lora_weights(source, **kwargs)
            _loaded_adapters[adapter_name] = source

        # Set active adapter and scale weight
        if force_active and hasattr(pipeline, "set_adapters"):
            pipeline.set_adapters([adapter_name], adapter_weights=[weight])
            _active_adapter = adapter_name
            print(f"[LoRA Router] Activated adapter '{adapter_name}' with weight {weight}")

    except Exception as e:
        print(f"[LoRA Router] Failed to load/set LoRA '{adapter_name}': {e}. Continuing without LoRA.")
        clear_adapters(pipeline)

    return trigger_words
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_lora_router.py`  
Expected: PASS (Ran 3 tests in <0.05s, OK)

- [ ] **Step 5: Commit Task 1**

```bash
git add modules/lora_router.py tests/test_lora_router.py
git commit -m "feat: implement zero-reload PEFT LoRA router for category presets"
```

---

### Task 2: Create SDXL Diffusers Engine with Juggernaut XL & Dual-Speed Mode (`modules/sdxl_pipeline.py`)

**Files:**
- Create: `modules/sdxl_pipeline.py`
- Modify: `modules/config.py`
- Test: `tests/test_sdxl_pipeline.py`

**Interfaces:**
- Consumes: PyTorch, HuggingFace Diffusers (`StableDiffusionXLPipeline`, `DPMSolverMultistepScheduler`, `EulerDiscreteScheduler`), `modules/lora_router.py`.
- Produces:
  - `load_pipeline(speed_mode="fast", progress_callback=None)`: Loads Juggernaut XL in FP16 with SDPA and VAE tiling.
  - `generate(prompt, negative_prompt="", width=1024, height=1024, seed=-1, speed_mode="fast", category_cfg=None, progress_callback=None) -> (PIL.Image, int)`: Runs SDXL inference with baked LoRA trigger injection.
  - `unload_pipeline()`: Frees GPU VRAM and clears CUDA cache.
  - `is_loaded() -> bool`: Returns whether pipeline is currently in memory.

- [ ] **Step 1: Write the failing unit tests for SDXL Pipeline**

Create `tests/test_sdxl_pipeline.py`:
```python
import os
import unittest
from PIL import Image

os.environ["MOCK_IMAGE_GEN"] = "1"

from modules.sdxl_pipeline import (
    load_pipeline,
    generate,
    generate_variations,
    unload_pipeline,
    is_loaded,
)

class TestSDXLPipeline(unittest.TestCase):
    def tearDown(self):
        unload_pipeline()

    def test_mock_load_and_is_loaded(self):
        self.assertFalse(is_loaded())
        pipe = load_pipeline()
        self.assertEqual(pipe, "mock_pipeline")
        self.assertTrue(is_loaded())

    def test_mock_generate_fast_mode(self):
        prompt = "flat vector mountain icon #3b82f6"
        image, seed = generate(prompt, width=512, height=512, seed=1234, speed_mode="fast")
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (512, 512))
        self.assertEqual(seed, 1234)

    def test_mock_generate_with_category_cfg(self):
        cat_cfg = {
            "name": "Adobe Stock Silhouette",
            "lora": {
                "source": "models/loras/silhouette.safetensors",
                "weight": 0.85,
                "trigger_words": "solid black silhouette"
            }
        }
        image, seed = generate(
            "owl on branch",
            width=512,
            height=512,
            seed=42,
            speed_mode="master",
            category_cfg=cat_cfg
        )
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(seed, 42)

    def test_mock_generate_variations(self):
        variations = generate_variations(
            prompt="geometric logo",
            width=256,
            height=256,
            base_seed=100,
            count=2,
            speed_mode="fast"
        )
        self.assertEqual(len(variations), 2)
        for img, s in variations:
            self.assertIsInstance(img, Image.Image)
            self.assertEqual(img.size, (256, 256))

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_sdxl_pipeline.py`  
Expected: FAIL with `ModuleNotFoundError: No module named 'modules.sdxl_pipeline'`

- [ ] **Step 3: Implement `modules/sdxl_pipeline.py`**

Create `modules/sdxl_pipeline.py`:
```python
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

from modules.lora_router import apply_category_lora, clear_adapters

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

        # Configure scheduler for speed mode
        if speed_mode == "fast":
            # Load 4-step Lightning configuration
            from diffusers import EulerDiscreteScheduler
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
                _pipeline.set_adapters(["lightning_fast"], adapter_weights=[1.0])
            except Exception as e:
                print(f"[SDXL Pipeline] Lightning adapter notice: {e}")
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
        speed_mode: "fast" (5 steps) or "master" (25 steps).
        category_cfg: Category configuration containing optional baked LoRA.
        progress_callback: Progress reporting callable.

    Returns:
        Tuple of (PIL.Image, used_seed).
    """
    if seed == -1 or seed is None:
        seed = random.randint(0, 2**32 - 1)

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

    pipe = load_pipeline(speed_mode=speed_mode, progress_callback=progress_callback)

    # Apply category baked LoRA if present
    trigger_words = ""
    if category_cfg:
        trigger_words = apply_category_lora(pipe, category_cfg)

    final_prompt = f"{trigger_words}, {prompt}".strip(", ") if trigger_words else prompt

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
```

- [ ] **Step 4: Update `modules/config.py` defaults for SDXL**

Edit `modules/config.py` lines 17–24 to reflect SDXL defaults:
```python
# Generation defaults
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_SPEED_MODE = "fast"
DEFAULT_STEPS_FAST = 6
DEFAULT_STEPS_MASTER = 28
DEFAULT_GUIDANCE_FAST = 1.8
DEFAULT_GUIDANCE_MASTER = 6.0
DEFAULT_MAX_IMAGE_NUMBER = 1
DEFAULT_OUTPUT_FORMAT = 'png'
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m unittest tests/test_sdxl_pipeline.py`  
Expected: PASS (Ran 4 tests in <0.1s, OK)

- [ ] **Step 6: Commit Task 2**

```bash
git add modules/sdxl_pipeline.py modules/config.py tests/test_sdxl_pipeline.py
git commit -m "feat: implement SDXL diffusers pipeline with Juggernaut XL and dual-speed mode"
```

---

### Task 3: Bake LoRA Presets into Categories & Enhance Loader (`config/design_categories.json` & `modules/design_categories.py`)

**Files:**
- Modify: `config/design_categories.json`
- Modify: `modules/design_categories.py`
- Test: `tests/test_design_categories.py`

**Interfaces:**
- Consumes: `config/design_categories.json`.
- Produces:
  - `get_category_lora(category_name: str) -> Optional[Dict[str, Any]]`: Returns baked LoRA source, weight, and trigger words.
  - `load_categories() -> List[Dict[str, Any]]`: Returns categories with LoRA configs included.

- [ ] **Step 1: Write the failing test for baked category LoRAs**

Create `tests/test_design_categories.py`:
```python
import unittest
from modules.design_categories import (
    load_categories,
    get_category,
    get_category_lora,
    get_category_names,
)

class TestDesignCategories(unittest.TestCase):
    def test_baked_loras_exist_in_categories(self):
        silhouette = get_category("Adobe Stock Silhouette")
        self.assertIsNotNone(silhouette)
        lora_cfg = get_category_lora("Adobe Stock Silhouette")
        self.assertIsNotNone(lora_cfg)
        self.assertIn("trigger_words", lora_cfg)
        self.assertIn("weight", lora_cfg)
        self.assertGreaterEqual(lora_cfg["weight"], 0.7)

        vector = get_category("Adobe Stock Flat Vector")
        self.assertIsNotNone(vector)
        v_lora = get_category_lora("Adobe Stock Flat Vector")
        self.assertIsNotNone(v_lora)
        self.assertIn("flat", v_lora["trigger_words"].lower())

    def test_category_without_lora(self):
        artwork = get_category("Artwork")
        self.assertIsNotNone(artwork)
        lora_cfg = get_category_lora("Artwork")
        self.assertIsNone(lora_cfg)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_design_categories.py`  
Expected: FAIL with `AttributeError: module 'modules.design_categories' has no attribute 'get_category_lora'`

- [ ] **Step 3: Update `config/design_categories.json` with baked LoRA configs**

Update `config/design_categories.json` to include `"lora"` metadata for Silhouette, Flat Vector, Sticker, and Logo:
```json
{
  "categories": [
    {
      "name": "Adobe Stock Silhouette",
      "master_negative": "color, grayscale, shading, shadows, realistic, photograph, 3d render, blurry, noisy, text, watermark, background textures, complex details, gradient background",
      "prompt_enhancement": "isolated solid black silhouette, flat design, clean vector style, sharp edges, high contrast, minimalist, clean SVG path vector, clip art, graphic asset, professional commercial asset",
      "default_transparent": true,
      "default_aspect_ratio": "1024*1024",
      "icon": "👤",
      "lora": {
        "source": "DoctorDiffusion/doctor-diffusion-s-stylized-silhouette-photography-xl-lora",
        "weight": 0.9,
        "trigger_words": "solid black silhouette, vector style"
      }
    },
    {
      "name": "Adobe Stock Flat Vector",
      "master_negative": "photograph, realistic photo, 3d render, gradients, shading, blurry, noisy, shadows, texture, text, watermark, human face, complex detailed background",
      "prompt_enhancement": "flat minimalist vector graphic, clean vector style, solid colors, smooth clean outlines, isolated, commercial graphic asset, Adobe Stock style vector, scalable graphic element, 2d illustration",
      "default_transparent": true,
      "default_aspect_ratio": "1024*1024",
      "icon": "💎",
      "lora": {
        "source": "DoctorDiffusion/doctor-diffusion-s-controllable-vector-art-xl-lora",
        "weight": 0.85,
        "trigger_words": "flat minimalist vector graphic, vector art style"
      }
    },
    {
      "name": "Adobe Stock Sticker/Clipart",
      "master_negative": "gradients, shading, blurry, noisy, shadows, texture, text, watermark, complex detailed background, photograph, realistic photo, 3d render",
      "prompt_enhancement": "die-cut sticker design, clean white border, cute clipart asset, flat color vector illustration, isolated white background, commercial graphic asset, clean paths, vector patch, high contrast",
      "default_transparent": true,
      "default_aspect_ratio": "1024*1024",
      "icon": "🏷️",
      "lora": {
        "source": "artificialguybr/StickersRedmond",
        "weight": 0.8,
        "trigger_words": "die-cut sticker, white contour border"
      }
    },
    {
      "name": "Logo",
      "master_negative": "photograph, realistic photo, 3d render, blurry, noisy, text artifacts, watermark, human, person, face, body, animal, landscape, complex background, gradient background",
      "prompt_enhancement": "minimalist logo design, flat vector style, scalable, clean lines, professional branding, centered composition, high contrast",
      "default_transparent": true,
      "default_aspect_ratio": "1024*1024",
      "icon": "🎯",
      "lora": {
        "source": "artificialguybr/LogoRedmond-LogoLoraForSDXL-V2",
        "weight": 0.8,
        "trigger_words": "clean branding vector logo, centered composition"
      }
    },
    {
      "name": "Adobe Stock Seamless Pattern",
      "master_negative": "photograph, realistic, 3d render, blurry, noisy, text, watermark, border, frame, edges, asymmetric, non-repeating",
      "prompt_enhancement": "seamless repeating pattern, vector illustration tile, beautiful continuous design element, isolated motif, commercial textile backdrop, flat design, high resolution, symmetrical tileable",
      "default_transparent": false,
      "default_aspect_ratio": "1024*1024",
      "icon": "🌀",
      "lora": null
    },
    {
      "name": "Poster",
      "master_negative": "low quality, blurry, pixelated, watermark, text artifacts, deformed, ugly, bad anatomy, cropped",
      "prompt_enhancement": "professional poster design, high resolution, print quality, graphic design, vivid colors, bold composition, editorial layout",
      "default_transparent": false,
      "default_aspect_ratio": "768*1280",
      "icon": "📰",
      "lora": null
    },
    {
      "name": "Banner",
      "master_negative": "low quality, blurry, pixelated, watermark, text artifacts, deformed, cropped edges",
      "prompt_enhancement": "web banner design, wide format, professional, clean layout, modern design, digital marketing, high resolution",
      "default_transparent": false,
      "default_aspect_ratio": "1536*640",
      "icon": "🖼️",
      "lora": null
    },
    {
      "name": "Vector Silhouette",
      "master_negative": "photograph, realistic, 3d render, blurry, noisy, watermark, complex shading, gradients, texture, detailed background",
      "prompt_enhancement": "black silhouette on white background, clean vector style, flat design, sharp edges, minimalist, icon design, scalable graphic",
      "default_transparent": true,
      "default_aspect_ratio": "1024*1024",
      "icon": "✂️",
      "lora": {
        "source": "DoctorDiffusion/doctor-diffusion-s-stylized-silhouette-photography-xl-lora",
        "weight": 0.9,
        "trigger_words": "solid black silhouette, minimalist icon"
      }
    },
    {
      "name": "Artwork",
      "master_negative": "low quality, blurry, watermark, text, deformed, ugly, bad composition, amateur",
      "prompt_enhancement": "professional digital artwork, high quality illustration, detailed, artistic, beautiful composition, vibrant colors, concept art quality",
      "default_transparent": false,
      "default_aspect_ratio": "1024*1024",
      "icon": "🎨",
      "lora": null
    },
    {
      "name": "Stock Image",
      "master_negative": "blurry, low quality, watermark, text overlay, deformed, ugly, bad anatomy, distorted, amateur photography",
      "prompt_enhancement": "professional stock photography style, high resolution, commercial quality, well-lit, clean composition, editorial quality, authentic",
      "default_transparent": false,
      "default_aspect_ratio": "1152*896",
      "icon": "📷",
      "lora": null
    },
    {
      "name": "Stock Image Mockup",
      "master_negative": "blurry, low quality, watermark, deformed, ugly, unrealistic lighting, amateur",
      "prompt_enhancement": "professional product mockup, studio photography, clean background, commercial quality, realistic lighting, professional presentation",
      "default_transparent": false,
      "default_aspect_ratio": "1024*1024",
      "icon": "📦",
      "lora": null
    },
    {
      "name": "Edit Design/Photo",
      "master_negative": "",
      "prompt_enhancement": "",
      "default_transparent": false,
      "default_aspect_ratio": "1024*1024",
      "icon": "✏️",
      "lora": null
    },
    {
      "name": "Variation",
      "master_negative": "",
      "prompt_enhancement": "",
      "default_transparent": false,
      "default_aspect_ratio": "1024*1024",
      "icon": "🔄",
      "lora": null
    }
  ]
}
```

- [ ] **Step 4: Update `modules/design_categories.py` to expose `get_category_lora`**

Add to `modules/design_categories.py`:
```python
def get_category_lora(name: str):
    """Get the baked LoRA configuration for a category, if any."""
    cat = get_category(name)
    if cat and "lora" in cat and cat["lora"]:
        return cat["lora"]
    return None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m unittest tests/test_design_categories.py`  
Expected: PASS (Ran 2 tests in <0.05s, OK)

- [ ] **Step 6: Commit Task 3**

```bash
git add config/design_categories.json modules/design_categories.py tests/test_design_categories.py
git commit -m "feat: bake category-specific LoRAs into presets and expose loader helper"
```

---

### Task 4: Connect SDXL Pipeline & Baked LoRAs to the Turnkey UI (`ui/design_main.py` & `webui.py`)

**Files:**
- Modify: `ui/design_main.py`
- Modify: `webui.py`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `modules.sdxl_pipeline`, `modules.design_categories`, `modules.lora_router`.
- Produces:
  - Interactive Turnkey UI with Speed Switch (⚡ Fast vs 🎯 Master) and category LoRA wiring.
  - VRAM cleaner mapped to `sdxl_pipeline.unload_pipeline()`.

- [ ] **Step 1: Update `ui/design_main.py` to wire `sdxl_pipeline` and speed toggle**

Update the imports and generation dispatch in `ui/design_main.py`:
- Replace `from modules.zimage_pipeline import generate as zimage_generate` with `from modules.sdxl_pipeline import generate as sdxl_generate`.
- Replace the legacy "AI Model" dropdown with a clean **Speed Mode Radio** (`⚡ Fast (~3s)` vs `🎯 Master (~15s)`).
- Pass `speed_mode` and the retrieved `category_cfg` from `get_category(category)` into `sdxl_generate`.

```python
# In ui/design_main.py:
def _generate(category, prompt, negative_prompt, color1, color2, color3, color4, color5,
              use_master_neg, use_enhancement, remove_bg, vector_mode, concept_grid, aspect_ratio, seed_val, speed_mode_label):
    ...
    speed_mode = "fast" if "Fast" in speed_mode_label else "master"
    cat_cfg = get_category(category) if category else None

    # Call sdxl_generate with baked LoRA category config
    image, used_seed = sdxl_generate(
        prompt=final_prompt,
        negative_prompt=final_negative,
        width=width,
        height=height,
        seed=seed,
        speed_mode=speed_mode,
        category_cfg=cat_cfg
    )
```

- [ ] **Step 2: Update `webui.py` VRAM cleaner to unload `sdxl_pipeline`**

In `webui.py`:
- Replace `from modules.zimage_pipeline import unload_pipeline, is_loaded as is_z` with `from modules.sdxl_pipeline import unload_pipeline, is_loaded as is_sdxl`.
- Update `clean_vram()` to unload SDXL:
```python
if is_sdxl():
    unload_pipeline()
    freed.append("Juggernaut-XL (SDXL)")
```

- [ ] **Step 3: Update existing test suite in `tests/test_app.py`**

In `tests/test_app.py`:
- Update import from `modules.zimage_pipeline` to `modules.sdxl_pipeline`.
- Verify all tests pass including palette injection, category loading, vectorization, and background removal.

- [ ] **Step 4: Run full test suite**

Run: `python -m unittest discover tests`  
Expected: PASS (All tests passing)

- [ ] **Step 5: Commit Task 4**

```bash
git add ui/design_main.py webui.py tests/test_app.py
git commit -m "feat: wire SDXL pipeline and speed mode toggle into Gradio UI"
```

---

### Task 5: End-to-End Colab Verification & Cleanup

**Files:**
- Modify: `colab_setup.py`
- Modify: `readme.md`
- Delete: `modules/zimage_pipeline.py` (legacy code cleanup)

**Interfaces:**
- Produces: Working Colab setup script downloading SDXL diffusers requirements and running verification.

- [ ] **Step 1: Remove deprecated `modules/zimage_pipeline.py`**

Run: `git rm modules/zimage_pipeline.py`  
Expected: File deleted from git tracking.

- [ ] **Step 2: Update `colab_setup.py` for SDXL Diffusers**

Ensure `colab_setup.py` installs clean dependencies without bitsandbytes 4-bit Flux baggage, and launches with `--share`.

```python
"""
Google Colab Bootstrap Script for Fooocus Designer 2.0
"""
import subprocess
import sys

def setup():
    print("🎨 Setting up Fooocus Designer 2.0 (SDXL Engine)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"], check=True)
    print("🚀 Launching Fooocus Designer 2.0 with public Gradio link...")
    subprocess.run([sys.executable, "launch.py", "--share"])

if __name__ == "__main__":
    setup()
```

- [ ] **Step 3: Update `readme.md` architecture and feature list**

Update `readme.md` to reflect:
- Juggernaut XL SDXL Engine (replacing Z-Image/Flux).
- Baked LoRA Presets (Stock Silhouette, Flat Vector, Stickers, Logos).
- Dual-speed inference (⚡ Fast 3s vs 🎯 Master 15s).

- [ ] **Step 4: Run full automated verification**

Run: `python -m unittest discover -s tests -p "test_*.py"`  
Expected: All tests pass cleanly.

- [ ] **Step 5: Commit Task 5**

```bash
git add colab_setup.py readme.md
git commit -m "chore: clean up legacy Z-Image pipeline, update Colab launcher and documentation"
```
