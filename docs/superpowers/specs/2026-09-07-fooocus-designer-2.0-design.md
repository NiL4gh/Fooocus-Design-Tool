# Fooocus Designer 2.0 — Architecture & Design Specification

**Date:** 2026-09-07  
**Status:** Validated / Ready for Review  
**Target Environment:** Google Colab (NVIDIA T4 / L4 / A100) & 12GB+ GPUs  
**Repository:** `Fooocus-Design-Tool`

---

## 1. Problem Statement & Vision

Current generative AI tools (Automatic1111, ComfyUI, raw Fooocus) are designed for AI engineers who enjoy tweaking dozens of technical parameters (CFG, schedulers, denoise strength, LoRA stack weights). 

**Fooocus Designer 2.0** is built on the opposite philosophy: **a turnkey graphic design appliance**. 
A commercial asset designer should be able to:
1. Open the tool in Google Colab with 1 click.
2. Pick an asset category (*Stock Silhouette, Flat Vector, Die-Cut Sticker, Minimalist Logo, Seamless Pattern, Texture/Background, or Poster*).
3. Type a simple subject prompt (*e.g., "origami geometric wolf"*).
4. Click **Generate** and receive a production-ready asset (isolated on transparent background, color-harmonized, and optionally vectorized to SVG) without touching technical sliders.

---

## 2. Core Architecture

```
Fooocus-Design-Tool/
├── config/
│   ├── design_categories.json      # Asset presets with baked LoRAs & parameters
│   └── sdxl_styles/                # Fooocus style JSON library ({prompt} templates)
├── modules/
│   ├── sdxl_pipeline.py            # Primary SDXL Diffusers Engine (Replaces zimage_pipeline)
│   ├── lora_router.py              # Zero-reload PEFT multi-LoRA manager & auto-trigger injector
│   ├── design_categories.py        # Preset loader & category configuration manager
│   ├── auto_prompt_enhancer.py     # Dual-mode prompt expansion (Design Tags vs Fooocus GPT-2)
│   ├── palette_control.py          # Color harmony prompt injection & post-processing
│   ├── background_remover.py       # rembg GPU/CPU transparency engine
│   ├── starvector_pipeline.py      # Raster-to-SVG vectorizer engine
│   ├── metadata_manager.py         # PNG Info embedding (Fooocus-MRE style reproduction)
│   └── config.py                   # Global dimensions, aspect ratios, paths
├── ui/
│   ├── theme.py                    # Clean designer dark theme
│   ├── design_main.py              # Turnkey Generate tab
│   ├── design_edit.py              # Simplified img2img modification tab
│   ├── design_variations.py        # Seed mixing variation generator
│   └── design_mockup.py            # Commercial product mockup generator
├── colab_setup.py                  # Zero-touch 1-click Colab bootstrapping
├── fooocus_design_colab.ipynb      # Standalone Colab notebook
├── launch.py                       # Application launcher
└── webui.py                        # Gradio application server
```

---

## 3. Subsystem Specifications

### 3.1 Primary Engine (`modules/sdxl_pipeline.py`)
Replaces `zimage_pipeline.py` (Flux-4bit / Z-Image) with standard HuggingFace `diffusers.StableDiffusionXLPipeline`.

- **Base Checkpoint**: `RunDiffusion/Juggernaut-XL-v9` (FP16 safetensors).
- **Dual-Speed Execution Mode**:
  - **⚡ Fast Mode (Brainstorming / Iteration)**: Dynamically injects ByteDance's `SDXL-Lightning` 4-step/8-step scheduler/adapter. Runs at 5–6 steps, CFG 1.5–2.0, producing 1024×1024 assets in ~3.5 seconds on Colab T4.
  - **🎯 Master Mode (Final Asset Export)**: Runs native Juggernaut-XL at 25–30 steps (DPM++ 2M Karras), CFG 6.0, allowing category LoRAs to apply 100% of their trained detail.
- **Colab Memory Profile (Zero-OOM Guarantee)**:
  - `torch.float16` with `low_cpu_mem_usage=True` to eliminate system RAM spikes.
  - Native PyTorch 2.x SDPA (`scaled_dot_product_attention`) enabled by default.
  - VAE Tiling (`pipe.vae.enable_tiling()`) and VAE Slicing (`pipe.vae.enable_slicing()`) enabled to constrain high-res decode memory under 500MB.
  - VRAM target: **7.8 – 8.5 GB** total utilization.

### 3.2 Baked LoRA Router (`modules/lora_router.py`)
- Built on HuggingFace `diffusers` PEFT adapter interface (`load_lora_weights`, `set_adapters`, `disable_lora`).
- Maintains an in-memory cache of loaded LoRAs. Switching between categories or altering weights takes **<10 milliseconds** without reloading the base model.
- Automatically handles trigger words: when a category requires specific trigger phrases, the router injects them directly into the conditioning prompt.
- **Graceful Fallback**: If a remote LoRA is unreachable or offline, the pipeline proceeds with category prompt enhancement rather than throwing fatal exceptions.

### 3.3 Asset Category Presets (`config/design_categories.json`)
Each preset acts as a complete recipe:

1. **👤 Adobe Stock Silhouette**:
   - LoRA: Dedicated Silhouette LoRA (weight: 0.90)
   - Trigger: `solid black silhouette, vector style`
   - Enhancer: Tag-based isolated black on white background
   - Master Negative: Suppresses color, 3D realism, shading, gradients
   - Defaults: Transparent PNG ON, 1024×1024
2. **💎 Adobe Stock Flat Vector**:
   - LoRA: Flat Vector LoRA (weight: 0.85)
   - Trigger: `flat minimalist vector graphic`
   - Enhancer: Clean vector paths, solid colors, smooth outlines
   - Master Negative: Suppresses photorealism, textures, 3D render
   - Defaults: Transparent PNG ON, 1024×1024
3. **🏷️ Die-Cut Sticker / Clipart**:
   - LoRA: Sticker LoRA (weight: 0.80)
   - Trigger: `die-cut sticker, white contour border`
   - Defaults: Transparent PNG ON, 1024×1024
4. **🎯 Minimalist Logo**:
   - LoRA: Minimalist Logo LoRA (weight: 0.80)
   - Trigger: `clean branding vector logo, centered composition`
   - Defaults: Transparent PNG ON, 1024×1024, optional 2×2 Concept Grid
5. **🌀 Seamless Pattern**:
   - Enhancer: Continuous repeating tileable pattern
   - Defaults: Transparent PNG OFF, 1024×1024
6. **🪵 Backgrounds & Textures**:
   - Enhancer: High resolution commercial wallpaper/texture
   - Defaults: Transparent PNG OFF, 1536×640 (Wide Banner)
7. **🎨 Artwork & Poster**:
   - Enhancer: Unlocks creative Fooocus prompt expansion
   - Defaults: Transparent PNG OFF, 768×1280 (Portrait Poster)

### 3.4 Dual-Mode Prompt Expansion (`modules/auto_prompt_enhancer.py`)
- **Deterministic Design Tag Expansion (Default for Vectors/Silhouettes/Logos)**:
  - Formulates strict, tag-based prompts adhering to microstock guidelines (shape, structure, flat color, no photographic filler).
- **Fooocus Creative Expansion (Optional for Artwork/Posters)**:
  - Reads Fooocus style templates (`sdxl_styles/*.json`) for artistic and cinematic variations.

### 3.5 Metadata & Asset Management (`modules/metadata_manager.py`)
- Borrowed from **Fooocus-MRE**: Every generated PNG embeds a JSON metadata dictionary in its PNG `tEXt` chunks containing:
  `{"category": ..., "prompt": ..., "seed": ..., "model": ..., "loras": ..., "colors": ..., "version": "2.0"}`
- Enables single-click prompt & setting recovery by dropping an image back into the UI.

---

## 4. Implementation Plan (Phase Breakdown)

- **Phase 1: Engine & LoRA Pipeline Replacement**
  - Create `modules/sdxl_pipeline.py` with Juggernaut XL + SDXL-Lightning support.
  - Implement `modules/lora_router.py` for dynamic adapter loading and trigger injection.
  - Update `modules/config.py` and remove legacy Z-Image/Flux 4-bit baggage.
- **Phase 2: Preset LoRA Baking & Categories**
  - Update `config/design_categories.json` with baked LoRA configs.
  - Refactor `modules/auto_prompt_enhancer.py` with dual-mode tag expansion.
  - Port Fooocus style JSON loader.
- **Phase 3: Turnkey UI & Interactive Controls**
  - Refactor `ui/design_main.py` into the Turnkey Appliance UX.
  - Add Speed toggle (⚡ Fast vs 🎯 Master) and Color Palette swatches.
  - Wire metadata embedding.
- **Phase 4: Colab Bootstrapper & Verification**
  - Update `colab_setup.py` and `fooocus_design_colab.ipynb` for 1-click startup.
  - Run full verification suite on mock and CUDA environments.
