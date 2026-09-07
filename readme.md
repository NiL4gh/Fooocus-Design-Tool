# 🎨 Fooocus Designer 2.0

**AI-Powered Graphic Design Asset Generator** — Built on RunDiffusion/Juggernaut-XL-v9 (SDXL) with baked specialized LoRAs for high-quality, production-ready graphic design asset generation.

> Forked from [Fooocus](https://github.com/lllyasviel/Fooocus) and completely rebuilt for graphic designers.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **⚡ Dual-Speed Modes** | ⚡ **Fast** (~3s via SDXL-Lightning 4-step) vs 🎯 **Master** (~15s via Juggernaut XL 25-step) |
| **🎯 Baked Design LoRAs** | Automated adapter routing for Silhouettes, Flat Vectors, Stickers, and Logos |
| **🎨 Fooocus Style Engine** | Multi-style `{prompt}` interpolation and negative stacking (`Fooocus V2`, `SAI Line Art`, `MRE Flat 2D Art`, etc.) |
| **🎨 Design Categories** | Adobe Stock presets (Silhouette, Flat Vector), Logo, Poster, Banner, Artwork, Sticker, Mockup |
| **🧠 Auto Prompt Enhancement** | Category-specific keywords and trigger words automatically improve your results |
| **🚫 Master Negative Prompts** | Per-category negative prompts eliminate common artifacts and 3D rendering clutter |
| **🎨 Color Palette Control** | Inject hex colors into generation via prompt engineering + post-processing |
| **🔲 Transparent PNG** | Automatic background removal (rembg) for logos, silhouettes, and icons |
| **✏️ Vector Mode (SVG)** | Convert rasters to lossless SVG using StarVector-1B |
| **✏️ Design Editing** | Simplified inpaint/img2img editing with strength control |
| **🔄 Variations** | Generate 2-4 similar designs with seed spread |
| **📦 Logo Mockup Engine** | Place transparent logos on photorealistic products (T-Shirt, Mug, Bento layout, Ambient) |
| **🧹 VRAM Management** | One-click GPU cache clear and automated cleanup for seamless Colab T4 operation |

## 🚀 Quick Start

### Local Installation

```bash
# Clone the repo
git clone https://github.com/NiL4gh/Fooocus-Design-Tool.git
cd Fooocus-Design-Tool

# Install dependencies
pip install -r requirements.txt

# Launch
python launch.py
```

The UI will open at `http://localhost:7865`

### Google Colab (Free T4 GPU)

```python
!git clone https://github.com/NiL4gh/Fooocus-Design-Tool.git
%cd Fooocus-Design-Tool
!pip install -r requirements.txt -q
!python launch.py --share
```

Or use the provided `colab_setup.py`:
```python
!python colab_setup.py
```

---

## 🎯 Design Categories & Baked LoRAs

| Category | Baked LoRA Adapter | Auto-Enhancement | Transparent BG | Best For |
|----------|-------------------|-----------------|----------------|----------|
| Adobe Stock Silhouette | `silhouette_lora` (0.85) | ✅ Black vector silhouette, white background | ✅ Default ON | Stock vector icons, clip art |
| Adobe Stock Flat Vector | `flat_vector_lora` (0.80) | ✅ Clean geometric vector illustration | ❌ | Commercial flat graphics |
| Logo | `logo_minimal` (0.85) | ✅ Minimalist, flat vector, scalable | ✅ Default ON | Brand logos, emblems |
| Sticker | `sticker_diecut` (0.85) | ✅ Die-cut border, clean outline | ✅ Default ON | Print stickers, badges |
| Poster | None (Pure Juggernaut XL) | ✅ Print quality, bold typography layout | ❌ | Event posters, ads |
| Banner | None (Pure Juggernaut XL) | ✅ Wide format, commercial digital marketing | ❌ | Web banners, headers |
| Artwork | None (Pure Juggernaut XL) | ✅ Concept art quality, vibrant illustration | ❌ | Art prints, digital painting |

---

## 🎨 Fooocus Style Engine & Style Presets

Fooocus Designer 2.0 incorporates the Fooocus Style Engine, driven by template JSON files located in `config/sdxl_styles/*.json` (including the Fooocus, SAI, and MRE style collections).

### How It Works
- **Multi-Style Interpolation**: Select multiple styles concurrently. Each style's positive prompt template injects the prompt text into `{prompt}` sequentially.
- **Negative Prompt Stacking**: Automatically stacks each selected style's negative additions with category-specific master negative prompts while deduplicating and cleanly stripping commas.
- **Zero Hallucination / Native Diffusers**: Runs directly on SDXL prompt conditioning without altering network weights or slowing down inference.

### Available Styles
Loaded dynamically from `config/sdxl_styles/*.json`:

| Style | Collection | Description |
|-------|------------|-------------|
| **Fooocus V2** | Fooocus | Flagship aesthetic expansion; boosts detail, composition, and realistic lighting |
| **Fooocus Masterpiece** | Fooocus | Artistic flair with rich textures, fine details, and painterly quality |
| **Fooocus Photograph** | Fooocus | Photorealistic camera parameters, natural skin tones, and lighting |
| **Fooocus Negative** | Fooocus | Comprehensive artifact reduction and rendering cleanup |
| **Fooocus Cinematic** | Fooocus | Dramatic cinematic lighting, anamorphic lens flares, and film contrast |
| **SAI Line Art** | Stability AI | Clean, crisp contour lines, minimalist line-drawing aesthetic, ink outlines |
| **SAI Digital Art** | Stability AI | Vibrant, modern digital illustration styling |
| **SAI Flat Color** | Stability AI | Flat color fills, reduced gradients, clean vector aesthetic |
| **SAI Origami** | Stability AI | Geometric folded paper style with clean dimensional shadows |
| **MRE Flat 2D Art** | MRE | Pure 2D flat design, crisp vector silhouettes, minimal shading |
| **MRE Comic Art** | MRE | Dynamic comic ink, cel-shading, and narrative illustration flair |

### 🧠 Smart Category Presets
The UI intelligently configures style defaults based on the chosen category:
- **Clean Vector & Silhouette Categories** (`Adobe Stock Silhouette`, `Adobe Stock Flat Vector`, `Logo`, `Sticker`):
  Isolate prompt tags and LoRA adapters by leaving style checkboxes unchecked by default. This prevents photorealistic or 3D shading bleed from compromising clean vector lines and cutouts.
- **Artwork & Poster Categories** (`Artwork`, `Poster`):
  Automatically enable **Fooocus V2** by default for rich compositional depth, cinematic lighting, and enhanced visual detail.
- **Full Customizability**: Users can freely select, combine, or deselect any combination of styles in the UI accordion at any time.

---

## ⚙️ Architecture

```
Fooocus-Design-Tool/
├── config/
│   ├── design_categories.json    # Category configurations & baked LoRAs
│   └── sdxl_styles/              # Fooocus, SAI, and MRE style JSON templates
├── modules/
│   ├── config.py                 # App configuration & aspect ratios
│   ├── sdxl_pipeline.py          # Primary SDXL Diffusers Engine (Juggernaut XL)
│   ├── lora_router.py            # LoRA loading, switching & weight management
│   ├── style_engine.py           # Multi-style prompt interpolation & negative stacking
│   ├── design_categories.py      # Category & LoRA metadata loader
│   ├── auto_prompt_enhancer.py   # Category prompt enhancement & negative builder
│   ├── palette_control.py        # Color palette prompt injection & post-processing
│   ├── starvector_pipeline.py    # StarVector-1B (lossless SVG conversion)
│   ├── background_remover.py     # rembg background removal wrapper
│   ├── variation.py              # Multi-seed variation generator
│   └── logo_mockup.py            # Dynamic product mockup compositing
├── ui/
│   ├── theme.py                  # Premium dark theme
│   ├── design_main.py            # Primary generation UI with dual-speed mode
│   ├── design_edit.py            # Inpaint / img2img editing tab
│   ├── design_variations.py      # Multi-output variation generator tab
│   └── design_mockup.py          # Logo product mockup staging tab
├── webui.py                      # Main Gradio application & tab layout
├── launch.py                     # Launcher script
├── colab_setup.py                # Automated Google Colab setup script
└── requirements.txt              # Production dependencies
```

## 🔧 Performance Benchmarks (Colab T4 16GB)

- **⚡ Fast Mode (SDXL-Lightning)**: ~3-4 seconds per 1024×1024 asset (5 steps, CFG 1.8)
- **🎯 Master Mode (Juggernaut XL)**: ~15-18 seconds per 1024×1024 asset (28 steps, CFG 6.0)
- **LoRA Adapter Switching**: <0.3s (cached in memory, dynamic weight scaling)
- **rembg Background Removal**: <1 second (CPU/ONNX)
- **StarVector-1B Vectorization**: ~10-15 seconds
- **VRAM Optimizations**: PyTorch SDPA, FP16 precision, VAE tiling/slicing (fits comfortably in 12-16GB VRAM with zero OOMs)

## 📄 License

GPL-3.0 (inherited from Fooocus)
