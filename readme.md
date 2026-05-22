# 🎨 Fooocus Design Tool

**AI-Powered Graphic Design Asset Generator** — Built on Z-Image-Turbo for fast, high-quality design asset generation.

> Forked from [Fooocus](https://github.com/lllyasviel/Fooocus) and completely rebuilt for graphic designers.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **🎯 Design Categories** | Logo, Poster, Banner, Vector Silhouette, Artwork, Stock Image, Mockup |
| **🧠 Auto Prompt Enhancement** | Category-specific keywords automatically improve your results |
| **🚫 Master Negative Prompts** | Per-category negative prompts eliminate common artifacts |
| **🎨 Color Palette Control** | Inject hex colors into generation via prompt engineering |
| **🔲 Transparent PNG** | Automatic background removal (rembg) for logos and icons |
| **✏️ Vector Mode (SVG)** | Convert rasters to SVG using StarVector-1B (optional) |
| **✏️ Design Editing** | Simplified img2img editing with strength control |
| **🔄 Variations** | Generate 2-4 similar designs with seed mixing |
| **📦 Logo Mockup** | Product mockup generation (coming soon) |

## 🚀 Quick Start

### Local Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/Fooocus-Design-Tool.git
cd Fooocus-Design-Tool

# Install dependencies
pip install -r requirements.txt

# Launch
python launch.py
```

The UI will open at `http://localhost:7865`

### Google Colab (Free GPU)

```python
!git clone https://github.com/YOUR_USERNAME/Fooocus-Design-Tool.git
%cd Fooocus-Design-Tool
!pip install -r requirements.txt -q
!python launch.py --share
```

Or use the provided `colab_setup.py`:
```python
!python colab_setup.py
```

---

## 🎯 Design Categories

| Category | Auto-Enhancement | Transparent BG | Best For |
|----------|-----------------|----------------|----------|
| Logo | ✅ Minimalist, flat vector, scalable | ✅ Default ON | Brand logos, icons |
| Poster | ✅ Print quality, bold composition | ❌ | Event posters, ads |
| Banner | ✅ Wide format, digital marketing | ❌ | Web banners, headers |
| Vector Silhouette | ✅ Black silhouette, flat design | ✅ Default ON | Icons, clip art |
| Artwork | ✅ Concept art quality, vibrant | ❌ | Illustrations, art prints |
| Stock Image | ✅ Commercial quality, authentic | ❌ | Stock photography |
| Stock Image Mockup | ✅ Studio photography, product | ❌ | Product presentations |

---

## ⚙️ Architecture

```
Fooocus-Design-Tool/
├── config/
│   └── design_categories.json    # Category configs
├── modules/
│   ├── config.py                 # App configuration
│   ├── design_categories.py      # Category loader
│   ├── auto_prompt_enhancer.py   # Prompt enhancement
│   ├── palette_control.py        # Color palette control
│   ├── zimage_pipeline.py        # Z-Image-Turbo (raster)
│   ├── starvector_pipeline.py    # StarVector-1B (SVG)
│   ├── background_remover.py     # rembg wrapper
│   ├── variation.py              # Seed-mixing variations
│   └── logo_mockup.py            # Mockup stub
├── ui/
│   ├── theme.py                  # Premium dark theme
│   ├── design_main.py            # Generate tab
│   ├── design_edit.py            # Edit tab
│   ├── design_variations.py      # Variations tab
│   └── design_mockup.py          # Mockup tab (stub)
├── webui.py                      # Main app
├── launch.py                     # Launcher
├── entry_with_update.py          # Entry point
└── requirements.txt              # Dependencies
```

## 🔧 Performance (Colab T4)

- **Z-Image-Turbo**: ~3 seconds per image at 1024×1024
- **rembg**: <1 second (CPU)
- **StarVector-1B**: ~10-15 seconds (first use includes download)
- **VRAM usage**: ~8-10GB at FP16 with CPU offload

## 📄 License

GPL-3.0 (inherited from Fooocus)
