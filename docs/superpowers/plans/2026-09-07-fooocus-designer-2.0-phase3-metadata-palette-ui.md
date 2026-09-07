# Fooocus Designer 2.0 Phase 3: Metadata Embedding, Color Swatches, and Turnkey UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Fooocus-MRE style PNG metadata embedding and recovery (`modules/metadata_manager.py`), curated commercial design color palette presets (`modules/palette_control.py`), drag-and-drop parameter restoration in `ui/design_main.py`, and unified ⚡ Fast vs 🎯 Master engine controls across Mockup, Edit, and Variations tabs.

**Architecture:** 
- `modules/metadata_manager.py` utilizes PIL's `PngInfo` to embed structured JSON metadata (`fooocus_designer_metadata` and standard A1111/Fooocus `parameters` chunks) at save time, and extracts them reliably upon file drop.
- `modules/palette_control.py` exposes curated 5-color palettes (Pastel Dreams, Cyberpunk Neon, Earthy Boho, Corporate Tech, Retro Sunset, Luxury Gold, Nordic Minimalist) that populate the UI color pickers with 1 click.
- `ui/design_main.py` saves images with complete reproduction metadata, exposes a drop-zone for instant settings recovery, and links palette presets to the color pickers.
- `ui/design_mockup.py`, `ui/design_edit.py`, and `ui/design_variations.py` are harmonized with SDXL engine modes and metadata embedding.

**Tech Stack:** Python 3.9+, Pillow (PIL `PngImagePlugin`), Gradio 3.x/4.x, HuggingFace Diffusers, PyTorch, unittest.

## Global Constraints
- **Test Mode Parity:** When `MOCK_IMAGE_GEN=1`, all operations (saving, loading, metadata parsing, UI generation) must execute deterministically in memory without downloading models or requiring GPU access.
- **Cross-Platform & Windows:** PowerShell-compatible commands; forward slashes in URLs/links; backslashes in Windows file paths.
- **Non-Destructive Metadata:** Embedding metadata in PNG `tEXt` / `iTXt` chunks must never alter image pixel data or dimensions.
- **Turnkey Simplicity:** No technical sliders exposed; sensible designer defaults always provided.
- **Zero Regressions:** All 52 existing unit and integration tests must continue to pass.

---

### Task 1: Create Metadata Manager (`modules/metadata_manager.py`)

**Files:**
- Create: `modules/metadata_manager.py`
- Test: `tests/test_metadata_manager.py`

**Interfaces:**
- Produces:
  - `build_metadata(category, prompt, negative_prompt, seed, speed_mode, selected_styles, colors, width, height, loras=None, version="2.0") -> dict`
  - `save_image_with_metadata(image: PIL.Image.Image, filepath: str, metadata: dict, fmt: str = "png") -> str`
  - `extract_metadata(image_or_path: Union[PIL.Image.Image, str]) -> Optional[dict]`
  - `format_metadata_for_display(metadata: dict) -> str`

- [ ] **Step 1: Write the failing tests in `tests/test_metadata_manager.py`**

```python
import os
import tempfile
import unittest
from PIL import Image
from modules.metadata_manager import (
    build_metadata,
    save_image_with_metadata,
    extract_metadata,
    format_metadata_for_display,
)

class TestMetadataManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_img = Image.new("RGBA", (128, 128), color=(255, 0, 0, 255))

    def tearDown(self):
        for f in os.listdir(self.temp_dir):
            try:
                os.remove(os.path.join(self.temp_dir, f))
            except Exception:
                pass
        try:
            os.rmdir(self.temp_dir)
        except Exception:
            pass

    def test_build_metadata(self):
        meta = build_metadata(
            category="Adobe Stock Flat Vector",
            prompt="origami fox",
            negative_prompt="blurry",
            seed=42,
            speed_mode="fast",
            selected_styles=["Fooocus V2"],
            colors=["#FF0000", "#00FF00"],
            width=1024,
            height=1024,
            loras=[{"name": "flat_vector", "weight": 0.85}],
        )
        self.assertEqual(meta["category"], "Adobe Stock Flat Vector")
        self.assertEqual(meta["prompt"], "origami fox")
        self.assertEqual(meta["seed"], 42)
        self.assertEqual(meta["speed_mode"], "fast")
        self.assertEqual(meta["version"], "2.0")
        self.assertIn("loras", meta)

    def test_save_and_extract_roundtrip(self):
        meta = build_metadata(
            category="Adobe Stock Silhouette",
            prompt="running horse",
            negative_prompt="color, 3D",
            seed=12345,
            speed_mode="master",
            selected_styles=[],
            colors=[],
            width=1024,
            height=1024,
        )
        filepath = os.path.join(self.temp_dir, "test_asset.png")
        saved_path = save_image_with_metadata(self.test_img, filepath, meta)
        self.assertTrue(os.path.exists(saved_path))

        # Extract from file path
        extracted = extract_metadata(saved_path)
        self.assertIsNotNone(extracted)
        self.assertEqual(extracted["category"], "Adobe Stock Silhouette")
        self.assertEqual(extracted["prompt"], "running horse")
        self.assertEqual(extracted["seed"], 12345)
        self.assertEqual(extracted["speed_mode"], "master")

        # Extract from PIL image opened directly
        with Image.open(saved_path) as loaded_img:
            extracted_img = extract_metadata(loaded_img)
            self.assertIsNotNone(extracted_img)
            self.assertEqual(extracted_img["category"], "Adobe Stock Silhouette")

    def test_extract_from_image_without_metadata(self):
        filepath = os.path.join(self.temp_dir, "plain.png")
        self.test_img.save(filepath)
        extracted = extract_metadata(filepath)
        self.assertIsNone(extracted)

    def test_format_metadata_for_display(self):
        meta = {
            "category": "Die-Cut Sticker",
            "prompt": "cute cat",
            "seed": 999,
            "speed_mode": "fast",
            "selected_styles": ["SAI Line Art"],
            "colors": ["#FFA500"],
            "version": "2.0",
        }
        display_str = format_metadata_for_display(meta)
        self.assertIn("Die-Cut Sticker", display_str)
        self.assertIn("cute cat", display_str)
        self.assertIn("999", display_str)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_metadata_manager.py`
Expected: FAIL (`ModuleNotFoundError: No module named 'modules.metadata_manager'`)

- [ ] **Step 3: Implement `modules/metadata_manager.py`**

```python
"""
Metadata Manager Module — Fooocus Designer 2.0
Embeds structured JSON metadata into PNG tEXt chunks and extracts parameters
for single-click recreation and UI recovery.
"""
import json
import os
from typing import Dict, Any, Optional, Union, List
from PIL import Image
from PIL.PngImagePlugin import PngInfo

METADATA_KEY = "fooocus_designer_metadata"
PARAMETERS_KEY = "parameters"


def build_metadata(
    category: str,
    prompt: str,
    negative_prompt: str = "",
    seed: int = -1,
    speed_mode: str = "fast",
    selected_styles: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    width: int = 1024,
    height: int = 1024,
    loras: Optional[List[Dict[str, Any]]] = None,
    version: str = "2.0",
) -> Dict[str, Any]:
    """Construct a clean, serializable metadata dictionary."""
    return {
        "version": version,
        "category": category or "",
        "prompt": prompt or "",
        "negative_prompt": negative_prompt or "",
        "seed": seed,
        "speed_mode": speed_mode or "fast",
        "selected_styles": list(selected_styles or []),
        "colors": list(colors or []),
        "dimensions": [width, height],
        "loras": list(loras or []),
        "model": "RunDiffusion/Juggernaut-XL-v9",
    }


def save_image_with_metadata(
    image: Image.Image,
    filepath: str,
    metadata: Dict[str, Any],
    fmt: str = "png"
) -> str:
    """
    Save a PIL Image with embedded PNG metadata.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    if fmt.lower() == "png":
        png_info = PngInfo()
        json_str = json.dumps(metadata, ensure_ascii=False)
        png_info.add_text(METADATA_KEY, json_str)
        # Add standard parameters text for interoperability
        png_info.add_text(
            PARAMETERS_KEY,
            f"{metadata.get('prompt', '')}\n"
            f"Negative prompt: {metadata.get('negative_prompt', '')}\n"
            f"Steps: {6 if metadata.get('speed_mode') == 'fast' else 28}, "
            f"Seed: {metadata.get('seed', -1)}, "
            f"Model: {metadata.get('model', 'Juggernaut-XL')}"
        )
        image.save(filepath, format="PNG", pnginfo=png_info)
    else:
        image.save(filepath, format=fmt.upper(), quality=95)

    return filepath


def extract_metadata(image_or_path: Union[Image.Image, str]) -> Optional[Dict[str, Any]]:
    """
    Extract embedded Fooocus Designer metadata from a PIL Image or image file path.
    Returns parsed dictionary or None if absent.
    """
    try:
        if isinstance(image_or_path, str):
            if not os.path.exists(image_or_path):
                return None
            with Image.open(image_or_path) as img:
                info = img.info
                return _parse_info_dict(info)
        elif isinstance(image_or_path, Image.Image):
            return _parse_info_dict(image_or_path.info)
        return None
    except Exception:
        return None


def _parse_info_dict(info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Helper to extract and parse metadata from an image info dictionary."""
    if not info:
        return None
    
    # 1. Primary Fooocus Designer metadata JSON
    if METADATA_KEY in info:
        try:
            val = info[METADATA_KEY]
            if isinstance(val, bytes):
                val = val.decode("utf-8", errors="ignore")
            return json.loads(val)
        except Exception:
            pass

    return None


def format_metadata_for_display(metadata: Dict[str, Any]) -> str:
    """Format metadata into a human-readable summary string for the UI."""
    if not metadata:
        return "No metadata found."
    lines = [
        f"🎯 Category: {metadata.get('category', 'None')}",
        f"⚡ Mode: {metadata.get('speed_mode', 'fast').upper()}",
        f"✨ Prompt: {metadata.get('prompt', '')}",
        f"🎲 Seed: {metadata.get('seed', -1)}",
    ]
    styles = metadata.get("selected_styles", [])
    if styles:
        lines.append(f"🎨 Styles: {', '.join(styles)}")
    colors = metadata.get("colors", [])
    if colors:
        lines.append(f"🎨 Colors: {', '.join(colors)}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_metadata_manager.py`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add modules/metadata_manager.py tests/test_metadata_manager.py
git commit -m "feat: implement PNG metadata embedding and extraction for asset reproduction"
```

---

### Task 2: Curated Color Palette Presets (`modules/palette_control.py`)

**Files:**
- Modify: `modules/palette_control.py`
- Modify: `tests/test_palette.py`

**Interfaces:**
- Produces:
  - `PALETTE_PRESETS: Dict[str, List[str]]`
  - `get_palette_presets() -> List[str]`
  - `get_preset_colors(preset_name: str) -> List[str]`

- [ ] **Step 1: Write tests for palette presets in `tests/test_palette.py`**

Add tests:
```python
    def test_get_palette_presets(self):
        from modules.palette_control import get_palette_presets, get_preset_colors
        presets = get_palette_presets()
        self.assertIn("Custom / None", presets)
        self.assertIn("Pastel Dreams", presets)
        self.assertIn("Cyberpunk Neon", presets)
        self.assertIn("Earthy Boho", presets)
        self.assertIn("Corporate Tech", presets)
        self.assertIn("Retro Sunset", presets)
        self.assertIn("Luxury Gold", presets)
        self.assertIn("Nordic Minimalist", presets)

        # Verify each preset returns exactly 5 hex colors
        for name in presets:
            colors = get_preset_colors(name)
            self.assertEqual(len(colors), 5)
            for c in colors:
                self.assertTrue(c.startswith("#"), f"Color {c} in preset {name} must start with #")
                self.assertEqual(len(c), 7)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_palette.py`
Expected: FAIL (`ImportError: cannot import name 'get_palette_presets'`)

- [ ] **Step 3: Update `modules/palette_control.py`**

Add curated palette dictionary and helper functions:
```python
PALETTE_PRESETS = {
    "Custom / None": ["#000000", "#000000", "#000000", "#000000", "#000000"],
    "Pastel Dreams": ["#FFB3BA", "#BAFFC9", "#BAE1FF", "#FFFFBA", "#E8BAFF"],
    "Cyberpunk Neon": ["#00F0FF", "#FF003C", "#FCEE09", "#05D9E8", "#D1F7FF"],
    "Earthy Boho": ["#C29B7F", "#8C6239", "#556B2F", "#D4AF37", "#EEDC82"],
    "Corporate Tech": ["#0052CC", "#00B8D9", "#36B37E", "#172B4D", "#F4F5F7"],
    "Retro Sunset": ["#FD5E53", "#FC9C54", "#FFE373", "#432C7A", "#FF7A5A"],
    "Luxury Gold": ["#D4AF37", "#FFD700", "#1A1A1A", "#2C2C2C", "#F5F5DC"],
    "Nordic Minimalist": ["#E5E5E5", "#333333", "#4A6572", "#F9AA33", "#344955"],
}


def get_palette_presets() -> list:
    """Return list of available palette preset names."""
    return list(PALETTE_PRESETS.keys())


def get_preset_colors(preset_name: str) -> list:
    """Return the 5 hex colors for the given preset name, or default black."""
    return list(PALETTE_PRESETS.get(preset_name, PALETTE_PRESETS["Custom / None"]))
```

- [ ] **Step 4: Run tests to verify all palette tests pass**

Run: `python -m unittest tests/test_palette.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add modules/palette_control.py tests/test_palette.py
git commit -m "feat: add curated commercial color palette presets to palette_control"
```

---

### Task 3: Wire Metadata Embedding, Drop-to-Load, & Palette Presets into UI (`ui/design_main.py`)

**Files:**
- Modify: `ui/design_main.py`
- Create: `tests/test_ui_metadata_palette_integration.py`

**Interfaces:**
- Consumes:
  - `modules.metadata_manager.build_metadata`
  - `modules.metadata_manager.save_image_with_metadata`
  - `modules.metadata_manager.extract_metadata`
  - `modules.palette_control.get_palette_presets`
  - `modules.palette_control.get_preset_colors`

- [ ] **Step 1: Write integration tests in `tests/test_ui_metadata_palette_integration.py`**

```python
import os
import unittest
from PIL import Image
from modules.metadata_manager import build_metadata, save_image_with_metadata, extract_metadata
from ui.design_main import _save_image, _on_palette_preset_change, _on_image_drop_inspect

class TestUIMetadataPaletteIntegration(unittest.TestCase):
    def test_save_image_embeds_metadata(self):
        img = Image.new("RGBA", (64, 64), (0, 128, 255, 255))
        meta = build_metadata(
            category="Adobe Stock Flat Vector",
            prompt="flat vector logo",
            seed=777,
            speed_mode="fast",
        )
        saved_path = _save_image(img, "outputs", metadata=meta)
        self.assertTrue(os.path.exists(saved_path))
        extracted = extract_metadata(saved_path)
        self.assertIsNotNone(extracted)
        self.assertEqual(extracted["prompt"], "flat vector logo")
        self.assertEqual(extracted["seed"], 777)
        # Cleanup
        try:
            os.remove(saved_path)
        except Exception:
            pass

    def test_on_palette_preset_change(self):
        updates = _on_palette_preset_change("Cyberpunk Neon")
        self.assertEqual(len(updates), 5)
        # Each update should have the hex color value
        self.assertEqual(updates[0]["value"], "#00F0FF")
        self.assertEqual(updates[1]["value"], "#FF003C")

    def test_on_image_drop_inspect(self):
        # Create a test image with metadata
        img = Image.new("RGB", (64, 64), (200, 200, 200))
        meta = build_metadata(
            category="Die-Cut Sticker",
            prompt="holographic sticker of astronaut",
            negative_prompt="blurry",
            seed=42,
            speed_mode="master",
            selected_styles=["Fooocus V2"],
            colors=["#FFB3BA", "#BAFFC9", "#000000", "#000000", "#000000"],
        )
        temp_path = os.path.abspath("outputs/test_inspect.png")
        save_image_with_metadata(img, temp_path, meta)

        try:
            status, cat, pr, neg, spd, stl, sd, c1, c2, c3, c4, c5 = _on_image_drop_inspect(temp_path)
            self.assertIn("Loaded settings", status)
            self.assertEqual(cat["value"], "Die-Cut Sticker")
            self.assertEqual(pr["value"], "holographic sticker of astronaut")
            self.assertEqual(neg["value"], "blurry")
            self.assertEqual(spd["value"], "🎯 Master (~15s)")
            self.assertEqual(stl["value"], ["Fooocus V2"])
            self.assertEqual(sd["value"], "42")
            self.assertEqual(c1["value"], "#FFB3BA")
            self.assertEqual(c2["value"], "#BAFFC9")
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_ui_metadata_palette_integration.py`
Expected: FAIL (`ImportError` on helper functions or missing arguments)

- [ ] **Step 3: Update `ui/design_main.py`**
- Import `build_metadata`, `save_image_with_metadata`, `extract_metadata` from `modules.metadata_manager`.
- Import `get_palette_presets`, `get_preset_colors` from `modules.palette_control`.
- Update `_save_image(image, output_dir, fmt='png', metadata=None)`.
- Implement `_on_palette_preset_change(preset_name)`.
- Implement `_on_image_drop_inspect(image_file)`.
- In `build_tab`:
  - Add Palette Preset Dropdown (`palette_presets_dropdown`) inside Color Palette accordion.
  - Wire `palette_presets_dropdown.change` to update `color1` through `color5`.
  - Add Metadata Dropper / Inspector accordion with image upload component `inspect_image`.
  - Wire `inspect_image.change` to call `_on_image_drop_inspect` and update category, prompt, negative, speed, styles, seed, and color pickers.
  - In `_generate`, construct `metadata = build_metadata(...)` and pass to `_save_image`.

- [ ] **Step 4: Run integration tests to verify all pass**

Run: `python -m unittest tests/test_ui_metadata_palette_integration.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ui/design_main.py tests/test_ui_metadata_palette_integration.py
git commit -m "feat: embed PNG metadata on save, add palette preset picker, and enable drop-to-load settings"
```

---

### Task 4: Harmonize Mockup, Edit, & Variations Tabs with Engine Modes and Metadata

**Files:**
- Modify: `ui/design_mockup.py`
- Modify: `ui/design_edit.py`
- Modify: `ui/design_variations.py`
- Modify: `tests/test_mockup.py`
- Modify: `tests/test_variation.py`

**Interfaces:**
- Updates UI dropdown choices in `ui/design_mockup.py`:
  - `choices=["⚡ Fast (~3s)", "🎯 Master (~15s)"]`, value `"⚡ Fast (~3s)"`
- Embeds metadata in `ui/design_mockup.py`, `ui/design_edit.py`, and `ui/design_variations.py` outputs.

- [ ] **Step 1: Write/update tests in `tests/test_mockup.py` and `tests/test_variation.py`**

Update tests to verify that `generate_mockup` and `generate_variations` work with the new speed labels and output images containing metadata.

- [ ] **Step 2: Run tests to verify failure/status**

Run: `python -m unittest tests/test_mockup.py tests/test_variation.py`

- [ ] **Step 3: Update `ui/design_mockup.py`, `ui/design_edit.py`, `ui/design_variations.py`**
- In `ui/design_mockup.py`: Replace `mockup_model_choice` choices with `["⚡ Fast (~3s)", "🎯 Master (~15s)"]`. Use `save_image_with_metadata`.
- In `ui/design_edit.py`: Use `save_image_with_metadata` with edit metadata.
- In `ui/design_variations.py`: Use `save_image_with_metadata` with variation metadata.

- [ ] **Step 4: Run tests to verify all pass**

Run: `python -m unittest tests/test_mockup.py tests/test_variation.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ui/design_mockup.py ui/design_edit.py ui/design_variations.py tests/test_mockup.py tests/test_variation.py
git commit -m "feat: harmonize mockup and edit tabs with unified engine modes and metadata embedding"
```

---

### Task 5: End-to-End Test Suite Verification & Documentation Update

**Files:**
- Modify: `tests/test_end_to_end.py`
- Modify: `readme.md`

- [ ] **Step 1: Add E2E tests for metadata and palette in `tests/test_end_to_end.py`**
- Test end-to-end generation with metadata embedding and full extraction.
- Test end-to-end setting restoration via drop-inspect handler.

- [ ] **Step 2: Update `readme.md`**
- Document PNG metadata embedding & recovery (`fooocus_designer_metadata`).
- Document Curated Color Palette Presets (Pastel, Cyberpunk, Boho, Corporate, etc.).
- Update tab descriptions and architecture diagram.

- [ ] **Step 3: Run the full test suite**

Run: `python -m unittest discover -s tests -p "test_*.py"`
Expected: ALL tests pass (55+ tests).

- [ ] **Step 4: Commit**

```bash
git add tests/test_end_to_end.py readme.md
git commit -m "docs: document Phase 3 metadata and palette features, expand E2E test verification"
```
