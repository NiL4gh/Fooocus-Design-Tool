# Fooocus Designer 2.0: Style Engine Port & Dual-Mode Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the official Fooocus style engine (`sdxl_styles/*.json`), integrate dual-mode prompt expansion (strict microstock tag formulation for vectors/silhouettes vs creative Fooocus styles for artwork/posters), and wire style stacking into the turnkey UI.

**Architecture:** A centralized `modules/style_engine.py` reads modular JSON style files from `config/sdxl_styles/` and performs `{prompt}` placeholder interpolation and negative prompt stacking. `modules/auto_prompt_enhancer.py` is refactored into a dual-mode engine that enforces pristine microstock tag rules for commercial vector/silhouette presets while seamlessly layering Fooocus styles when requested. `ui/design_main.py` is updated with an interactive style selector featuring smart category-aware defaults.

**Tech Stack:** Python 3.10+, JSON, Gradio 4.x, `unittest`.

## Global Constraints
- Target environment: Google Colab and local 12GB+ GPUs.
- In test environments (`MOCK_IMAGE_GEN=1`), all operations must run deterministically in memory without downloading checkpoints or requiring a GPU.
- Category integrity: Vector and Silhouette presets must default to empty Fooocus styles to prevent photographic contamination of clean 2D assets, while preserving user ability to add styles manually.
- Windows PowerShell compatibility: all commands and paths must function properly in `pwsh`.

---

### Task 1: Create Fooocus Style JSON Library (`config/sdxl_styles/*.json`)

**Files:**
- Create: `config/sdxl_styles/sdxl_styles_fooocus.json`
- Create: `config/sdxl_styles/sdxl_styles_sai.json`
- Create: `config/sdxl_styles/sdxl_styles_mre.json`
- Test: `tests/test_style_json_validity.py`

**Interfaces:**
- Produces: Valid JSON files containing lists of objects:
  `{"name": str, "prompt": str, "negative_prompt": str}`
  where `"prompt"` contains the `{prompt}` placeholder token.

- [ ] **Step 1: Write unit test validating style JSON schema and placeholders**

Create `tests/test_style_json_validity.py`:
```python
import os
import json
import glob
import unittest

class TestStyleJSONValidity(unittest.TestCase):
    def test_all_style_json_files_are_valid(self):
        style_dir = os.path.join("config", "sdxl_styles")
        json_files = glob.glob(os.path.join(style_dir, "*.json"))
        self.assertGreater(len(json_files), 0, "No style JSON files found in config/sdxl_styles")

        style_names = set()
        for fpath in json_files:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIsInstance(data, list, f"{fpath} must contain a list of style objects")
            for item in data:
                self.assertIn("name", item)
                self.assertIn("prompt", item)
                self.assertIn("negative_prompt", item)
                self.assertIn("{prompt}", item["prompt"], f"Style '{item['name']}' in {fpath} must contain '{{prompt}}'")
                self.assertNotIn(item["name"], style_names, f"Duplicate style name '{item['name']}' across files")
                style_names.add(item["name"])

        # Check key Fooocus styles exist
        self.assertIn("Fooocus V2", style_names)
        self.assertIn("Fooocus Masterpiece", style_names)
        self.assertIn("SAI Line Art", style_names)
        self.assertIn("MRE Flat 2D Art", style_names)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_style_json_validity.py`  
Expected: FAIL with `AssertionError: 0 not greater than 0 : No style JSON files found in config/sdxl_styles`

- [ ] **Step 3: Create style JSON files in `config/sdxl_styles/`**

Create `config/sdxl_styles/sdxl_styles_fooocus.json`:
```json
[
  {
    "name": "Fooocus V2",
    "prompt": "{prompt}, highly detailed, sharp focus, professional composition",
    "negative_prompt": "ugly, deformed, disfigured, poor details, bad anatomy"
  },
  {
    "name": "Fooocus Masterpiece",
    "prompt": "masterpiece, best quality, {prompt}, intricate details, elegant",
    "negative_prompt": "worst quality, low quality, normal quality, lowres, text, watermark"
  },
  {
    "name": "Fooocus Photograph",
    "prompt": "professional photography, natural lighting, {prompt}, 35mm lens, sharp, highly detailed, raw photo",
    "negative_prompt": "illustration, 3d render, painting, drawing, cartoon, anime, artificial"
  },
  {
    "name": "Fooocus Cinematic",
    "prompt": "cinematic still, {prompt}, dramatic lighting, 8k resolution, photorealistic, block buster movie atmosphere",
    "negative_prompt": "amateur, low quality, oversaturated, blurry, cartoon"
  }
]
```

Create `config/sdxl_styles/sdxl_styles_sai.json`:
```json
[
  {
    "name": "SAI Line Art",
    "prompt": "line art drawing, {prompt}, professional lineart, clean bold lines, monochrome, graphic design, minimalist",
    "negative_prompt": "shading, shadows, colored, 3d, realistic photo, gradients"
  },
  {
    "name": "SAI Origami",
    "prompt": "origami paper art, {prompt}, folded paper craft, paper texture, clean sharp folds, studio lighting",
    "negative_prompt": "smooth surfaces, realistic, photographic, metal, plastic"
  },
  {
    "name": "SAI Digital Art",
    "prompt": "digital art illustration, {prompt}, trending on artstation, vibrant colors, detailed brush strokes, concept art",
    "negative_prompt": "photograph, grainy, low resolution, messy"
  },
  {
    "name": "SAI Comic Book",
    "prompt": "comic book illustration, {prompt}, bold ink lines, halftone dots, vibrant flat colors, graphic novel style",
    "negative_prompt": "photograph, 3d render, smooth gradient, realistic"
  },
  {
    "name": "SAI Anime",
    "prompt": "anime aesthetic, {prompt}, key visual, studio anime artwork, clean linework, cel shaded",
    "negative_prompt": "photo, photorealistic, 3d render, ugly face"
  },
  {
    "name": "SAI Isometric",
    "prompt": "isometric 3d projection, {prompt}, clean orthographic view, miniature diorama, cute, smooth lighting",
    "negative_prompt": "perspective distortion, 2d flat, deformed"
  }
]
```

Create `config/sdxl_styles/sdxl_styles_mre.json`:
```json
[
  {
    "name": "MRE Flat 2D Art",
    "prompt": "flat 2d art illustration, {prompt}, solid colors, clean outlines, minimalist vector aesthetic, graphic asset",
    "negative_prompt": "3d render, photorealistic, realistic, shadows, gradients, specular"
  },
  {
    "name": "MRE Vector Art",
    "prompt": "scalable vector graphic art, {prompt}, clean paths, high contrast, smooth contours, commercial design element",
    "negative_prompt": "photo, texture, noise, blurry, complex shading, 3d"
  },
  {
    "name": "MRE Watercolor",
    "prompt": "expressive watercolor painting, {prompt}, paint bleed, wet paper texture, artistic pigment stains, soft edges",
    "negative_prompt": "photograph, 3d, hard plastic, vector"
  }
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_style_json_validity.py`  
Expected: PASS (1 test in <0.05s, OK)

- [ ] **Step 5: Commit Task 1**

```bash
git add config/sdxl_styles/ tests/test_style_json_validity.py
git commit -m "feat: add Fooocus, SAI, and MRE style JSON libraries"
```

---

### Task 2: Implement Fooocus Style Engine Module (`modules/style_engine.py`)

**Files:**
- Create: `modules/style_engine.py`
- Test: `tests/test_style_engine.py`

**Interfaces:**
- Produces:
  - `load_styles() -> Dict[str, Dict[str, str]]`: Scans `config/sdxl_styles/*.json` and returns mapping of style names to `{name, prompt, negative_prompt}`.
  - `get_available_styles() -> List[str]`: Returns alphabetically sorted list of all available style names.
  - `apply_styles(prompt: str, negative_prompt: str, style_names: List[str]) -> Tuple[str, str]`: Injects user prompt into `{prompt}` for each selected style sequentially and merges negative prompts cleanly.

- [ ] **Step 1: Write unit tests for `modules/style_engine.py`**

Create `tests/test_style_engine.py`:
```python
import unittest
from modules.style_engine import (
    load_styles,
    get_available_styles,
    apply_styles,
    get_style,
)

class TestStyleEngine(unittest.TestCase):
    def test_load_styles_returns_known_styles(self):
        styles = load_styles()
        self.assertIn("Fooocus V2", styles)
        self.assertIn("SAI Line Art", styles)
        self.assertIn("MRE Flat 2D Art", styles)

    def test_get_available_styles(self):
        names = get_available_styles()
        self.assertIsInstance(names, list)
        self.assertIn("Fooocus Masterpiece", names)

    def test_apply_single_style(self):
        user_prompt = "coffee cup logo"
        user_neg = "ugly"
        p, n = apply_styles(user_prompt, user_neg, ["Fooocus V2"])
        self.assertIn("coffee cup logo", p)
        self.assertIn("highly detailed", p)
        self.assertIn("ugly", n)
        self.assertIn("poor details", n)

    def test_apply_multiple_styles(self):
        user_prompt = "mountain landscape"
        user_neg = ""
        p, n = apply_styles(user_prompt, user_neg, ["Fooocus Masterpiece", "SAI Line Art"])
        self.assertIn("mountain landscape", p)
        self.assertIn("masterpiece", p)
        self.assertIn("line art drawing", p)
        self.assertIn("worst quality", n)
        self.assertIn("shading", n)

    def test_apply_empty_or_none_styles(self):
        p, n = apply_styles("raw prompt", "raw neg", [])
        self.assertEqual(p, "raw prompt")
        self.assertEqual(n, "raw neg")

        p, n = apply_styles("raw prompt", "raw neg", None)
        self.assertEqual(p, "raw prompt")
        self.assertEqual(n, "raw neg")

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_style_engine.py`  
Expected: FAIL with `ModuleNotFoundError: No module named 'modules.style_engine'`

- [ ] **Step 3: Implement `modules/style_engine.py`**

Create `modules/style_engine.py`:
```python
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
    negative_parts = [negative_prompt.strip()] if negative_prompt.strip() else []

    for name in style_names:
        if name in styles:
            style_cfg = styles[name]
            p_template = style_cfg.get("prompt", "{prompt}")
            n_addition = style_cfg.get("negative_prompt", "").strip()

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_style_engine.py`  
Expected: PASS (Ran 5 tests in <0.05s, OK)

- [ ] **Step 5: Commit Task 2**

```bash
git add modules/style_engine.py tests/test_style_engine.py
git commit -m "feat: implement Fooocus style engine with multi-style interpolation"
```

---

### Task 3: Dual-Mode Auto Prompt Enhancer (`modules/auto_prompt_enhancer.py`)

**Files:**
- Modify: `modules/auto_prompt_enhancer.py`
- Test: `tests/test_auto_prompt_enhancer.py`

**Interfaces:**
- Consumes: `modules.design_categories`, `modules.style_engine`.
- Produces:
  - `enhance_prompt(prompt: str, category_name: str, use_enhancement: bool = True, selected_styles: Optional[List[str]] = None) -> str`
  - `build_negative_prompt(user_negative: str, category_name: str, use_master_negative: bool = True, selected_styles: Optional[List[str]] = None) -> str`
  - `is_vector_silhouette_category(category_name: str) -> bool`

- [ ] **Step 1: Write tests for dual-mode prompt enhancer**

Create `tests/test_auto_prompt_enhancer.py`:
```python
import unittest
from modules.auto_prompt_enhancer import (
    enhance_prompt,
    build_negative_prompt,
    is_vector_silhouette_category,
)

class TestAutoPromptEnhancer(unittest.TestCase):
    def test_vector_silhouette_category_detection(self):
        self.assertTrue(is_vector_silhouette_category("Adobe Stock Silhouette"))
        self.assertTrue(is_vector_silhouette_category("Adobe Stock Flat Vector"))
        self.assertTrue(is_vector_silhouette_category("Vector Silhouette"))
        self.assertFalse(is_vector_silhouette_category("Artwork"))
        self.assertFalse(is_vector_silhouette_category("Poster"))

    def test_enhance_prompt_with_styles(self):
        prompt = "majestic eagle"
        enhanced = enhance_prompt(prompt, "Artwork", use_enhancement=True, selected_styles=["Fooocus V2"])
        self.assertIn("majestic eagle", enhanced)
        self.assertIn("highly detailed", enhanced)

    def test_vector_enhancement_priority(self):
        prompt = "running cheetah"
        enhanced = enhance_prompt(prompt, "Adobe Stock Silhouette", use_enhancement=True)
        self.assertIn("solid black silhouette", enhanced)
        self.assertIn("running cheetah", enhanced)

    def test_build_negative_prompt_with_styles(self):
        neg = build_negative_prompt("blurry", "Adobe Stock Flat Vector", use_master_negative=True, selected_styles=["MRE Flat 2D Art"])
        self.assertIn("blurry", neg)
        # Check category master negative
        self.assertIn("photograph", neg)
        # Check style negative
        self.assertIn("3d render", neg)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_auto_prompt_enhancer.py`  
Expected: FAIL with `ImportError: cannot import name 'is_vector_silhouette_category'`

- [ ] **Step 3: Update `modules/auto_prompt_enhancer.py`**

Refactor `modules/auto_prompt_enhancer.py`:
```python
"""
Auto Prompt Enhancer Module — Fooocus Designer 2.0
Automatically enhances user prompts with category-specific microstock keywords
and layers Fooocus style templates ({prompt} interpolation) into generation.
"""

from typing import List, Optional
from modules.design_categories import get_enhancement_template, get_master_negative
from modules.style_engine import apply_styles

VECTOR_SILHOUETTE_CATEGORIES = {
    "adobe stock silhouette",
    "adobe stock flat vector",
    "adobe stock sticker/clipart",
    "vector silhouette",
    "logo",
}


def is_vector_silhouette_category(category_name: str) -> bool:
    """Return whether category is a clean vector, silhouette, sticker, or logo."""
    if not category_name:
        return False
    return category_name.strip().lower() in VECTOR_SILHOUETTE_CATEGORIES


def enhance_prompt(
    prompt: str,
    category_name: str,
    use_enhancement: bool = True,
    selected_styles: Optional[List[str]] = None,
) -> str:
    """
    Enhance a user prompt with category-specific style keywords and Fooocus styles.

    Args:
        prompt: Raw user prompt text.
        category_name: The selected design category name.
        use_enhancement: Whether to apply category enhancement.
        selected_styles: Optional list of Fooocus style template names.

    Returns:
        Enhanced prompt string.
    """
    current_prompt = prompt.strip()

    if use_enhancement and category_name:
        template = get_enhancement_template(category_name)
        if template:
            # Prepend category enhancement template to anchor CLIP conditioning
            current_prompt = f"{template}, {current_prompt}" if current_prompt else template

    # Apply selected Fooocus styles if requested
    if selected_styles:
        current_prompt, _ = apply_styles(current_prompt, "", selected_styles)

    return current_prompt


def build_negative_prompt(
    user_negative: str,
    category_name: str,
    use_master_negative: bool = True,
    selected_styles: Optional[List[str]] = None,
) -> str:
    """
    Build the final negative prompt by combining user negative, category master negative,
    and style negative prompts.

    Args:
        user_negative: User's custom negative prompt.
        category_name: The selected design category name.
        use_master_negative: Whether to append category master negative.
        selected_styles: Optional list of Fooocus style template names.

    Returns:
        Combined negative prompt string.
    """
    negative_parts = [user_negative.strip()] if user_negative.strip() else []

    if use_master_negative and category_name:
        master = get_master_negative(category_name)
        if master and master not in negative_parts:
            negative_parts.append(master)

    combined_neg = ", ".join(negative_parts).strip(", ")

    # Layer negative terms from selected styles
    if selected_styles:
        _, combined_neg = apply_styles("", combined_neg, selected_styles)

    return combined_neg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_auto_prompt_enhancer.py`  
Expected: PASS (Ran 4 tests in <0.05s, OK)

- [ ] **Step 5: Commit Task 3**

```bash
git add modules/auto_prompt_enhancer.py tests/test_auto_prompt_enhancer.py
git commit -m "feat: implement dual-mode prompt enhancement and category style layering"
```

---

### Task 4: Connect Style Engine & Category Defaults to Turnkey UI (`ui/design_main.py`)

**Files:**
- Modify: `ui/design_main.py`
- Test: `tests/test_ui_styles_integration.py`

**Interfaces:**
- Adds an interactive **🎨 Fooocus Styles** multi-select dropdown in `ui/design_main.py`.
- Smart category presets:
  - Vector/Silhouette categories default to `[]` (pure vector/LoRA isolation).
  - "Artwork" and "Poster" default to `["Fooocus V2"]`.
- Passes `selected_styles` to `enhance_prompt` and `build_negative_prompt`.

- [ ] **Step 1: Write failing integration test for UI style wiring**

Create `tests/test_ui_styles_integration.py`:
```python
import os
import unittest
os.environ["MOCK_IMAGE_GEN"] = "1"

from ui.design_main import _generate, _on_category_change
from modules.style_engine import get_available_styles

class TestUIStylesIntegration(unittest.TestCase):
    def test_category_change_sets_smart_style_defaults(self):
        # Silhouette should default to empty styles
        res_sil = _on_category_change("Adobe Stock Silhouette")
        # outputs: [remove_bg, aspect_ratio, status, selected_styles]
        self.assertEqual(res_sil[3]["value"], [])

        # Artwork should default to ["Fooocus V2"]
        res_art = _on_category_change("Artwork")
        self.assertEqual(res_art[3]["value"], ["Fooocus V2"])

    def test_generate_with_selected_styles(self):
        # Test generator yielding results with style applied
        gen = _generate(
            category="Artwork",
            prompt="cyberpunk city",
            negative_prompt="blurry",
            color1="#000000", color2="#000000", color3="#000000", color4="#000000", color5="#000000",
            use_master_neg=True,
            use_enhancement=True,
            remove_bg=False,
            vector_mode=False,
            concept_grid=False,
            aspect_ratio="1024×1024 (1:1)",
            seed_val="42",
            speed_mode_label="⚡ Fast (~3s)",
            selected_styles=["Fooocus V2"]
        )
        steps = list(gen)
        self.assertGreater(len(steps), 0)
        final_status, final_img, final_gallery = steps[-1]
        self.assertIn("Done", final_status)
        self.assertIsNotNone(final_img)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_ui_styles_integration.py`  
Expected: FAIL with `IndexError: tuple index out of range` or signature mismatch in `_generate`

- [ ] **Step 3: Update `ui/design_main.py`**

In `ui/design_main.py`:
1. Import `get_available_styles` from `modules.style_engine`.
2. Update `_on_category_change(category)` to return smart style defaults:
   - If `category in ["Artwork", "Poster"]`: default to `["Fooocus V2"]`.
   - Otherwise: default to `[]`.
3. Update `_generate(...)` to accept `selected_styles=None`.
   - Pass `selected_styles` to `enhance_prompt(prompt, category, use_enhancement=True, selected_styles=selected_styles)`.
   - Pass `selected_styles` to `build_negative_prompt(negative_prompt, category, use_master_neg, selected_styles=selected_styles)`.
4. In `build_tab()`:
   - Add styles multi-select dropdown:
     ```python
     available_styles = get_available_styles()
     styles_selector = gr.Dropdown(
         label="🎨 Fooocus Styles",
         choices=available_styles,
         value=[],
         multiselect=True,
         interactive=True,
         elem_id="styles_dropdown",
     )
     ```
   - Wire `category.change` to also update `styles_selector`.
   - Pass `styles_selector` as an input to `generate_btn.click(...)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_ui_styles_integration.py`  
Expected: PASS (Ran 2 tests in <0.1s, OK)

- [ ] **Step 5: Commit Task 4**

```bash
git add ui/design_main.py tests/test_ui_styles_integration.py
git commit -m "feat: wire Fooocus style selector and category-aware defaults into UI"
```

---

### Task 5: Full Test Suite Verification & Documentation Update

**Files:**
- Modify: `tests/test_app.py`
- Modify: `tests/test_end_to_end.py`
- Modify: `readme.md`

- [ ] **Step 1: Update `tests/test_app.py` and `tests/test_end_to_end.py`**

Add tests verifying style engine integration, category smart defaults, and full end-to-end pipeline execution with Fooocus styles enabled.

- [ ] **Step 2: Update `readme.md`**

Add section detailing:
- Fooocus Style Engine support (`sdxl_styles/*.json`).
- Available styles: Fooocus V2, Fooocus Masterpiece, SAI Line Art, MRE Flat 2D Art, etc.
- Smart category isolation (clean vectors/silhouettes default clean, artwork/posters auto-layer Fooocus V2).

- [ ] **Step 3: Run full repository test suite**

Run: `python -m unittest discover -s tests -p "test_*.py"`  
Expected: ALL tests pass cleanly.

- [ ] **Step 4: Commit Task 5**

```bash
git add tests/ readme.md
git commit -m "docs: document Fooocus style engine and expand full test suite verification"
```
