"""
Fooocus Design Tool - Comprehensive Test Suite
Tests all components in mock mode to verify app stability, pipeline flow, and compositing correctness.
"""
import os
import unittest
import numpy as np
from PIL import Image

# Enable mock generation environment variable
os.environ["MOCK_IMAGE_GEN"] = "1"

# Imports from app
from modules.config import parse_aspect_ratio, get_aspect_ratio_labels
from modules.design_categories import (
    load_categories,
    get_category_names,
    get_category,
    get_master_negative,
    get_enhancement_template,
)
from modules.auto_prompt_enhancer import enhance_prompt, build_negative_prompt
from modules.palette_control import inject_palette_prompt, apply_palette_post
from modules.sdxl_pipeline import generate, generate_variations, load_pipeline
from modules.starvector_pipeline import image_to_svg, load_model
from modules.background_remover import remove_background
from modules.logo_mockup import generate_mockup, cylinder_warp, apply_shading_blend


class TestFooocusDesignTool(unittest.TestCase):
    def setUp(self):
        # Ensure outputs directory exists
        os.makedirs("outputs", exist_ok=True)

    def test_config_and_categories(self):
        """Test configuration values and design categories loading."""
        categories = load_categories()
        self.assertGreater(len(categories), 0)
        
        names = get_category_names()
        self.assertIn("Logo", names)
        self.assertIn("Poster", names)
        
        # Test aspect ratio parsing
        labels = get_aspect_ratio_labels()
        self.assertGreater(len(labels), 0)
        
        # 1024*1024 parsing
        w, h = parse_aspect_ratio("1024×1024 (1:1)")
        self.assertEqual(w, 1024)
        self.assertEqual(h, 1024)

    def test_prompt_enhancement(self):
        """Test automatic category enhancements and negative prompt construction."""
        original_prompt = "minimalist rocket"
        enhanced = enhance_prompt(original_prompt, "Logo", use_enhancement=True)
        self.assertIn("minimalist rocket", enhanced)
        # Check that it appended category-specific words
        self.assertIn("logo", enhanced.lower())
        
        neg = build_negative_prompt("bad quality", "Logo", use_master_negative=True)
        self.assertIn("bad quality", neg)
        # Check master negative words from Logo category config
        self.assertIn("3d render", neg.lower())

    def test_color_palette(self):
        """Test color palette injection and post-processing."""
        prompt = "modern poster"
        colors = ["#ff5733", "#33ff57", "#3357ff"]
        injected = inject_palette_prompt(prompt, colors)
        self.assertIn("ff5733", injected)
        self.assertIn("33ff57", injected)
        self.assertIn("3357ff", injected)
        
        # Test palette post-processing on image
        img = Image.new("RGB", (100, 100), color="#ffffff")
        processed_img = apply_palette_post(img, colors, strength=0.5)
        self.assertEqual(processed_img.size, (100, 100))

    def test_sdxl_pipeline_mock(self):
        """Test mock SDXL diffusion generation runs and creates valid PIL images."""
        prompt = "A glowing retro flyer with #e11d48 and #2563eb accents"
        img, seed = generate(prompt, width=256, height=256, seed=42)
        
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (256, 256))
        self.assertEqual(seed, 42)
        
        # Verify the pipeline returns mock indicator
        pipe = load_pipeline()
        self.assertEqual(pipe, "mock_pipeline")

    def test_starvector_mock(self):
        """Test mock vectorization creates valid SVGs."""
        img = Image.new("RGB", (100, 100), color="#0f172a")
        svg_code = image_to_svg(img)
        self.assertIn("<svg", svg_code)
        self.assertIn("</svg>", svg_code)
        self.assertIn("MOCK SVG LOGO", svg_code)

    def test_background_remover_mock(self):
        """Test mock background remover makes dark pixels transparent."""
        # Create image with Slate Dark color '#0f172a' (approx 15, 23, 42)
        img = Image.new("RGB", (10, 10), color="#0f172a")
        rgba = remove_background(img)
        self.assertEqual(rgba.mode, "RGBA")
        # Check that it's made transparent (alpha = 0)
        pixel = rgba.getpixel((0, 0))
        self.assertEqual(pixel[3], 0)

    def test_variations_mock(self):
        """Test seed variation generator creates count variations."""
        prompt = "cyberpunk aesthetic"
        results = generate_variations(prompt, width=128, height=128, base_seed=100, count=3)
        self.assertEqual(len(results), 3)
        for img, seed in results:
            self.assertIsInstance(img, Image.Image)
            self.assertEqual(img.size, (128, 128))

    def test_logo_mockup_mock(self):
        """Test mock mockup composer works end to end with cylinder wrap and fabric shading blend."""
        logo = Image.new("RGB", (120, 120), color="#ec4899")
        
        # Test T-Shirt mockup
        mockup_img, status = generate_mockup(logo, "T-Shirt", prompt="studio display", mockup_style="Single Product (Centered)")
        self.assertIsInstance(mockup_img, Image.Image)
        self.assertEqual(mockup_img.size, (1024, 1024))
        self.assertIn("generated successfully", status)
        
        # Test Mug mockup (involves cylinder warp)
        mockup_mug, status_mug = generate_mockup(logo, "Mug", prompt="on desk", mockup_style="Realistic Ambient Scene")
        self.assertIsInstance(mockup_mug, Image.Image)
        self.assertIn("generated successfully", status_mug)

    def test_speed_mode_switching_mock(self):
        """Test speed mode switching in mock mode tracks the current speed mode correctly."""
        from modules import sdxl_pipeline
        sdxl_pipeline.unload_pipeline()
        
        # Load fast mode
        pipe1 = load_pipeline(speed_mode="fast")
        self.assertEqual(pipe1, "mock_pipeline")
        self.assertEqual(sdxl_pipeline.get_current_speed_mode(), "fast")
        
        # Generate with fast mode
        img1, seed1 = generate("simple prompt", seed=42, speed_mode="fast")
        self.assertIsInstance(img1, Image.Image)
        self.assertEqual(sdxl_pipeline.get_current_speed_mode(), "fast")
        
        # Switch to master mode
        pipe2 = load_pipeline(speed_mode="master")
        self.assertEqual(pipe2, "mock_pipeline")
        self.assertEqual(sdxl_pipeline.get_current_speed_mode(), "master")
        
        # Generate with master mode
        img2, seed2 = generate("simple prompt", seed=42, speed_mode="master")
        self.assertIsInstance(img2, Image.Image)
        self.assertEqual(sdxl_pipeline.get_current_speed_mode(), "master")
        
        # Unload pipeline
        sdxl_pipeline.unload_pipeline()
        self.assertIsNone(sdxl_pipeline._pipeline)
        self.assertIsNone(sdxl_pipeline.get_current_speed_mode())

    test_model_switching_mock = test_speed_mode_switching_mock

    def test_clean_vram_sdxl(self):
        """Test webui clean_vram correctly unloads SDXL Juggernaut."""
        from webui import clean_vram
        from modules.sdxl_pipeline import load_pipeline
        load_pipeline()
        result_html = clean_vram()
        self.assertIn("Juggernaut-XL (SDXL)", result_html)

    def test_ui_design_main_generation_flow(self):
        """Test ui.design_main generation flow end-to-end in mock mode."""
        from ui.design_main import _generate, build_tab
        import gradio as gr

        # Verify build_tab constructs without error
        with gr.Blocks():
            comps = build_tab()
            self.assertEqual(len(comps), 5)

        # Verify _generate produces successful output
        gen = _generate(
            category="Logo",
            prompt="modern tech icon",
            negative_prompt="",
            color1="#000000", color2="#000000", color3="#000000", color4="#000000", color5="#000000",
            use_master_neg=True,
            use_enhancement=True,
            remove_bg=False,
            vector_mode=False,
            concept_grid=False,
            aspect_ratio="1024×1024 (1:1)",
            seed_val="42",
            speed_mode_label="⚡ Fast (~3s)"
        )
        steps = list(gen)
        self.assertGreater(len(steps), 0)
        status, img, gallery = steps[-1]
        self.assertIn("Done", status)
        self.assertIsNotNone(img)


if __name__ == "__main__":
    unittest.main()
