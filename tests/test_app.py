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
from modules.zimage_pipeline import generate, generate_variations, load_pipeline
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

    def test_zimage_pipeline_mock(self):
        """Test mock diffusion generation runs and creates valid PIL images."""
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

    def test_model_switching_mock(self):
        """Test model switching in mock mode tracks the current model correctly."""
        from modules import zimage_pipeline
        zimage_pipeline.unload_pipeline()
        
        # Load first model
        pipe1 = load_pipeline(model_name="FLUX.1-schnell")
        self.assertEqual(pipe1, "mock_pipeline")
        self.assertEqual(zimage_pipeline._current_model_name, "FLUX.1-schnell")
        
        # Generate with first model
        img1, seed1 = generate("simple prompt", seed=42, model_name="FLUX.1-schnell")
        self.assertIsInstance(img1, Image.Image)
        self.assertEqual(zimage_pipeline._current_model_name, "FLUX.1-schnell")
        
        # Switch to second model
        pipe2 = load_pipeline(model_name="Z-Image-Turbo")
        self.assertEqual(pipe2, "mock_pipeline")
        self.assertEqual(zimage_pipeline._current_model_name, "Z-Image-Turbo")
        
        # Generate with second model
        img2, seed2 = generate("simple prompt", seed=42, model_name="Z-Image-Turbo")
        self.assertIsInstance(img2, Image.Image)
        self.assertEqual(zimage_pipeline._current_model_name, "Z-Image-Turbo")
        
        # Unload pipeline
        zimage_pipeline.unload_pipeline()
        self.assertIsNone(zimage_pipeline._pipeline)
        self.assertIsNone(zimage_pipeline._current_model_name)


if __name__ == "__main__":
    unittest.main()
