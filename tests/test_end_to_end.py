import os
import unittest
os.environ["MOCK_IMAGE_GEN"] = "1"

class TestEndToEndSystem(unittest.TestCase):
    def test_legacy_zimage_pipeline_removed(self):
        """Ensure zimage_pipeline is completely eliminated."""
        import sys
        self.assertFalse(os.path.exists("modules/zimage_pipeline.py"))

    def test_all_modules_and_ui_importable(self):
        """Verify all app modules and UI tabs import cleanly with SDXL pipeline."""
        from modules import sdxl_pipeline, lora_router, design_categories, variation, logo_mockup
        from ui import design_main, design_edit, design_variations, design_mockup, theme
        import webui
        self.assertTrue(hasattr(sdxl_pipeline, "generate"))
        self.assertTrue(hasattr(lora_router, "apply_category_lora"))

    def test_end_to_end_category_generation(self):
        """Test full generation through multiple design categories."""
        from modules.sdxl_pipeline import generate
        from modules.design_categories import get_category
        from PIL import Image

        for cat_name in ["Adobe Stock Silhouette", "Adobe Stock Flat Vector", "Logo", "Poster"]:
            cat_cfg = get_category(cat_name)
            img, seed = generate(
                prompt="test subject",
                speed_mode="fast",
                category_cfg=cat_cfg
            )
            self.assertIsInstance(img, Image.Image)
            self.assertEqual(img.size, (1024, 1024))

    def test_end_to_end_style_engine_integration(self):
        """Test full generation flow integrated with Fooocus Style Engine."""
        from modules.auto_prompt_enhancer import enhance_prompt, build_negative_prompt
        from modules.sdxl_pipeline import generate
        from modules.design_categories import get_category
        from PIL import Image

        for style_name in ["Fooocus V2", "SAI Line Art"]:
            cat_cfg = get_category("Artwork")
            styled_p = enhance_prompt(
                "cyberpunk skyline",
                "Artwork",
                use_enhancement=True,
                selected_styles=[style_name],
            )
            styled_n = build_negative_prompt(
                "",
                "Artwork",
                use_master_negative=True,
                selected_styles=[style_name],
            )

            if style_name == "Fooocus V2":
                self.assertIn("cyberpunk skyline", styled_p)
            elif style_name == "SAI Line Art":
                self.assertIn("line art", styled_p.lower())

            img, seed = generate(
                prompt=styled_p,
                negative_prompt=styled_n,
                speed_mode="fast",
                category_cfg=cat_cfg,
            )
            self.assertIsInstance(img, Image.Image)
            self.assertEqual(img.size, (1024, 1024))

            # Also verify through the full UI generation generator
            from ui.design_main import _generate
            gen = _generate(
                category="Poster",
                prompt="cyberpunk skyline",
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
                selected_styles=[style_name],
            )
            steps = list(gen)
            status, out_img, paths = steps[-1]
            self.assertIn("Done", status)
            self.assertIsInstance(out_img, Image.Image)

    def test_style_engine_get_style(self):
        """Verify get_style returns configuration for valid styles and None for invalid styles."""
        from modules.style_engine import get_style

        # Valid styles
        fooocus_v2 = get_style("Fooocus V2")
        self.assertIsNotNone(fooocus_v2)
        self.assertEqual(fooocus_v2["name"], "Fooocus V2")
        self.assertIn("{prompt}", fooocus_v2["prompt"])
        self.assertIn("negative_prompt", fooocus_v2)

        sai_line = get_style("SAI Line Art")
        self.assertIsNotNone(sai_line)
        self.assertEqual(sai_line["name"], "SAI Line Art")

        # Invalid style
        invalid_style = get_style("Nonexistent Style XYZ")
        self.assertIsNone(invalid_style)


if __name__ == "__main__":
    unittest.main()

