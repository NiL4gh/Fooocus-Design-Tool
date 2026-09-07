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

if __name__ == "__main__":
    unittest.main()
