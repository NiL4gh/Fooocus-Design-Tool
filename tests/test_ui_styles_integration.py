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
