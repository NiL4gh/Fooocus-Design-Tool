import os
import unittest
from PIL import Image
import gradio as gr

os.environ["MOCK_IMAGE_GEN"] = "1"

from modules.metadata_manager import extract_metadata
from ui.design_mockup import _gen_mockup, build_tab as build_mockup_tab
from ui.design_edit import _edit_generate, build_tab as build_edit_tab
from ui.design_variations import _gen_variations, build_tab as build_var_tab

class TestTabsHarmonization(unittest.TestCase):
    def setUp(self):
        os.makedirs("outputs", exist_ok=True)
        self.test_logo = Image.new("RGBA", (100, 100), (255, 0, 128, 255))

    def test_mockup_generation_embeds_metadata(self):
        gen = _gen_mockup(
            logo_image=self.test_logo,
            product_type="T-Shirt",
            scene_prompt="studio clean",
            mockup_style="Single Product (Centered)",
            model_choice="⚡ Fast (~3s)",
        )
        steps = list(gen)
        status, img = steps[-1]
        self.assertIn("Saved:", status)
        self.assertIsNotNone(img)

        # Find the latest saved mockup
        saved_fn = status.split("Saved:")[-1].strip()
        saved_path = os.path.join("outputs", saved_fn)
        self.assertTrue(os.path.exists(saved_path))
        meta = extract_metadata(saved_path)
        self.assertIsNotNone(meta)
        self.assertEqual(meta["speed_mode"], "fast")
        self.assertIn("Mockup: T-Shirt", meta["category"])

        try:
            os.remove(saved_path)
        except Exception:
            pass

    def test_variations_generation_embeds_metadata(self):
        gen = _gen_variations(
            prompt="futuristic car",
            negative="blurry",
            var_count=2,
            var_strength=50,
            seed_val="100",
            model_choice="🎯 Master (~15s)",
        )
        steps = list(gen)
        status, paths = steps[-1]
        self.assertEqual(len(paths), 2)
        for p in paths:
            self.assertTrue(os.path.exists(p))
            meta = extract_metadata(p)
            self.assertIsNotNone(meta)
            self.assertEqual(meta["speed_mode"], "master")
            self.assertEqual(meta["category"], "Variation")
            try:
                os.remove(p)
            except Exception:
                pass

    def test_edit_generation_embeds_metadata(self):
        gen = _edit_generate(
            edit_image=self.test_logo,
            edit_prompt="change color to blue",
            edit_negative="green",
            edit_strength=0.6,
        )
        steps = list(gen)
        status, img, paths = steps[-1]
        self.assertEqual(len(paths), 1)
        p = paths[0]
        self.assertTrue(os.path.exists(p))
        meta = extract_metadata(p)
        self.assertIsNotNone(meta)
        self.assertEqual(meta["category"], "Edit Design/Photo")
        self.assertEqual(meta["prompt"], "change color to blue")
        try:
            os.remove(p)
        except Exception:
            pass

    def test_tab_builds_without_error(self):
        with gr.Blocks():
            build_mockup_tab()
            build_edit_tab()
            build_var_tab()

if __name__ == "__main__":
    unittest.main()
