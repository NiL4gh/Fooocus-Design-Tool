import os
import unittest

os.environ["MOCK_IMAGE_GEN"] = "1"

from modules.sdxl_pipeline import (
    get_base_model_choices,
    resolve_base_model_id,
    load_pipeline,
    generate,
    unload_pipeline,
    DEFAULT_SDXL_MODEL,
)
from ui.design_main import _get_lora_status


class TestModelSelection(unittest.TestCase):
    def tearDown(self):
        unload_pipeline()

    def test_base_model_choices_contain_defaults(self):
        choices = get_base_model_choices()
        self.assertGreater(len(choices), 0)
        self.assertTrue(any("Juggernaut" in c for c in choices))
        self.assertTrue(any("Animagine" in c for c in choices))

    def test_resolve_base_model_id(self):
        default_resolved = resolve_base_model_id(None)
        self.assertEqual(default_resolved, DEFAULT_SDXL_MODEL)

        custom_label = get_base_model_choices()[0]
        self.assertEqual(resolve_base_model_id(custom_label), "RunDiffusion/Juggernaut-XL-v9")

        raw_id = "some-org/custom-sdxl"
        self.assertEqual(resolve_base_model_id(raw_id), raw_id)

    def test_lora_status_formatting(self):
        status_sil = _get_lora_status("Adobe Stock Silhouette")
        self.assertIn("Active Baked LoRA", status_sil)
        self.assertIn("silhouette", status_sil.lower())

        status_none = _get_lora_status("Artwork")
        self.assertIn("None", status_none)

    def test_mock_generation_with_custom_model(self):
        img, seed = generate(
            prompt="vector badge",
            base_model="Animagine XL 3.1 (Anime & Stylized Art)",
            speed_mode="fast",
        )
        self.assertIsNotNone(img)
    def test_batch_generation_ui(self):
        from ui.design_main import _generate
        gen = _generate(
            category="Logo",
            prompt="minimal coffee logo",
            negative_prompt="",
            color1="#000000", color2="#000000", color3="#000000", color4="#000000", color5="#000000",
            use_master_neg=True,
            use_enhancement=True,
            remove_bg=False,
            vector_mode=False,
            aspect_ratio="1024×1024 (1:1)",
            seed_val="100",
            speed_mode_label="🎯 Master (~15s)",
            batch_size=2,
        )
        steps = list(gen)
        self.assertGreater(len(steps), 0)
        final_status, final_img, final_gallery = steps[-1]
        self.assertIn("Done", final_status)
        self.assertEqual(len(final_gallery), 2)


if __name__ == "__main__":
    unittest.main()
