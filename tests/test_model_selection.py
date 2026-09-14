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
        self.assertGreater(seed, 0)


if __name__ == "__main__":
    unittest.main()
