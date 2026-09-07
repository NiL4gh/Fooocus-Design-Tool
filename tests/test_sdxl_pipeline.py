import os
import unittest
from PIL import Image

os.environ["MOCK_IMAGE_GEN"] = "1"

from modules.sdxl_pipeline import (
    load_pipeline,
    generate,
    generate_variations,
    unload_pipeline,
    is_loaded,
)

class TestSDXLPipeline(unittest.TestCase):
    def tearDown(self):
        unload_pipeline()

    def test_mock_load_and_is_loaded(self):
        self.assertFalse(is_loaded())
        pipe = load_pipeline()
        self.assertEqual(pipe, "mock_pipeline")
        self.assertTrue(is_loaded())

    def test_mock_generate_fast_mode(self):
        prompt = "flat vector mountain icon #3b82f6"
        image, seed = generate(prompt, width=512, height=512, seed=1234, speed_mode="fast")
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (512, 512))
        self.assertEqual(seed, 1234)

    def test_mock_generate_with_category_cfg(self):
        cat_cfg = {
            "name": "Adobe Stock Silhouette",
            "lora": {
                "source": "models/loras/silhouette.safetensors",
                "weight": 0.85,
                "trigger_words": "solid black silhouette"
            }
        }
        image, seed = generate(
            "owl on branch",
            width=512,
            height=512,
            seed=42,
            speed_mode="master",
            category_cfg=cat_cfg
        )
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(seed, 42)

    def test_mock_generate_variations(self):
        variations = generate_variations(
            prompt="geometric logo",
            width=256,
            height=256,
            base_seed=100,
            count=2,
            speed_mode="fast"
        )
        self.assertEqual(len(variations), 2)
        for img, s in variations:
            self.assertIsInstance(img, Image.Image)
            self.assertEqual(img.size, (256, 256))

if __name__ == "__main__":
    unittest.main()
