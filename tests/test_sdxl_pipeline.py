import os
import unittest
from PIL import Image

os.environ["MOCK_IMAGE_GEN"] = "1"

import modules.sdxl_pipeline as sdxl_pipeline
from modules.sdxl_pipeline import (
    load_pipeline,
    generate,
    generate_variations,
    unload_pipeline,
    is_loaded,
    get_current_speed_mode,
)
from modules.lora_router import get_active_adapter, clear_adapters


class DummyPipeline:
    def __init__(self):
        self.loaded_adapters = {}
        self.active_adapters = []
        self.adapter_weights = []
        self.is_lora_enabled = True

    def load_lora_weights(self, source, weight_name=None, adapter_name=None):
        self.loaded_adapters[adapter_name] = source

    def set_adapters(self, adapter_names, adapter_weights=None):
        self.active_adapters = list(adapter_names)
        self.adapter_weights = list(adapter_weights) if adapter_weights else []

    def disable_lora(self):
        self.active_adapters = []
        self.adapter_weights = []
        self.is_lora_enabled = False

    def enable_lora(self):
        self.is_lora_enabled = True


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
        self.assertEqual(get_active_adapter(), "adobe_stock_silhouette")

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

    def test_consecutive_generations_clears_adapter(self):
        cat_cfg = {
            "name": "Adobe Stock Silhouette",
            "lora": {
                "source": "models/loras/silhouette.safetensors",
                "weight": 0.85,
                "trigger_words": "solid black silhouette"
            }
        }
        # First call with category LoRA activates adapter
        generate("owl on branch", width=512, height=512, seed=42, speed_mode="fast", category_cfg=cat_cfg)
        self.assertEqual(get_active_adapter(), "adobe_stock_silhouette")

        # Consecutive call without category LoRA clears adapter
        generate("mountain landscape", width=512, height=512, seed=43, speed_mode="fast", category_cfg=None)
        self.assertIsNone(get_active_adapter())

    def test_generate_ensures_pipeline_is_loaded(self):
        unload_pipeline()
        self.assertFalse(is_loaded())
        generate("abstract vector icon", width=512, height=512, seed=10)
        self.assertTrue(is_loaded())

    def test_mode_switching(self):
        unload_pipeline()
        pipe_fast = load_pipeline("fast")
        self.assertTrue(is_loaded())
        self.assertEqual(pipe_fast, "mock_pipeline")
        self.assertEqual(get_current_speed_mode(), "fast")

        pipe_master = load_pipeline("master")
        self.assertTrue(is_loaded())
        self.assertEqual(pipe_master, "mock_pipeline")
        self.assertEqual(get_current_speed_mode(), "master")

        pipe_fast_again = load_pipeline("fast")
        self.assertTrue(is_loaded())
        self.assertEqual(pipe_fast_again, "mock_pipeline")
        self.assertEqual(get_current_speed_mode(), "fast")

    def test_adapter_coexistence_fast_mode(self):
        dummy_pipe = DummyPipeline()
        sdxl_pipeline._pipeline = dummy_pipe
        cat_cfg = {
            "name": "Adobe Stock Silhouette",
            "lora": {
                "source": "models/loras/silhouette.safetensors",
                "weight": 0.85,
                "trigger_words": "solid black silhouette"
            }
        }
        generate("owl on branch", width=512, height=512, seed=42, speed_mode="fast", category_cfg=cat_cfg)
        self.assertEqual(dummy_pipe.active_adapters, ["lightning_fast", "adobe_stock_silhouette"])
        self.assertEqual(dummy_pipe.adapter_weights, [1.0, 0.85])
        self.assertTrue(dummy_pipe.is_lora_enabled)

    def test_adapter_master_mode(self):
        dummy_pipe = DummyPipeline()
        sdxl_pipeline._pipeline = dummy_pipe
        cat_cfg = {
            "name": "Adobe Stock Silhouette",
            "lora": {
                "source": "models/loras/silhouette.safetensors",
                "weight": 0.85,
                "trigger_words": "solid black silhouette"
            }
        }
        generate("owl on branch", width=512, height=512, seed=42, speed_mode="master", category_cfg=cat_cfg)
        self.assertEqual(dummy_pipe.active_adapters, ["adobe_stock_silhouette"])
        self.assertEqual(dummy_pipe.adapter_weights, [0.85])
        self.assertTrue(dummy_pipe.is_lora_enabled)

    def test_adapter_clear_on_fast_mode_without_lora(self):
        dummy_pipe = DummyPipeline()
        sdxl_pipeline._pipeline = dummy_pipe
        generate("owl on branch", width=512, height=512, seed=42, speed_mode="fast", category_cfg=None)
        self.assertEqual(dummy_pipe.active_adapters, ["lightning_fast"])
        self.assertEqual(dummy_pipe.adapter_weights, [1.0])
        self.assertTrue(dummy_pipe.is_lora_enabled)

    def test_adapter_clear_on_master_mode_without_lora(self):
        dummy_pipe = DummyPipeline()
        sdxl_pipeline._pipeline = dummy_pipe
        generate("owl on branch", width=512, height=512, seed=42, speed_mode="master", category_cfg=None)
        self.assertEqual(dummy_pipe.active_adapters, [])
        self.assertFalse(dummy_pipe.is_lora_enabled)


if __name__ == "__main__":
    unittest.main()
