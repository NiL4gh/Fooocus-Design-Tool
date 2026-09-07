import os
import unittest
os.environ["MOCK_IMAGE_GEN"] = "1"

from modules.lora_router import (
    apply_category_lora,
    get_active_adapter,
    clear_adapters,
    is_lora_available,
)

class DummyPipeline:
    def __init__(self):
        self.loaded_adapters = {}
        self.active_adapters = []

    def load_lora_weights(self, source, weight_name=None, adapter_name=None):
        self.loaded_adapters[adapter_name] = source

    def set_adapters(self, adapter_names, adapter_weights=None):
        self.active_adapters = adapter_names

    def disable_lora(self):
        self.active_adapters = []

class TestLoraRouter(unittest.TestCase):
    def setUp(self):
        self.pipe = DummyPipeline()
        clear_adapters(self.pipe)

    def test_apply_category_lora_with_valid_config(self):
        cfg = {
            "name": "Adobe Stock Silhouette",
            "lora": {
                "source": "models/loras/silhouette.safetensors",
                "weight": 0.85,
                "trigger_words": "solid black silhouette, vector style"
            }
        }
        trigger = apply_category_lora(self.pipe, cfg)
        self.assertEqual(trigger, "solid black silhouette, vector style")
        self.assertEqual(get_active_adapter(), "adobe_stock_silhouette")
        self.assertIn("adobe_stock_silhouette", self.pipe.active_adapters)

    def test_apply_category_lora_without_lora(self):
        cfg = {
            "name": "Artwork",
            "lora": None
        }
        trigger = apply_category_lora(self.pipe, cfg)
        self.assertEqual(trigger, "")
        self.assertIsNone(get_active_adapter())

    def test_clear_adapters(self):
        cfg = {
            "name": "Flat Vector",
            "lora": {
                "source": "models/loras/flat_vector.safetensors",
                "weight": 0.8,
                "trigger_words": "flat vector"
            }
        }
        apply_category_lora(self.pipe, cfg)
        self.assertIsNotNone(get_active_adapter())
        clear_adapters(self.pipe)
        self.assertIsNone(get_active_adapter())
        self.assertEqual(len(self.pipe.active_adapters), 0)

if __name__ == "__main__":
    unittest.main()
