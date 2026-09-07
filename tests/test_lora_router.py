import os
import tempfile
import unittest
os.environ["MOCK_IMAGE_GEN"] = "1"

import modules.lora_router as lora_router
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
        self.adapter_weights = []
        self.is_lora_enabled = True
        self.load_count = 0

    def load_lora_weights(self, source, weight_name=None, adapter_name=None):
        self.load_count += 1
        self.loaded_adapters[adapter_name] = source

    def set_adapters(self, adapter_names, adapter_weights=None):
        self.active_adapters = list(adapter_names)
        self.adapter_weights = list(adapter_weights) if adapter_weights is not None else []

    def disable_lora(self):
        self.active_adapters = []
        self.adapter_weights = []
        self.is_lora_enabled = False

    def enable_lora(self):
        self.is_lora_enabled = True

class TestLoraRouter(unittest.TestCase):
    def setUp(self):
        self.pipe = DummyPipeline()
        clear_adapters(self.pipe)
        lora_router._loaded_adapters.clear()

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
        self.assertTrue(self.pipe.is_lora_enabled)

    def test_apply_category_lora_without_lora(self):
        cfg = {
            "name": "Artwork",
            "lora": None
        }
        trigger = apply_category_lora(self.pipe, cfg)
        self.assertEqual(trigger, "")
        self.assertIsNone(get_active_adapter())
        self.assertFalse(self.pipe.is_lora_enabled)

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
        self.assertTrue(self.pipe.is_lora_enabled)
        clear_adapters(self.pipe)
        self.assertIsNone(get_active_adapter())
        self.assertEqual(len(self.pipe.active_adapters), 0)
        self.assertFalse(self.pipe.is_lora_enabled)

    def test_apply_category_lora_with_base_adapters(self):
        cfg = {
            "name": "Adobe Stock Silhouette",
            "lora": {
                "source": "models/loras/silhouette.safetensors",
                "weight": 0.85,
                "trigger_words": "solid black silhouette"
            }
        }
        trigger = apply_category_lora(
            self.pipe,
            cfg,
            base_adapters=["lightning_fast"],
            base_weights=[1.0]
        )
        self.assertEqual(trigger, "solid black silhouette")
        self.assertEqual(get_active_adapter(), "adobe_stock_silhouette")
        self.assertEqual(self.pipe.active_adapters, ["lightning_fast", "adobe_stock_silhouette"])
        self.assertEqual(self.pipe.adapter_weights, [1.0, 0.85])
        self.assertTrue(self.pipe.is_lora_enabled)

    def test_clear_adapters_with_base_adapters(self):
        cfg = {
            "name": "Flat Vector",
            "lora": {
                "source": "models/loras/flat_vector.safetensors",
                "weight": 0.8,
                "trigger_words": "flat vector"
            }
        }
        apply_category_lora(
            self.pipe,
            cfg,
            base_adapters=["lightning_fast"],
            base_weights=[1.0]
        )
        self.assertEqual(self.pipe.active_adapters, ["lightning_fast", "flat_vector"])

        clear_adapters(self.pipe, base_adapters=["lightning_fast"], base_weights=[1.0])
        self.assertIsNone(get_active_adapter())
        self.assertEqual(self.pipe.active_adapters, ["lightning_fast"])
        self.assertEqual(self.pipe.adapter_weights, [1.0])
        self.assertTrue(self.pipe.is_lora_enabled)

    def test_is_lora_available(self):
        # Empty string and None
        self.assertFalse(is_lora_available(""))
        self.assertFalse(is_lora_available(None))

        # Nonexistent safetensors / bin / pt / ckpt files (must be False)
        self.assertFalse(is_lora_available("nonexistent_model_12345.safetensors"))
        self.assertFalse(is_lora_available("models/loras/nonexistent.safetensors"))
        self.assertFalse(is_lora_available("missing.bin"))
        self.assertFalse(is_lora_available("weights/missing.pt"))
        self.assertFalse(is_lora_available("checkpoint.ckpt"))

        # Valid HF repo (True)
        self.assertTrue(is_lora_available("ByteDance/SDXL-Lightning"))
        self.assertTrue(is_lora_available("latent-consistency/lcm-lora-sdxl"))
        self.assertTrue(is_lora_available("org-name/repo_model.v1"))

        # Invalid HF repo strings
        self.assertFalse(is_lora_available("just_a_word"))
        self.assertFalse(is_lora_available("/leading_slash/repo"))
        self.assertFalse(is_lora_available("trailing_slash/repo/"))
        self.assertFalse(is_lora_available("a/b/c"))

        # Existing local file (True)
        self.assertTrue(is_lora_available(__file__))
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            self.assertTrue(is_lora_available(tmp_path))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_apply_category_lora_force_active_false(self):
        cfg = {
            "name": "Line Art",
            "lora": {
                "source": "models/loras/line_art.safetensors",
                "weight": 0.7,
                "trigger_words": "clean line art"
            }
        }
        trigger = apply_category_lora(self.pipe, cfg, force_active=False)
        self.assertEqual(trigger, "clean line art")
        # Adapter should be loaded in pipeline and cache
        self.assertIn("line_art", self.pipe.loaded_adapters)
        self.assertIn("line_art", lora_router._loaded_adapters)
        # But not activated
        self.assertEqual(self.pipe.active_adapters, [])
        self.assertIsNone(get_active_adapter())

    def test_switching_lifecycle(self):
        cfg_a = {
            "name": "Category A",
            "lora": {
                "source": "models/loras/a.safetensors",
                "weight": 0.75,
                "trigger_words": "trigger_a"
            }
        }
        cfg_b = {
            "name": "Category B",
            "lora": None
        }
        cfg_c = {
            "name": "Category C",
            "lora": {
                "source": "models/loras/c.safetensors",
                "weight": 0.9,
                "trigger_words": "trigger_c"
            }
        }

        # Step 1: Category A (active)
        trigger_a = apply_category_lora(self.pipe, cfg_a, force_active=True)
        self.assertEqual(trigger_a, "trigger_a")
        self.assertEqual(get_active_adapter(), "category_a")
        self.assertIn("category_a", self.pipe.active_adapters)
        self.assertTrue(self.pipe.is_lora_enabled)

        # Step 2: Category B (none, disable_lora called)
        trigger_b = apply_category_lora(self.pipe, cfg_b)
        self.assertEqual(trigger_b, "")
        self.assertIsNone(get_active_adapter())
        self.assertEqual(self.pipe.active_adapters, [])
        self.assertFalse(self.pipe.is_lora_enabled)

        # Step 3: Category C (active, verify enable_lora called and active adapter updated)
        trigger_c = apply_category_lora(self.pipe, cfg_c, force_active=True)
        self.assertEqual(trigger_c, "trigger_c")
        self.assertEqual(get_active_adapter(), "category_c")
        self.assertIn("category_c", self.pipe.active_adapters)
        self.assertTrue(self.pipe.is_lora_enabled)

    def test_mock_cache_parity(self):
        cfg = {
            "name": "Adobe Stock Silhouette",
            "lora": {
                "source": "models/loras/silhouette.safetensors",
                "weight": 0.85,
                "trigger_words": "solid black silhouette"
            }
        }
        # First call loads weights
        apply_category_lora(self.pipe, cfg)
        self.assertEqual(self.pipe.load_count, 1)
        self.assertIn("adobe_stock_silhouette", lora_router._loaded_adapters)

        # Second call uses cache without calling load_lora_weights again
        apply_category_lora(self.pipe, cfg)
        self.assertEqual(self.pipe.load_count, 1)

if __name__ == "__main__":
    unittest.main()
