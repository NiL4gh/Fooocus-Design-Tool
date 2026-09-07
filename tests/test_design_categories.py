import unittest
from modules.design_categories import (
    load_categories,
    get_category,
    get_category_lora,
    get_category_names,
)

class TestDesignCategories(unittest.TestCase):
    def test_baked_loras_exist_in_categories(self):
        silhouette = get_category("Adobe Stock Silhouette")
        self.assertIsNotNone(silhouette)
        lora_cfg = get_category_lora("Adobe Stock Silhouette")
        self.assertIsNotNone(lora_cfg)
        self.assertIn("trigger_words", lora_cfg)
        self.assertIn("weight", lora_cfg)
        self.assertGreaterEqual(lora_cfg["weight"], 0.7)

        vector = get_category("Adobe Stock Flat Vector")
        self.assertIsNotNone(vector)
        v_lora = get_category_lora("Adobe Stock Flat Vector")
        self.assertIsNotNone(v_lora)
        self.assertIn("flat", v_lora["trigger_words"].lower())

    def test_category_without_lora(self):
        artwork = get_category("Artwork")
        self.assertIsNotNone(artwork)
        lora_cfg = get_category_lora("Artwork")
        self.assertIsNone(lora_cfg)

if __name__ == "__main__":
    unittest.main()
