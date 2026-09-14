import os
import tempfile
import unittest
from PIL import Image
from modules.metadata_manager import (
    build_metadata,
    save_image_with_metadata,
    extract_metadata,
    format_metadata_for_display,
)

class TestMetadataManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_img = Image.new("RGBA", (128, 128), color=(255, 0, 0, 255))

    def tearDown(self):
        for f in os.listdir(self.temp_dir):
            try:
                os.remove(os.path.join(self.temp_dir, f))
            except Exception:
                pass
        try:
            os.rmdir(self.temp_dir)
        except Exception:
            pass

    def test_build_metadata(self):
        meta = build_metadata(
            category="Adobe Stock Flat Vector",
            prompt="origami fox",
            negative_prompt="blurry",
            seed=42,
            speed_mode="fast",
            selected_styles=["Fooocus V2"],
            colors=["#FF0000", "#00FF00"],
            width=1024,
            height=1024,
            loras=[{"name": "flat_vector", "weight": 0.85}],
        )
        self.assertEqual(meta["category"], "Adobe Stock Flat Vector")
        self.assertEqual(meta["prompt"], "origami fox")
        self.assertEqual(meta["seed"], 42)
        self.assertEqual(meta["speed_mode"], "fast")
        self.assertEqual(meta["version"], "2.0")
        self.assertIn("loras", meta)

    def test_save_and_extract_roundtrip(self):
        meta = build_metadata(
            category="Adobe Stock Silhouette",
            prompt="running horse",
            negative_prompt="color, 3D",
            seed=12345,
            speed_mode="master",
            selected_styles=[],
            colors=[],
            width=1024,
            height=1024,
        )
        filepath = os.path.join(self.temp_dir, "test_asset.png")
        saved_path = save_image_with_metadata(self.test_img, filepath, meta)
        self.assertTrue(os.path.exists(saved_path))

        # Extract from file path
        extracted = extract_metadata(saved_path)
        self.assertIsNotNone(extracted)
        self.assertEqual(extracted["category"], "Adobe Stock Silhouette")
        self.assertEqual(extracted["prompt"], "running horse")
        self.assertEqual(extracted["seed"], 12345)
        self.assertEqual(extracted["speed_mode"], "master")

        # Extract from PIL image opened directly
        with Image.open(saved_path) as loaded_img:
            extracted_img = extract_metadata(loaded_img)
            self.assertIsNotNone(extracted_img)
            self.assertEqual(extracted_img["category"], "Adobe Stock Silhouette")

    def test_extract_from_image_without_metadata(self):
        filepath = os.path.join(self.temp_dir, "plain.png")
        self.test_img.save(filepath)
        extracted = extract_metadata(filepath)
        self.assertIsNone(extracted)

    def test_format_metadata_for_display(self):
        meta = {
            "category": "Die-Cut Sticker",
            "prompt": "cute cat",
            "seed": 999,
            "speed_mode": "fast",
            "selected_styles": ["SAI Line Art"],
            "colors": ["#FFA500"],
            "version": "2.0",
        }
        display_str = format_metadata_for_display(meta)
        self.assertIn("Die-Cut Sticker", display_str)
        self.assertIn("cute cat", display_str)
        self.assertIn("999", display_str)

    def test_stealth_sanitization(self):
        """Verify stealth sanitization completely strips AI signatures and injects clean commercial metadata."""
        from modules.metadata_manager import save_sanitized_image
        meta = build_metadata(category="Logo", prompt="secret ai prompt", seed=777)
        filepath = os.path.join(self.temp_dir, "stealth_asset.png")
        
        # Save with stealth_mode=True
        save_image_with_metadata(self.test_img, filepath, meta, stealth_mode=True)
        
        # Verify AI metadata extractor finds nothing
        extracted = extract_metadata(filepath)
        self.assertIsNone(extracted)
        
        # Inspect raw PNG chunks directly
        with Image.open(filepath) as img:
            self.assertNotIn("fooocus_designer_metadata", img.info)
            self.assertNotIn("parameters", img.info)
            self.assertIn("Software", img.info)
            self.assertTrue(any(brand in img.info["Software"] for brand in ["Adobe", "Affinity", "Corel"]))


if __name__ == "__main__":
    unittest.main()
