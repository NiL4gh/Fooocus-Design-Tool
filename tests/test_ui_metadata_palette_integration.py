import os
import unittest
from PIL import Image
from modules.metadata_manager import build_metadata, save_image_with_metadata, extract_metadata
from ui.design_main import _save_image, _on_palette_preset_change, _on_image_drop_inspect

class TestUIMetadataPaletteIntegration(unittest.TestCase):
    def test_save_image_embeds_metadata(self):
        img = Image.new("RGBA", (64, 64), (0, 128, 255, 255))
        meta = build_metadata(
            category="Adobe Stock Flat Vector",
            prompt="flat vector logo",
            seed=777,
            speed_mode="fast",
        )
        saved_path = _save_image(img, "outputs", metadata=meta)
        self.assertTrue(os.path.exists(saved_path))
        extracted = extract_metadata(saved_path)
        self.assertIsNotNone(extracted)
        self.assertEqual(extracted["prompt"], "flat vector logo")
        self.assertEqual(extracted["seed"], 777)
        try:
            os.remove(saved_path)
        except Exception:
            pass

    def test_on_palette_preset_change(self):
        updates = _on_palette_preset_change("Cyberpunk Neon")
        self.assertEqual(len(updates), 5)
        self.assertEqual(updates[0]["value"], "#00F0FF")
        self.assertEqual(updates[1]["value"], "#FF003C")

    def test_on_image_drop_inspect(self):
        img = Image.new("RGB", (64, 64), (200, 200, 200))
        meta = build_metadata(
            category="Die-Cut Sticker",
            prompt="holographic sticker of astronaut",
            negative_prompt="blurry",
            seed=42,
            speed_mode="master",
            selected_styles=["Fooocus V2"],
            colors=["#FFB3BA", "#BAFFC9", "#000000", "#000000", "#000000"],
        )
        os.makedirs("outputs", exist_ok=True)
        temp_path = os.path.abspath("outputs/test_inspect.png")
        save_image_with_metadata(img, temp_path, meta)

        try:
            status, cat, pr, neg, spd, stl, sd, c1, c2, c3, c4, c5 = _on_image_drop_inspect(temp_path)
            self.assertIn("Loaded settings", status)
            self.assertEqual(cat["value"], "Die-Cut Sticker")
            self.assertEqual(pr["value"], "holographic sticker of astronaut")
            self.assertEqual(neg["value"], "blurry")
            self.assertEqual(spd["value"], "🎯 Master (~15s)")
            self.assertEqual(stl["value"], ["Fooocus V2"])
            self.assertEqual(sd["value"], "42")
            self.assertEqual(c1["value"], "#FFB3BA")
            self.assertEqual(c2["value"], "#BAFFC9")
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    def test_on_image_drop_inspect_no_metadata(self):
        img = Image.new("RGB", (32, 32), (50, 50, 50))
        os.makedirs("outputs", exist_ok=True)
        plain_path = os.path.abspath("outputs/plain_inspect.png")
        img.save(plain_path)
        try:
            res = _on_image_drop_inspect(plain_path)
            self.assertIn("No Fooocus Designer metadata", res[0])
        finally:
            if os.path.exists(plain_path):
                try:
                    os.remove(plain_path)
                except Exception:
                    pass

if __name__ == "__main__":
    unittest.main()
