import unittest
from PIL import Image
from modules.palette_control import (
    PALETTE_PRESETS,
    get_palette_presets,
    get_preset_colors,
    inject_palette_prompt,
    apply_palette_post,
    hex_to_color_name,
)

class TestPaletteControl(unittest.TestCase):
    def test_get_palette_presets(self):
        presets = get_palette_presets()
        self.assertIn("Custom / None", presets)
        self.assertIn("Pastel Dreams", presets)
        self.assertIn("Cyberpunk Neon", presets)
        self.assertIn("Earthy Boho", presets)
        self.assertIn("Corporate Tech", presets)
        self.assertIn("Retro Sunset", presets)
        self.assertIn("Luxury Gold", presets)
        self.assertIn("Nordic Minimalist", presets)

        # Verify each preset returns exactly 5 hex colors
        for name in presets:
            colors = get_preset_colors(name)
            self.assertEqual(len(colors), 5, f"Preset {name} must contain exactly 5 colors")
            for c in colors:
                self.assertTrue(c.startswith("#"), f"Color {c} in preset {name} must start with #")
                self.assertEqual(len(c), 7, f"Color {c} in preset {name} must be 7 characters (#RRGGBB)")

    def test_unknown_preset_fallback(self):
        colors = get_preset_colors("Nonexistent Preset")
        self.assertEqual(colors, ["#000000", "#000000", "#000000", "#000000", "#000000"])

    def test_inject_palette_prompt(self):
        prompt = "minimalist vector fox"
        colors = get_preset_colors("Pastel Dreams")
        injected = inject_palette_prompt(prompt, colors)
        self.assertIn("using a color palette of", injected)
        self.assertIn("minimalist vector fox", injected)

    def test_apply_palette_post(self):
        img = Image.new("RGB", (64, 64), color=(100, 100, 100))
        colors = get_preset_colors("Cyberpunk Neon")
        tinted = apply_palette_post(img, colors, strength=0.5)
        self.assertEqual(tinted.size, (64, 64))

if __name__ == "__main__":
    unittest.main()
