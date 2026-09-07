import unittest
from modules.style_engine import (
    load_styles,
    get_available_styles,
    apply_styles,
    get_style,
)

class TestStyleEngine(unittest.TestCase):
    def test_load_styles_returns_known_styles(self):
        styles = load_styles()
        self.assertIn("Fooocus V2", styles)
        self.assertIn("SAI Line Art", styles)
        self.assertIn("MRE Flat 2D Art", styles)

    def test_get_available_styles(self):
        names = get_available_styles()
        self.assertIsInstance(names, list)
        self.assertIn("Fooocus Masterpiece", names)

    def test_apply_single_style(self):
        user_prompt = "coffee cup logo"
        user_neg = "ugly"
        p, n = apply_styles(user_prompt, user_neg, ["Fooocus V2"])
        self.assertIn("coffee cup logo", p)
        self.assertIn("highly detailed", p)
        self.assertIn("ugly", n)
        self.assertIn("poor details", n)

    def test_apply_multiple_styles(self):
        user_prompt = "mountain landscape"
        user_neg = ""
        p, n = apply_styles(user_prompt, user_neg, ["Fooocus Masterpiece", "SAI Line Art"])
        self.assertIn("mountain landscape", p)
        self.assertIn("masterpiece", p)
        self.assertIn("line art drawing", p)
        self.assertIn("worst quality", n)
        self.assertIn("shading", n)

    def test_apply_empty_or_none_styles(self):
        p, n = apply_styles("raw prompt", "raw neg", [])
        self.assertEqual(p, "raw prompt")
        self.assertEqual(n, "raw neg")

        p, n = apply_styles("raw prompt", "raw neg", None)
        self.assertEqual(p, "raw prompt")
        self.assertEqual(n, "raw neg")

if __name__ == "__main__":
    unittest.main()
