import os
import unittest

os.environ["MOCK_IMAGE_GEN"] = "1"

from modules.ai_copilot import enhance_prompt_with_ai, _heuristic_enhance


class TestAICopilot(unittest.TestCase):
    def test_heuristic_enhance_logo(self):
        prompt = "coffee shop"
        p, n = _heuristic_enhance(prompt, "Logo")
        self.assertIn("coffee shop", p)
        self.assertIn("minimalist", p.lower())
        self.assertIn("vector", p.lower())
        self.assertIn("photograph", n.lower())

    def test_heuristic_enhance_silhouette(self):
        prompt = "falcon in flight"
        p, n = _heuristic_enhance(prompt, "Adobe Stock Silhouette")
        self.assertIn("falcon in flight", p)
        self.assertIn("silhouette", p.lower())
        self.assertIn("color", n.lower())

    def test_enhance_prompt_with_ai_fallback(self):
        prompt = "vintage car"
        p, n = enhance_prompt_with_ai(prompt, category="Poster")
        self.assertIn("vintage car", p)
        self.assertIn("poster", p.lower())
        self.assertGreater(len(n), 5)

    def test_enhance_empty_prompt(self):
        p, n = enhance_prompt_with_ai("")
        self.assertGreater(len(p), 0)
        self.assertGreater(len(n), 0)


if __name__ == "__main__":
    unittest.main()
