import unittest
from modules.auto_prompt_enhancer import (
    enhance_prompt,
    build_negative_prompt,
    is_vector_silhouette_category,
)

class TestAutoPromptEnhancer(unittest.TestCase):
    def test_vector_silhouette_category_detection(self):
        self.assertTrue(is_vector_silhouette_category("Adobe Stock Silhouette"))
        self.assertTrue(is_vector_silhouette_category("Adobe Stock Flat Vector"))
        self.assertTrue(is_vector_silhouette_category("Vector Silhouette"))
        self.assertFalse(is_vector_silhouette_category("Artwork"))
        self.assertFalse(is_vector_silhouette_category("Poster"))

    def test_enhance_prompt_with_styles(self):
        prompt = "majestic eagle"
        enhanced = enhance_prompt(prompt, "Artwork", use_enhancement=True, selected_styles=["Fooocus V2"])
        self.assertIn("majestic eagle", enhanced)
        self.assertIn("highly detailed", enhanced)

    def test_vector_enhancement_priority(self):
        prompt = "running cheetah"
        enhanced = enhance_prompt(prompt, "Adobe Stock Silhouette", use_enhancement=True)
        self.assertIn("solid black silhouette", enhanced)
        self.assertIn("running cheetah", enhanced)

    def test_build_negative_prompt_with_styles(self):
        neg = build_negative_prompt("blurry", "Adobe Stock Flat Vector", use_master_negative=True, selected_styles=["MRE Flat 2D Art"])
        self.assertIn("blurry", neg)
        # Check category master negative
        self.assertIn("photograph", neg)
        # Check style negative
        self.assertIn("3d render", neg)

    def test_none_inputs_handled_safely(self):
        # Should not raise AttributeError when prompt or user_negative is None
        enhanced = enhance_prompt(None, "Artwork", use_enhancement=False)
        self.assertEqual(enhanced, "")

        neg = build_negative_prompt(None, "Adobe Stock Silhouette", use_master_negative=True)
        self.assertIn("photograph", neg)

if __name__ == "__main__":
    unittest.main()
