import os
import json
import glob
import unittest

class TestStyleJSONValidity(unittest.TestCase):
    def test_all_style_json_files_are_valid(self):
        style_dir = os.path.join("config", "sdxl_styles")
        json_files = glob.glob(os.path.join(style_dir, "*.json"))
        self.assertGreater(len(json_files), 0, "No style JSON files found in config/sdxl_styles")

        style_names = set()
        for fpath in json_files:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIsInstance(data, list, f"{fpath} must contain a list of style objects")
            for item in data:
                self.assertIn("name", item)
                self.assertIn("prompt", item)
                self.assertIn("negative_prompt", item)
                self.assertIn("{prompt}", item["prompt"], f"Style '{item['name']}' in {fpath} must contain '{{prompt}}'")
                self.assertNotIn(item["name"], style_names, f"Duplicate style name '{item['name']}' across files")
                style_names.add(item["name"])

        # Check key Fooocus styles exist
        self.assertIn("Fooocus V2", style_names)
        self.assertIn("Fooocus Masterpiece", style_names)
        self.assertIn("SAI Line Art", style_names)
        self.assertIn("MRE Flat 2D Art", style_names)

if __name__ == "__main__":
    unittest.main()
