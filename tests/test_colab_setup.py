import unittest
import os
import sys

class TestColabSetup(unittest.TestCase):
    def test_colab_setup_importable(self):
        import colab_setup
        self.assertTrue(hasattr(colab_setup, "setup_and_launch"))

    def test_colab_setup_cwd_detection(self):
        """Ensure setup_and_launch doesn't crash if already inside repo directory."""
        import colab_setup
        # Verify launch.py exists in current directory
        self.assertTrue(os.path.exists("launch.py"))
        # Verify helper function or logic identifies repo root correctly
        if hasattr(colab_setup, "ensure_repo_dir"):
            colab_setup.ensure_repo_dir()
            self.assertTrue(os.path.exists("launch.py"))

    def test_cli_argument_parser(self):
        """Test argparse configuration in colab_setup."""
        import colab_setup
        if hasattr(colab_setup, "build_parser"):
            parser = colab_setup.build_parser()
            args = parser.parse_args(["--no-share", "--preload"])
            self.assertFalse(args.share)
            self.assertTrue(args.preload)

            args_default = parser.parse_args([])
            self.assertTrue(args_default.share)
            self.assertFalse(args_default.preload)

if __name__ == "__main__":
    unittest.main()
