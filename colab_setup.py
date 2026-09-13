"""
Fooocus Designer 2.0 — Google Colab Setup Script
Run this in a Colab cell to set up and launch Fooocus Designer 2.0.
"""
import argparse
import os
import subprocess
import sys


def ensure_repo_dir(repo_url: str = "https://github.com/NiL4gh/Fooocus-Design-Tool.git"):
    """Ensure working directory is inside the cloned repository root."""
    if os.path.exists("launch.py") and os.path.exists("requirements.txt"):
        return

    if not os.path.exists("Fooocus-Design-Tool"):
        print("\n📦 Cloning repository...")
        subprocess.run(["git", "clone", repo_url], check=True)

    if os.path.exists("Fooocus-Design-Tool"):
        os.chdir("Fooocus-Design-Tool")


def build_parser():
    """Build command-line parser for colab_setup."""
    parser = argparse.ArgumentParser(description="Fooocus Designer 2.0 Colab Setup & Bootstrapper")
    parser.add_argument("--share", action="store_true", default=True, help="Enable Gradio public share link (default: True)")
    parser.add_argument("--no-share", action="store_false", dest="share", help="Disable Gradio public share link")
    parser.add_argument("--ngrok", action="store_true", default=False, help="Use ngrok tunnel instead of Gradio share")
    parser.add_argument("--ngrok-token", type=str, default=None, help="ngrok authentication token")
    parser.add_argument("--preload", action="store_true", default=False, help="Pre-download SDXL base weights into cache")
    parser.add_argument("--demo", action="store_true", default=False, help="Launch in mock demo mode")
    return parser


def setup_and_launch(share=True, use_ngrok=False, ngrok_token=None, preload=False, demo=False):
    """
    Set up and launch Fooocus Designer 2.0 on Google Colab.
    
    Args:
        share: Use Gradio's built-in sharing (default True).
        use_ngrok: Use ngrok instead of Gradio share.
        ngrok_token: Your ngrok auth token (required if use_ngrok=True).
        preload: Pre-download and cache SDXL model weights.
        demo: Launch in mock demo mode (no GPU required).
    """
    print("=" * 60)
    print("🎨 Fooocus Designer 2.0 — Colab Setup")
    print("   Engine: RunDiffusion/Juggernaut-XL-v9 (SDXL) + Baked LoRAs")
    print("   Performance: ⚡ Fast (~3s Lightning) | 🎯 Master (~15s SDXL)")
    print("=" * 60)

    # 1. Ensure repo root directory
    ensure_repo_dir()

    # 2. Install dependencies
    print("\n📥 Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"], check=True)

    # 3. Check / Install PyTorch with CUDA if needed
    try:
        import torch
        if not torch.cuda.is_available() and not demo:
            print("⚠️ CUDA not available. Installing PyTorch with CUDA...")
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision", "--extra-index-url",
                "https://download.pytorch.org/whl/cu121", "-q"
            ], check=True)
    except ImportError:
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "--extra-index-url",
            "https://download.pytorch.org/whl/cu121", "-q"
        ], check=True)

    # 4. Optional model pre-caching
    if preload:
        print("\n📥 Pre-downloading SDXL models into cache...")
        subprocess.run([
            sys.executable, "-c",
            "from modules.sdxl_pipeline import load_pipeline, unload_pipeline; "
            "load_pipeline(speed_mode='fast'); unload_pipeline(); "
            "print('✅ SDXL models cached successfully!')"
        ], check=True)

    # 5. Optional ngrok setup
    if use_ngrok and ngrok_token:
        print("\n🔗 Setting up ngrok tunnel...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyngrok", "-q"], check=True)
        from pyngrok import ngrok
        ngrok.set_auth_token(ngrok_token)
        tunnel = ngrok.connect(7865)
        print(f"🌐 Public URL: {tunnel.public_url}")

    # 6. Launch
    print("\n🚀 Launching Fooocus Designer 2.0...")
    print("=" * 60)

    launch_cmd = [sys.executable, "launch.py"]
    if share and not use_ngrok:
        launch_cmd.append("--share")
    if demo:
        launch_cmd.append("--demo")

    subprocess.run(launch_cmd)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    setup_and_launch(
        share=args.share,
        use_ngrok=args.ngrok,
        ngrok_token=args.ngrok_token,
        preload=args.preload,
        demo=args.demo
    )
