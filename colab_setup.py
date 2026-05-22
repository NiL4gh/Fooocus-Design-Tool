"""
Fooocus Design Tool — Google Colab Setup Script
Run this in a Colab cell to set up and launch the tool.
"""
import subprocess
import sys
import os

def setup_and_launch(share=True, use_ngrok=False, ngrok_token=None):
    """
    Set up and launch Fooocus Design Tool on Google Colab.
    
    Args:
        share: Use Gradio's built-in sharing (default True).
        use_ngrok: Use ngrok instead of Gradio share.
        ngrok_token: Your ngrok auth token (required if use_ngrok=True).
    """
    print("=" * 60)
    print("🎨 Fooocus Design Tool — Colab Setup")
    print("=" * 60)

    # Check if already cloned
    if not os.path.exists('Fooocus-Design-Tool'):
        print("\n📦 Cloning repository...")
        subprocess.run(['git', 'clone', 'https://github.com/NiL4gh/Fooocus-Design-Tool.git'], check=True)
    
    os.chdir('Fooocus-Design-Tool')

    # Install dependencies
    print("\n📥 Installing dependencies...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt', '-q'], check=True)

    # Install PyTorch with CUDA if not present
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠️ CUDA not available. Installing PyTorch with CUDA...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', 
                          'torch', 'torchvision', '--extra-index-url', 
                          'https://download.pytorch.org/whl/cu121', '-q'], check=True)
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install',
                      'torch', 'torchvision', '--extra-index-url',
                      'https://download.pytorch.org/whl/cu121', '-q'], check=True)

    # Optional ngrok setup
    if use_ngrok and ngrok_token:
        print("\n🔗 Setting up ngrok tunnel...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'pyngrok', '-q'], check=True)
        from pyngrok import ngrok
        ngrok.set_auth_token(ngrok_token)
        tunnel = ngrok.connect(7865)
        print(f"🌐 Public URL: {tunnel.public_url}")

    # Launch
    print("\n🚀 Launching Fooocus Design Tool...")
    print("=" * 60)

    if share:
        subprocess.run([sys.executable, 'launch.py', '--share'])
    else:
        subprocess.run([sys.executable, 'launch.py'])


if __name__ == '__main__':
    setup_and_launch(share=True)
