"""
Fooocus Design Tool — Launcher
Installs dependencies and launches the web UI.
"""
import os
import sys
import subprocess

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)

print('[Fooocus Design Tool] Checking dependencies...')


def is_installed(package):
    try:
        __import__(package)
        return True
    except ImportError:
        return False


def install_requirements():
    """Install required packages."""
    req_file = os.path.join(root, 'requirements.txt')
    if os.path.exists(req_file):
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', req_file, '-q'])


# Check core dependencies
core_deps = ['torch', 'gradio', 'diffusers', 'transformers', 'accelerate', 'PIL']
missing = [d for d in core_deps if not is_installed(d.split('.')[0] if '.' in d else d)]

if missing:
    print(f'[Setup] Installing missing dependencies: {missing}')
    install_requirements()

# Create output directory
os.makedirs(os.path.join(root, 'outputs'), exist_ok=True)
os.makedirs(os.path.join(root, 'models'), exist_ok=True)

print('[Fooocus Design Tool] Dependencies OK. Launching UI...')

# Import and launch
from webui import app

app.queue().launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7865")),
    share="--share" in sys.argv,
    inbrowser="--no-browser" not in sys.argv,
)
