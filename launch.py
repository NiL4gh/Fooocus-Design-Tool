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

def patch_gradio_client():
    """Apply type-safe patch to gradio_client to prevent Pydantic v2 OpenAPI schema crashes."""
    try:
        import inspect
        import gradio_client
        utils_file = inspect.getfile(gradio_client.utils)
        if os.path.exists(utils_file):
            with open(utils_file, 'r', encoding='utf-8') as f:
                content = f.read()
            modified = False
            old_get_type = 'def get_type(schema: dict):\n    if "const" in schema:'
            new_get_type = 'def get_type(schema: dict):\n    if not isinstance(schema, dict):\n        return "Any"\n    if "const" in schema:'
            if old_get_type in content:
                content = content.replace(old_get_type, new_get_type)
                modified = True
            old_json = 'def _json_schema_to_python_type(schema: Any, defs) -> str:\n    """Convert the json schema into a python type hint"""\n    if schema == {}:\n        return "Any"'
            new_json = 'def _json_schema_to_python_type(schema: Any, defs) -> str:\n    """Convert the json schema into a python type hint"""\n    if not isinstance(schema, dict):\n        return "Any"\n    if schema == {}:\n        return "Any"'
            if old_json in content:
                content = content.replace(old_json, new_json)
                modified = True
            if modified:
                with open(utils_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                print("[Setup] Successfully patched gradio_client utils.py to handle boolean OpenAPI schemas safely.")
    except Exception as e:
        print(f"[Setup] Warning: Could not patch gradio_client: {e}")

patch_gradio_client()

# Create output directory
os.makedirs(os.path.join(root, 'outputs'), exist_ok=True)
os.makedirs(os.path.join(root, 'models'), exist_ok=True)

import torch
if "--demo" in sys.argv or "--mock" in sys.argv or (not torch.cuda.is_available() and "--cpu" not in sys.argv):
    os.environ["MOCK_IMAGE_GEN"] = "1"
    print("[Fooocus Design Tool] Running in Local Demo (Mock) Mode. Generation will be instant and local (no massive model downloads or GPU required).")
    print("  To override and run the heavy model on your CPU, launch with '--cpu'")
else:
    print("[Fooocus Design Tool] GPU detected or --cpu flag active. Pipeline will attempt to run on device:", "cuda" if torch.cuda.is_available() else "cpu")

print('[Fooocus Design Tool] Dependencies OK. Launching UI...')

# Import and launch
from webui import app

app.queue().launch(
    server_name="127.0.0.1",
    server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7865")),
    share=True,
    inbrowser="--no-browser" not in sys.argv,
)
