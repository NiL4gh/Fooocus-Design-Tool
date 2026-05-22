"""
Background Remover Module
Uses rembg for post-processing background removal to create transparent PNGs.
"""
from PIL import Image

def remove_background(image, progress_cb=None):
    """Remove background from PIL Image using rembg. Returns RGBA PIL Image."""
    if progress_cb: progress_cb("Removing background...")
    try:
        from rembg import remove
        result = remove(image)
        if progress_cb: progress_cb("Background removed!")
        return result
    except ImportError:
        print("[BG Remover] rembg not installed. Run: pip install rembg[gpu]")
        return image
    except Exception as e:
        print(f"[BG Remover] Failed: {e}")
        return image
