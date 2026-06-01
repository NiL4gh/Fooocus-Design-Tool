"""
Background Remover Module
Uses rembg for post-processing background removal to create transparent PNGs.
"""
from PIL import Image

def remove_background(image, progress_cb=None):
    """Remove background from PIL Image using rembg. Returns RGBA PIL Image."""
    if progress_cb: progress_cb("Removing background...")
    import os
    if os.environ.get("MOCK_IMAGE_GEN") == "1":
        # Mock background remover: Convert "#0f172a" background pixels to transparent
        rgba = image.convert("RGBA")
        datas = rgba.getdata()
        newData = []
        for item in datas:
            # check if pixel is near #0f172a (our dark background)
            if item[0] < 20 and item[1] < 30 and item[2] < 50:
                newData.append((255, 255, 255, 0)) # transparent
            else:
                newData.append(item)
        rgba.putdata(newData)
        if progress_cb: progress_cb("[Mock] Background removed!")
        return rgba
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
