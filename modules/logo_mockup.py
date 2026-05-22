"""
Logo Mockup Module
Dynamic logo-on-product mockup generation using Z-Image-Turbo and PIL compositing.
Supports customizable layout styles (Single, Bento Knolling, Realistic Ambient).
"""
import os
import time
import random
import numpy as np
from PIL import Image

SUPPORTED_PRODUCTS = ["T-Shirt", "Mug", "Business Card", "Billboard", "Phone Case"]
MOCKUP_STYLES = ["Single Product (Centered)", "Bento Knolling Layout", "Realistic Ambient Scene"]

def get_product_types():
    return SUPPORTED_PRODUCTS

def get_mockup_styles():
    return MOCKUP_STYLES

def generate_mockup(logo_image, product_type, prompt="", mockup_style="Single Product (Centered)", progress_cb=None):
    """
    Generate a dynamic product mockup using Z-Image-Turbo, then place and blend the logo.
    
    Args:
        logo_image: PIL Image or numpy array of the logo.
        product_type: String, e.g., "T-Shirt".
        prompt: Optional custom scene description prompt.
        mockup_style: String, layout style (Single, Bento, or Ambient).
        progress_cb: Callable(msg) for UI progress updates.
        
    Returns:
        (mockup_image, status_message)
    """
    if logo_image is None:
        return None, "⚠️ No logo image provided."

    # Convert to PIL Image if it's a numpy array
    if isinstance(logo_image, np.ndarray):
        logo_pil = Image.fromarray(logo_image)
    else:
        logo_pil = logo_image

    # 1. Clean the logo (ensure background is transparent)
    if progress_cb:
        progress_cb("🧹 Pre-processing logo (removing background)...")
    
    try:
        from modules.background_remover import remove_background
        logo_transparent = remove_background(logo_pil)
    except Exception as e:
        print(f"[Mockup] Background removal failed: {e}")
        logo_transparent = logo_pil.convert("RGBA")

    logo_rgba = logo_transparent.convert("RGBA")

    # 2. Build mockup scene prompt based on style
    product_name = product_type.lower()
    
    if mockup_style == "Bento Knolling Layout":
        mockup_prompt = (
            f"a premium brand identity bento mockup, flat lay knolling arrangement of coordinated "
            f"corporate branding stationery, packaging, and blank empty solid {product_name} items, "
            f"minimal soft shadows, realistic paper and product textures, clean professional presentation layout"
        )
    elif mockup_style == "Realistic Ambient Scene":
        mockup_prompt = (
            f"a high-end editorial lifestyle photography of a blank plain unbranded solid {product_name} "
            f"in a realistic authentic daily usage environment, soft natural ambient lighting, "
            f"commercial stock photo quality, photorealistic"
        )
    else:  # "Single Product (Centered)"
        mockup_prompt = (
            f"a professional high-quality product photography of a blank plain empty solid {product_name} "
            f"on a clean modern minimalist aesthetic studio background, product mockup template, "
            f"high resolution, centered composition"
        )

    # Append user custom scene cues
    if prompt.strip():
        mockup_prompt += f", {prompt.strip()}"

    neg_prompt = "logo, text, logo graphic, branding, writing, watermark, pattern, print, low quality, blurry, deformed"

    # 3. Generate the mockup scene using Z-Image-Turbo
    if progress_cb:
        progress_cb(f"🎨 Generating {mockup_style} background using AI...")

    try:
        from modules.zimage_pipeline import generate as zimage_generate

        # Generate (standard 1024x1024)
        mockup_bg, seed = zimage_generate(
            prompt=mockup_prompt,
            negative_prompt=neg_prompt,
            width=1024,
            height=1024,
            seed=-1,
            progress_callback=None
        )
    except Exception as e:
        return None, f"❌ Failed to generate mockup background scene: {str(e)}"

    # 4. Scale and composite the logo onto the generated scene
    if progress_cb:
        progress_cb("📐 Positioning and blending logo onto product...")

    try:
        mockup_rgba = mockup_bg.convert("RGBA")
        mw, mh = mockup_rgba.size

        # Determine optimal size for the logo
        # Standard: 300px max dim. Bento: 220px to look neat on smaller objects.
        max_logo_dim = 220 if mockup_style == "Bento Knolling Layout" else 300
        
        lw, lh = logo_rgba.size
        scale = min(max_logo_dim / lw, max_logo_dim / lh)
        new_w = max(16, int(lw * scale))
        new_h = max(16, int(lh * scale))
        
        logo_resized = logo_rgba.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Calculate coordinates (center or custom offset depending on product)
        paste_x = (mw - new_w) // 2
        paste_y = (mh - new_h) // 2

        if product_type == "T-Shirt":
            paste_y = int((mh - new_h) * 0.42)
        elif product_type == "Mug":
            paste_y = int((mh - new_h) * 0.48)
        elif product_type == "Phone Case":
            paste_y = int((mh - new_h) * 0.5)

        # Blend logo alpha softly to allow realistic lighting to bleed through
        r, g, b, a = logo_resized.split()
        a = a.point(lambda p: int(p * 0.88))
        logo_blended = Image.merge("RGBA", (r, g, b, a))

        # Paste logo using alpha channel as mask
        mockup_rgba.paste(logo_blended, (paste_x, paste_y), logo_blended)
        
        final_mockup = mockup_rgba.convert("RGB")
        
        if progress_cb:
            progress_cb("✅ Mockup generation complete!")
            
        return final_mockup, f"✅ Mockup generated successfully! (seed: {seed})"

    except Exception as e:
        return None, f"❌ Compositing failed: {str(e)}"
