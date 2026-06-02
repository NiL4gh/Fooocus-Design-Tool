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

def cylinder_warp(img, angle_deg=60):
    """
    Warp an image around a cylinder using numpy bilinear interpolation.
    Simulates vertical curvature to match camera perspective of a mug print.
    """
    arr = np.array(img)
    h, w, c = arr.shape
    
    # Create grid of coordinates
    y_coords, x_coords = np.indices((h, w), dtype=np.float32)
    
    # Normalize X to [-1, 1]
    x_norm = (x_coords / (w - 1)) * 2.0 - 1.0
    
    # Cylinder mapping
    angle_rad = np.radians(angle_deg)
    theta = x_norm * angle_rad
    
    # Map back to flat plane source coordinates
    x_src_norm = np.sin(theta) / np.sin(angle_rad)
    x_src = ((x_src_norm + 1.0) / 2.0) * (w - 1)
    
    # Vertical curvature mapping (adds depth to top/bottom curves)
    curvature = h * 0.08
    y_src = y_coords - curvature * (1.0 - np.cos(theta))
    
    # Clip coordinates to bounds
    x_src = np.clip(x_src, 0, w - 1)
    y_src = np.clip(y_src, 0, h - 1)
    
    # Perform bilinear interpolation
    x0 = np.floor(x_src).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y0 = np.floor(y_src).astype(np.int32)
    y1 = np.minimum(y0 + 1, h - 1)
    
    wa = (x1 - x_src) * (y1 - y_src)
    wb = (x_src - x0) * (y1 - y_src)
    wc = (x1 - x_src) * (y_src - y0)
    wd = (x_src - x0) * (y_src - y0)
    
    # Interpolate channels
    warped_arr = np.zeros_like(arr)
    for channel in range(c):
        warped_arr[..., channel] = (
            wa * arr[y0, x0, channel] +
            wb * arr[y0, x1, channel] +
            wc * arr[y1, x0, channel] +
            wd * arr[y1, x1, channel]
        )
        
    # Mask out pixels that go off the cylinder edges
    mask = (x_src_norm >= -1.0) & (x_src_norm <= 1.0)
    warped_arr[~mask, 3] = 0  # set alpha to 0 for out of bounds
    
    return Image.fromarray(warped_arr)


def apply_shading_blend(logo_rgb, bg_crop_rgb):
    """
    Blend background textures, creases, and shadows onto the logo using a premium Soft-Light filter.
    """
    from PIL import ImageEnhance
    
    # 1. Convert bg crop to grayscale and enhance contrast of creases/shadows
    bg_gray = bg_crop_rgb.convert("L")
    enhancer = ImageEnhance.Contrast(bg_gray)
    bg_gray_enhanced = enhancer.enhance(1.4)
    
    # 2. Convert to numpy arrays
    logo_arr = np.array(logo_rgb, dtype=np.float32)
    bg_arr = np.array(bg_gray_enhanced, dtype=np.float32)
    
    # 3. Normalize shading map to center around neutral gray (128)
    mean_val = np.mean(bg_arr)
    bg_norm = (bg_arr - mean_val) * 0.7 + 128.0
    bg_norm = np.clip(bg_norm, 0.0, 255.0) / 255.0
    
    # Expand dims for broadcasting over RGB channels
    shading_map = np.expand_dims(bg_norm, axis=2)
    
    # 4. Apply standard W3C/Photoshop Soft-Light blending formula
    logo_norm = logo_arr / 255.0
    soft_light = (1.0 - 2.0 * shading_map) * (logo_norm ** 2) + 2.0 * shading_map * logo_norm
    soft_light = np.clip(soft_light * 255.0, 0, 255).astype(np.uint8)
    
    return Image.fromarray(soft_light)


def generate_mockup(logo_image, product_type, prompt="", mockup_style="Single Product (Centered)", progress_cb=None, model_name="FLUX.1-schnell"):
    """
    Generate a dynamic product mockup using Z-Image-Turbo or FLUX.1-schnell, then place and blend the logo.
    
    Args:
        logo_image: PIL Image or numpy array of the logo.
        product_type: String, e.g., "T-Shirt".
        prompt: Optional custom scene description prompt.
        mockup_style: String, layout style (Single, Bento, or Ambient).
        progress_cb: Callable(msg) for UI progress updates.
        model_name: The name of the model to use.
        
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
            progress_callback=None,
            model_name=model_name
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

        # 4a. Apply advanced 3D perspective warp for cylindrical objects (Mugs)
        if product_type == "Mug":
            if progress_cb:
                progress_cb("🌀 Wrapping logo around cylindrical cup curvature...")
            logo_resized = cylinder_warp(logo_resized, angle_deg=65)

        # Calculate coordinates (center or custom offset depending on product)
        paste_x = (mw - new_w) // 2
        paste_y = (mh - new_h) // 2

        if product_type == "T-Shirt":
            paste_y = int((mh - new_h) * 0.42)
        elif product_type == "Mug":
            paste_y = int((mh - new_h) * 0.48)
        elif product_type == "Phone Case":
            paste_y = int((mh - new_h) * 0.5)

        # 4b. Extract product shading overlay and blend using Soft-Light filter
        if progress_cb:
            progress_cb("🧵 Blending fabric folds and textures into logo...")
        bg_crop = mockup_rgba.crop((paste_x, paste_y, paste_x + new_w, paste_y + new_h)).convert("RGB")
        
        # Split logo into RGB and Alpha channels
        lr, lg, lb, la = logo_resized.split()
        logo_rgb = Image.merge("RGB", (lr, lg, lb))
        
        # Apply premium Soft-Light texture overlay
        logo_shaded_rgb = apply_shading_blend(logo_rgb, bg_crop)
        
        # Merge back with customized alpha (slightly translucent for print realism)
        la_blended = la.point(lambda p: int(p * 0.90))
        logo_blended = Image.merge("RGBA", (logo_shaded_rgb.split()[0], logo_shaded_rgb.split()[1], logo_shaded_rgb.split()[2], la_blended))

        # Paste logo using alpha channel as mask
        mockup_rgba.paste(logo_blended, (paste_x, paste_y), logo_blended)
        
        final_mockup = mockup_rgba.convert("RGB")
        
        if progress_cb:
            progress_cb("✅ Mockup generation complete!")
            
        return final_mockup, f"✅ Mockup generated successfully! (seed: {seed})"

    except Exception as e:
        return None, f"❌ Compositing failed: {str(e)}"
