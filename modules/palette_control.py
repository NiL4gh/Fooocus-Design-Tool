"""
Palette Control Module
Handles color palette injection into prompts and optional post-generation color mapping.
"""

import re
import numpy as np
from PIL import Image


# Common color name mappings for natural language injection
COLOR_NAMES = {
    '#FF0000': 'red', '#00FF00': 'green', '#0000FF': 'blue',
    '#FFFF00': 'yellow', '#FF00FF': 'magenta', '#00FFFF': 'cyan',
    '#FFFFFF': 'white', '#000000': 'black', '#FFA500': 'orange',
    '#800080': 'purple', '#FFC0CB': 'pink', '#A52A2A': 'brown',
    '#808080': 'gray', '#FFD700': 'gold', '#C0C0C0': 'silver',
    '#F5F5DC': 'beige', '#008080': 'teal', '#000080': 'navy',
    '#FF6347': 'tomato red', '#4B0082': 'indigo',
}


def hex_to_color_name(hex_color):
    """Convert hex color to nearest common name, or return hex description."""
    hex_upper = hex_color.upper().strip()
    if hex_upper in COLOR_NAMES:
        return COLOR_NAMES[hex_upper]

    # Parse RGB and describe
    hex_clean = hex_upper.lstrip('#')
    if len(hex_clean) == 6:
        r, g, b = int(hex_clean[:2], 16), int(hex_clean[2:4], 16), int(hex_clean[4:6], 16)
        # Find nearest named color
        min_dist = float('inf')
        nearest_name = hex_color
        for ref_hex, name in COLOR_NAMES.items():
            rr, gg, bb = int(ref_hex[1:3], 16), int(ref_hex[3:5], 16), int(ref_hex[5:7], 16)
            dist = (r - rr) ** 2 + (g - gg) ** 2 + (b - bb) ** 2
            if dist < min_dist:
                min_dist = dist
                nearest_name = name
        if min_dist < 5000:
            return f"similar to {nearest_name}"
        return f"color {hex_color}"
    return hex_color


def inject_palette_prompt(prompt, hex_colors):
    """
    Add color palette constraint text to a prompt.

    Args:
        prompt: The user's prompt text.
        hex_colors: List of hex color strings (e.g., ['#FF5733', '#33FF57']).

    Returns:
        Prompt with color instructions appended.
    """
    # Filter empty / invalid colors
    valid_colors = [c.strip() for c in hex_colors if c and re.match(r'^#[0-9A-Fa-f]{6}$', c.strip())]
    if not valid_colors:
        return prompt

    color_descriptions = [hex_to_color_name(c) for c in valid_colors]
    color_text = ', '.join(color_descriptions)
    hex_text = ', '.join(valid_colors)

    palette_instruction = f"using a color palette of {color_text} ({hex_text})"
    return f"{prompt}, {palette_instruction}" if prompt.strip() else palette_instruction


def apply_palette_post(image, hex_colors, strength=0.5):
    """
    Optional post-processing: shift image colors toward the target palette.
    Uses a lightweight approach - adjusts color balance without full recoloring.

    Args:
        image: PIL Image object.
        hex_colors: List of hex color strings.
        strength: How strongly to apply the color shift (0.0 to 1.0).

    Returns:
        PIL Image with adjusted colors.
    """
    if not hex_colors or strength <= 0:
        return image

    valid_colors = [c.strip() for c in hex_colors if c and re.match(r'^#[0-9A-Fa-f]{6}$', c.strip())]
    if not valid_colors:
        return image

    # Convert image to numpy
    img_array = np.array(image).astype(np.float32)
    has_alpha = img_array.shape[2] == 4 if len(img_array.shape) == 3 else False

    # Parse target palette
    palette = []
    for hex_color in valid_colors:
        hex_clean = hex_color.lstrip('#')
        palette.append([int(hex_clean[:2], 16), int(hex_clean[2:4], 16), int(hex_clean[4:6], 16)])
    palette = np.array(palette, dtype=np.float32)

    # Compute the average palette color for a subtle tint
    avg_palette = palette.mean(axis=0)
    avg_image = img_array[:, :, :3].mean(axis=(0, 1))

    # Shift the image color balance subtly toward the palette average
    shift = (avg_palette - avg_image) * strength * 0.3
    img_array[:, :, :3] = np.clip(img_array[:, :, :3] + shift, 0, 255)

    return Image.fromarray(img_array.astype(np.uint8))
