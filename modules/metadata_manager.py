"""
Metadata Manager Module — Fooocus Designer 2.0
Embeds structured JSON metadata into PNG tEXt chunks and extracts parameters
for single-click recreation and UI recovery.
"""
import json
import os
from typing import Dict, Any, Optional, Union, List
from PIL import Image
from PIL.PngImagePlugin import PngInfo

METADATA_KEY = "fooocus_designer_metadata"
PARAMETERS_KEY = "parameters"


def build_metadata(
    category: str,
    prompt: str,
    negative_prompt: str = "",
    seed: int = -1,
    speed_mode: str = "fast",
    selected_styles: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    width: int = 1024,
    height: int = 1024,
    loras: Optional[List[Dict[str, Any]]] = None,
    version: str = "2.0",
) -> Dict[str, Any]:
    """Construct a clean, serializable metadata dictionary."""
    return {
        "version": version,
        "category": category or "",
        "prompt": prompt or "",
        "negative_prompt": negative_prompt or "",
        "seed": seed,
        "speed_mode": speed_mode or "fast",
        "selected_styles": list(selected_styles or []),
        "colors": list(colors or []),
        "dimensions": [width, height],
        "loras": list(loras or []),
        "model": "RunDiffusion/Juggernaut-XL-v9",
    }


def save_image_with_metadata(
    image: Image.Image,
    filepath: str,
    metadata: Dict[str, Any],
    fmt: str = "png"
) -> str:
    """
    Save a PIL Image with embedded PNG metadata.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    if fmt.lower() == "png":
        png_info = PngInfo()
        json_str = json.dumps(metadata, ensure_ascii=False)
        png_info.add_text(METADATA_KEY, json_str)
        # Add standard parameters text for interoperability
        png_info.add_text(
            PARAMETERS_KEY,
            f"{metadata.get('prompt', '')}\n"
            f"Negative prompt: {metadata.get('negative_prompt', '')}\n"
            f"Steps: {6 if metadata.get('speed_mode') == 'fast' else 28}, "
            f"Seed: {metadata.get('seed', -1)}, "
            f"Model: {metadata.get('model', 'Juggernaut-XL')}"
        )
        image.save(filepath, format="PNG", pnginfo=png_info)
    else:
        save_img = image.convert("RGB") if (fmt.upper() in ("JPEG", "JPG") and image.mode in ("RGBA", "P", "LA")) else image
        save_img.save(filepath, format=fmt.upper(), quality=95)

    return filepath


def extract_metadata(image_or_path: Union[Image.Image, str]) -> Optional[Dict[str, Any]]:
    """
    Extract embedded Fooocus Designer metadata from a PIL Image or image file path.
    Returns parsed dictionary or None if absent.
    """
    try:
        if isinstance(image_or_path, str):
            if not os.path.exists(image_or_path):
                return None
            with Image.open(image_or_path) as img:
                info = img.info
                return _parse_info_dict(info)
        elif isinstance(image_or_path, Image.Image):
            return _parse_info_dict(image_or_path.info)
        return None
    except Exception:
        return None


def _parse_info_dict(info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Helper to extract and parse metadata from an image info dictionary."""
    if not info:
        return None
    
    # 1. Primary Fooocus Designer metadata JSON
    if METADATA_KEY in info:
        try:
            val = info[METADATA_KEY]
            if isinstance(val, bytes):
                val = val.decode("utf-8", errors="ignore")
            return json.loads(val)
        except Exception:
            pass

    return None


def format_metadata_for_display(metadata: Dict[str, Any]) -> str:
    """Format metadata into a human-readable summary string for the UI."""
    if not metadata:
        return "No metadata found."
    lines = [
        f"🎯 Category: {metadata.get('category', 'None')}",
        f"⚡ Mode: {metadata.get('speed_mode', 'fast').upper()}",
        f"✨ Prompt: {metadata.get('prompt', '')}",
        f"🎲 Seed: {metadata.get('seed', -1)}",
    ]
    styles = metadata.get("selected_styles", [])
    if styles:
        lines.append(f"🎨 Styles: {', '.join(styles)}")
    colors = metadata.get("colors", [])
    if colors:
        lines.append(f"🎨 Colors: {', '.join(colors)}")
    return "\n".join(lines)
