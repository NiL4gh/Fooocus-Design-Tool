"""
Variation Generator Module
Produces 2-4 similar outputs from a given prompt by mixing seeds.
"""

from modules.sdxl_pipeline import generate, generate_variations as sdxl_variations


def generate_variations(prompt, negative_prompt="", width=1024, height=1024,
                         base_seed=None, count=4, progress_cb=None, speed_mode="fast"):
    """Generate variations using seed mixing via sdxl_pipeline."""
    return sdxl_variations(
        prompt=prompt, negative_prompt=negative_prompt,
        width=width, height=height, base_seed=base_seed,
        count=count, speed_mode=speed_mode, progress_callback=progress_cb
    )
