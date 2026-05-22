"""
Variation Generator Module
Produces 2-4 similar outputs from a given prompt by mixing seeds.
"""

def generate_variations(prompt, negative_prompt="", width=1024, height=1024,
                         base_seed=None, count=4, progress_cb=None):
    """Generate variations using seed mixing via zimage_pipeline."""
    from modules.zimage_pipeline import generate_variations as zimage_variations
    return zimage_variations(
        prompt=prompt, negative_prompt=negative_prompt,
        width=width, height=height, base_seed=base_seed,
        count=count, progress_callback=progress_cb
    )
