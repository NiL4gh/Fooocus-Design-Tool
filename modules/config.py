"""
Design Tool Configuration
Minimal config for the Fooocus Design Tool.
"""
import os
import json

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, 'config', 'design_categories.json')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Generation defaults
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_STEPS = 4
DEFAULT_GUIDANCE_SCALE = 0.0
DEFAULT_MAX_IMAGE_NUMBER = 1
DEFAULT_OUTPUT_FORMAT = 'png'

# Feature flags
ENABLE_TRANSPARENT_BACKGROUND = True
ENABLE_VECTOR_MODE = False

# Aspect ratios available
ASPECT_RATIOS = [
    '704*1408', '768*1344', '768*1280', '832*1216', '832*1152',
    '896*1152', '896*1088', '960*1088', '960*1024', '1024*1024',
    '1024*960', '1088*960', '1088*896', '1152*896', '1152*832',
    '1216*832', '1280*768', '1344*768', '1344*704', '1408*704',
    '1536*640', '1600*640',
]

def get_aspect_ratio_labels():
    """Generate display labels for aspect ratios."""
    import math
    labels = []
    for ar in ASPECT_RATIOS:
        w, h = ar.split('*')
        w, h = int(w), int(h)
        g = math.gcd(w, h)
        labels.append(f'{w}×{h} ({w//g}:{h//g})')
    return labels

def parse_aspect_ratio(label):
    """Extract width and height from aspect ratio label or raw string."""
    # Handle both '1024*1024' and '1024×1024 (1:1)' formats
    for ar in ASPECT_RATIOS:
        w, h = ar.split('*')
        if f'{w}×{h}' in label or ar == label:
            return int(w), int(h)
    return DEFAULT_WIDTH, DEFAULT_HEIGHT
