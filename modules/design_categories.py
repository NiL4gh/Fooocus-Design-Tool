"""
Design Categories Module
Loads and manages design category configurations from config/design_categories.json.
Each category defines master negative prompts, prompt enhancements, transparency defaults,
and default aspect ratios.
"""

import json
import os

_categories = None
_categories_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'design_categories.json')


def load_categories():
    """Load design categories from JSON config. Cached after first load."""
    global _categories
    if _categories is not None:
        return _categories

    try:
        with open(_categories_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            _categories = data.get('categories', [])
    except Exception as e:
        print(f'[Design Categories] Failed to load categories: {e}')
        _categories = []

    return _categories


def get_category_names():
    """Return list of category names for UI dropdown."""
    categories = load_categories()
    return [cat['name'] for cat in categories]


def get_category(name):
    """Get a category config dict by name."""
    categories = load_categories()
    for cat in categories:
        if cat['name'] == name:
            return cat
    return None


def get_master_negative(name):
    """Get the master negative prompt for a category."""
    cat = get_category(name)
    if cat:
        return cat.get('master_negative', '')
    return ''


def get_enhancement_template(name):
    """Get the prompt enhancement string for a category."""
    cat = get_category(name)
    if cat:
        return cat.get('prompt_enhancement', '')
    return ''


def get_default_transparent(name):
    """Check if a category defaults to transparent background."""
    cat = get_category(name)
    if cat:
        return cat.get('default_transparent', False)
    return False


def get_default_aspect_ratio(name):
    """Get the default aspect ratio for a category."""
    cat = get_category(name)
    if cat:
        return cat.get('default_aspect_ratio', '1024*1024')
    return '1024*1024'


def get_category_icon(name):
    """Get the emoji icon for a category."""
    cat = get_category(name)
    if cat:
        return cat.get('icon', '🎨')
    return '🎨'
