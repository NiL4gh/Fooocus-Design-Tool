"""
Auto Prompt Enhancer Module
Automatically enhances user prompts with category-specific keywords
before sending to the generation pipeline.
"""

from modules.design_categories import get_enhancement_template, get_master_negative


def enhance_prompt(prompt, category_name, use_enhancement=True):
    """
    Enhance a user prompt with category-specific style keywords.

    Args:
        prompt: The user's raw prompt text.
        category_name: The selected design category name.
        use_enhancement: Whether to apply enhancement (user toggle).

    Returns:
        Enhanced prompt string.
    """
    if not use_enhancement or not category_name:
        return prompt

    template = get_enhancement_template(category_name)
    if not template:
        return prompt

    # Prepend the enhancement template before the user prompt
    # This ensures style keywords take priority in CLIP encoding
    enhanced = f"{template}, {prompt}" if prompt.strip() else template
    return enhanced


def build_negative_prompt(user_negative, category_name, use_master_negative=True):
    """
    Build the final negative prompt by combining user input with master negative.

    Args:
        user_negative: The user's custom negative prompt.
        category_name: The selected design category name.
        use_master_negative: Whether to append master negative (user toggle).

    Returns:
        Combined negative prompt string.
    """
    if not use_master_negative or not category_name:
        return user_negative

    master = get_master_negative(category_name)
    if not master:
        return user_negative

    if user_negative.strip():
        return f"{user_negative}, {master}"
    else:
        return master
