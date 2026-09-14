"""
Auto Prompt Enhancer Module — Fooocus Designer 2.0
Automatically enhances user prompts with category-specific microstock keywords
and layers Fooocus style templates ({prompt} interpolation) into generation.
"""

from typing import List, Optional
from modules.design_categories import get_enhancement_template, get_master_negative
from modules.style_engine import apply_styles

VECTOR_SILHOUETTE_CATEGORIES = {
    "adobe stock silhouette",
    "adobe stock flat vector",
    "adobe stock sticker/clipart",
    "vector silhouette",
    "logo",
}


def is_vector_silhouette_category(category_name: str) -> bool:
    """Return whether category is a clean vector, silhouette, sticker, or logo."""
    if not category_name:
        return False
    return category_name.strip().lower() in VECTOR_SILHOUETTE_CATEGORIES


def enhance_prompt(
    prompt: str,
    category_name: str,
    use_enhancement: bool = True,
    selected_styles: Optional[List[str]] = None,
) -> str:
    """
    Enhance a user prompt with category-specific style keywords and Fooocus styles.

    Args:
        prompt: Raw user prompt text.
        category_name: The selected design category name.
        use_enhancement: Whether to apply category enhancement.
        selected_styles: Optional list of Fooocus style template names.

    Returns:
        Enhanced prompt string.
    """
    current_prompt = (prompt or "").strip().rstrip(",").strip()

    if use_enhancement and category_name:
        template = get_enhancement_template(category_name)
        if template:
            # Prepend category enhancement template to anchor CLIP conditioning
            current_prompt = f"{template}, {current_prompt}" if current_prompt else template

    # Apply selected Fooocus styles if requested
    if selected_styles:
        current_prompt, _ = apply_styles(current_prompt, "", selected_styles)

    return current_prompt


def build_negative_prompt(
    user_negative: Optional[str] = None,
    category_name: str = "",
    use_master_negative: bool = True,
    selected_styles: Optional[List[str]] = None,
) -> str:
    """
    Build the final negative prompt by combining user negative, category master negative,
    and style negative prompts.

    Args:
        user_negative: User's custom negative prompt.
        category_name: The selected design category name.
        use_master_negative: Whether to append category master negative.
        selected_styles: Optional list of Fooocus style template names.

    Returns:
        Combined negative prompt string.
    """
    user_neg_clean = (user_negative or "").strip().rstrip(",").strip()
    negative_parts = [user_neg_clean] if user_neg_clean else []

    if use_master_negative and category_name:
        master = get_master_negative(category_name)
        if master and master not in negative_parts:
            negative_parts.append(master.strip().rstrip(",").strip())

    combined_neg = ", ".join(negative_parts).strip(", ")

    # Layer negative terms from selected styles
    if selected_styles:
        _, combined_neg = apply_styles("", combined_neg, selected_styles)

    return combined_neg
