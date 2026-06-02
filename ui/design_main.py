"""
Design Main Tab — Primary generation interface.
Category dropdown, prompt input, palette controls, toggles, and gallery output.
"""
import gradio as gr
import os
import time
import random

from modules.design_categories import get_category_names, get_category, get_default_transparent, get_default_aspect_ratio, get_category_icon
from modules.auto_prompt_enhancer import enhance_prompt, build_negative_prompt
from modules.palette_control import inject_palette_prompt, apply_palette_post
from modules import config


def _save_image(image, output_dir, fmt='png'):
    """Save a PIL image and return the file path."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time() * 1000)
    rand = random.randint(1000, 9999)
    filename = f"design_{timestamp}_{rand}.{fmt}"
    filepath = os.path.join(output_dir, filename)
    image.save(filepath, quality=95 if fmt == 'jpeg' else None)
    return filepath


def _generate(category, prompt, negative_prompt, color1, color2, color3, color4, color5,
              use_master_neg, use_enhancement, remove_bg, vector_mode, concept_grid, aspect_ratio, seed_val, model_name):
    """Core generation function wired to the Generate button."""

    if not prompt.strip() and not category:
        yield "⚠️ Please enter a prompt.", None, []
        return

    # Build final prompts
    final_prompt = prompt
    if use_enhancement and category:
        final_prompt = enhance_prompt(prompt, category, use_enhancement=True)
    
    # Inject concept grid layout instruction if enabled
    if concept_grid:
        grid_instruction = "arranged in a clean 2x2 grid layout, four distinct minimalist concepts, flat design, white dividing lines, high contrast"
        final_prompt = f"{final_prompt}, {grid_instruction}" if final_prompt.strip() else grid_instruction
    
    # Inject color palette
    colors = [c for c in [color1, color2, color3, color4, color5] if c and c != '#000000']
    if colors:
        final_prompt = inject_palette_prompt(final_prompt, colors)

    final_negative = build_negative_prompt(negative_prompt, category, use_master_neg)

    # Parse aspect ratio
    width, height = config.parse_aspect_ratio(aspect_ratio)

    # Parse seed
    seed = int(seed_val) if seed_val and str(seed_val).strip() and str(seed_val).strip() != '-1' else -1

    yield "🔄 Loading pipeline...", None, []

    if vector_mode:
        # SVG generation path
        yield "🔄 Loading StarVector (first use may download model)...", None, []
        try:
            from modules.starvector_pipeline import image_to_svg, load_model
            from modules.zimage_pipeline import generate as zimage_generate

            # First generate raster, then vectorize
            yield f"🎨 Generating raster base image using {model_name}...", None, []
            image, used_seed = zimage_generate(
                prompt=final_prompt, negative_prompt=final_negative,
                width=width, height=height, seed=seed, model_name=model_name
            )

            yield "✏️ Vectorizing to SVG...", None, []
            svg_code = image_to_svg(image)

            # Save SVG
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            svg_path = os.path.join(config.OUTPUT_DIR, f"design_{int(time.time()*1000)}.svg")
            with open(svg_path, 'w') as f:
                f.write(svg_code)

            # Also save raster preview
            raster_path = _save_image(image, config.OUTPUT_DIR)

            yield f"✅ Done! Seed: {used_seed} | SVG saved: {os.path.basename(svg_path)}", image, [raster_path]

        except Exception as e:
            yield f"❌ Vector generation failed: {str(e)}", None, []
    else:
        # Standard raster generation
        try:
            from modules.zimage_pipeline import generate as zimage_generate

            yield f"🎨 Generating image using {model_name}...", None, []
            image, used_seed = zimage_generate(
                prompt=final_prompt, negative_prompt=final_negative,
                width=width, height=height, seed=seed, model_name=model_name
            )

            # Apply palette post-processing
            if colors:
                image = apply_palette_post(image, colors, strength=0.3)

            # Background removal
            if remove_bg:
                yield "🧹 Removing background...", None, []
                from modules.background_remover import remove_background
                image = remove_background(image)

            # Save
            filepath = _save_image(image, config.OUTPUT_DIR)
            yield f"✅ Done! Seed: {used_seed}", image, [filepath]

        except Exception as e:
            yield f"❌ Generation failed: {str(e)}", None, []


def _on_category_change(category):
    """Update UI defaults when category changes."""
    if not category:
        return gr.update(), gr.update(), gr.update()
    
    transparent = get_default_transparent(category)
    ar = get_default_aspect_ratio(category)
    icon = get_category_icon(category)
    
    # Find matching aspect ratio label
    ar_labels = config.get_aspect_ratio_labels()
    w, h = ar.split('*')
    target = f'{w}×{h}'
    selected_ar = ar_labels[0]
    for label in ar_labels:
        if target in label:
            selected_ar = label
            break

    return gr.update(value=transparent), gr.update(value=selected_ar), gr.update()


def build_tab():
    """Build and return the main Generate tab components."""
    category_names = get_category_names()
    # Filter out Edit and Variation (they have their own tabs)
    generate_categories = [c for c in category_names if c not in ['Edit Design/Photo', 'Variation']]

    with gr.Row():
        # LEFT PANEL - Controls
        with gr.Column(scale=2):
            category = gr.Dropdown(
                label='🎯 Design Category',
                choices=generate_categories,
                value=generate_categories[0] if generate_categories else None,
                interactive=True,
                elem_id='category_dropdown'
            )
            model_choice = gr.Dropdown(
                label="🤖 AI Model",
                choices=["FLUX.1-schnell", "Z-Image-Turbo"],
                value="FLUX.1-schnell",
                interactive=True,
                elem_id="model_dropdown"
            )
            prompt = gr.Textbox(
                label='✨ Prompt',
                placeholder='Describe your design...',
                lines=3,
                elem_id='prompt_input'
            )
            with gr.Accordion('📝 Negative Prompt', open=False):
                negative_prompt = gr.Textbox(
                    label='Negative Prompt',
                    placeholder='What to avoid...',
                    lines=2,
                    show_label=False
                )
            
            with gr.Accordion('🎨 Color Palette', open=False):
                with gr.Row():
                    color1 = gr.ColorPicker(label='Color 1', value='#000000')
                    color2 = gr.ColorPicker(label='Color 2', value='#000000')
                    color3 = gr.ColorPicker(label='Color 3', value='#000000')
                    color4 = gr.ColorPicker(label='Color 4', value='#000000')
                    color5 = gr.ColorPicker(label='Color 5', value='#000000')
 
            with gr.Row():
                use_master_neg = gr.Checkbox(label='Master negative prompt', value=True)
                use_enhancement = gr.Checkbox(label='Category enhancement', value=True)
            with gr.Row():
                remove_bg = gr.Checkbox(label='Remove background (transparent PNG)', value=True, elem_id='remove_bg')
                vector_mode = gr.Checkbox(label='Vector mode (SVG)', value=False)
                concept_grid = gr.Checkbox(label='2x2 Concept Grid', value=False)
 
            aspect_ratio = gr.Dropdown(
                label='📐 Aspect Ratio',
                choices=config.get_aspect_ratio_labels(),
                value=config.get_aspect_ratio_labels()[9],  # 1024x1024
                interactive=True
            )
            seed_val = gr.Textbox(label='🎲 Seed (-1 = random)', value='-1', max_lines=1)
 
            generate_btn = gr.Button('🚀 Generate', variant='primary', elem_id='generate_btn')
 
        # RIGHT PANEL - Output
        with gr.Column(scale=3):
            status = gr.Textbox(label='Status', interactive=False, elem_id='status_display')
            preview = gr.Image(label='Preview', type='pil', interactive=False, height=512)
            gallery = gr.Gallery(label='Generated Images', columns=4, height=300,
                                 object_fit='contain', elem_id='output_gallery')
 
    # Wire events
    category.change(_on_category_change, inputs=[category],
                     outputs=[remove_bg, aspect_ratio, status])
 
    generate_btn.click(
        _generate,
        inputs=[category, prompt, negative_prompt,
                color1, color2, color3, color4, color5,
                use_master_neg, use_enhancement, remove_bg, vector_mode, concept_grid,
                aspect_ratio, seed_val, model_choice],
        outputs=[status, preview, gallery]
    )
 
    return category, prompt, negative_prompt, generate_btn, gallery
