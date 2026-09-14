"""
Design Main Tab — Primary generation interface.
Category dropdown, prompt input, palette controls, toggles, and gallery output.
"""
import gradio as gr
import os
import time
import random

from modules.design_categories import get_category_names, get_category, get_default_transparent, get_default_aspect_ratio
from modules.auto_prompt_enhancer import enhance_prompt, build_negative_prompt
from modules.palette_control import inject_palette_prompt, apply_palette_post, get_palette_presets, get_preset_colors
from modules.metadata_manager import build_metadata, save_image_with_metadata, extract_metadata
from modules.sdxl_pipeline import generate as sdxl_generate, get_base_model_choices, resolve_base_model_id
from modules.style_engine import get_available_styles
from modules import config


def _get_lora_status(category: str) -> str:
    """Return formatted status markdown describing the active LoRA for the selected category."""
    if not category:
        return "🏷️ **Active LoRA:** None"
    cfg = get_category(category)
    if cfg and cfg.get("lora"):
        lora_info = cfg.get("lora")
        source = lora_info.get("source", "")
        name = lora_info.get("name") or (source.split("/")[-1] if "/" in source else source) or "Custom LoRA"
        weight = lora_info.get("weight", 0.8)
        triggers = lora_info.get("trigger_words") or ", ".join(cfg.get("trigger_words", [])) or "None"
        return f"🏷️ **Active Baked LoRA:** `{name}` (Weight: `{weight:.2f}`) | **Triggers:** `{triggers}`"
    return "🏷️ **Active Baked LoRA:** None (Direct SDXL base model with trigger conditioning)"


def _save_image(image, output_dir, fmt='png', metadata=None):
    """Save a PIL image with optional embedded metadata and return the file path."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time() * 1000)
    rand = random.randint(1000, 9999)
    filename = f"design_{timestamp}_{rand}.{fmt}"
    filepath = os.path.join(output_dir, filename)
    if metadata is not None:
        save_image_with_metadata(image, filepath, metadata, fmt=fmt)
    else:
        image.save(filepath, quality=95 if fmt == 'jpeg' else None)
    return filepath


def _generate(category, prompt, negative_prompt, color1, color2, color3, color4, color5,
              use_master_neg, use_enhancement, remove_bg, vector_mode, concept_grid, aspect_ratio, seed_val, speed_mode_label,
              selected_styles=None, base_model=None):
    """Core generation function wired to the Generate button."""

    if not (prompt or "").strip() and not category:
        yield "⚠️ Please enter a prompt.", None, []
        return

    # Parse speed mode and category config
    speed_mode = "fast" if "fast" in str(speed_mode_label).lower() else "master"
    cat_cfg = get_category(category) if category else None

    # Build final prompts
    final_prompt = enhance_prompt(prompt, category, use_enhancement=use_enhancement, selected_styles=selected_styles)
    
    # Inject concept grid layout instruction if enabled
    if concept_grid:
        grid_instruction = "arranged in a clean 2x2 grid layout, four distinct minimalist concepts, flat design, white dividing lines, high contrast"
        final_prompt = f"{final_prompt}, {grid_instruction}" if (final_prompt or "").strip() else grid_instruction
    
    # Inject color palette
    colors = [c for c in [color1, color2, color3, color4, color5] if c and c != '#000000']
    if colors:
        final_prompt = inject_palette_prompt(final_prompt, colors)

    final_negative = build_negative_prompt(negative_prompt, category, use_master_negative=use_master_neg, selected_styles=selected_styles)

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

            model_label = base_model.split("(")[0].strip() if base_model else "Juggernaut XL v9"
            yield f"🎨 Generating raster base image using {model_label}...", None, []
            image, used_seed = sdxl_generate(
                prompt=final_prompt,
                negative_prompt=final_negative,
                width=width,
                height=height,
                seed=seed,
                speed_mode=speed_mode,
                category_cfg=cat_cfg,
                base_model=base_model,
            )

            yield "✏️ Vectorizing to SVG...", None, []
            svg_code = image_to_svg(image)

            # Save SVG
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            svg_path = os.path.join(config.OUTPUT_DIR, f"design_{int(time.time()*1000)}.svg")
            with open(svg_path, 'w') as f:
                f.write(svg_code)

            # Construct metadata for raster preview
            metadata = build_metadata(
                category=category,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=used_seed,
                speed_mode=speed_mode,
                selected_styles=selected_styles,
                colors=colors,
                width=width,
                height=height,
                loras=cat_cfg.get("loras") if cat_cfg else None,
            )

            # Also save raster preview
            raster_path = _save_image(image, config.OUTPUT_DIR, metadata=metadata)

            yield f"✅ Done! Seed: {used_seed} | SVG saved: {os.path.basename(svg_path)}", image, [raster_path]

        except Exception as e:
            yield f"❌ Vector generation failed: {str(e)}", None, []
    else:
        # Standard raster generation
        try:
            model_label = base_model.split("(")[0].strip() if base_model else "Juggernaut XL v9"
            yield f"🎨 Generating with {model_label} ({speed_mode} mode)...", None, []
            image, used_seed = sdxl_generate(
                prompt=final_prompt,
                negative_prompt=final_negative,
                width=width,
                height=height,
                seed=seed,
                speed_mode=speed_mode,
                category_cfg=cat_cfg,
                base_model=base_model,
            )

            # Apply palette post-processing
            if colors:
                image = apply_palette_post(image, colors, strength=0.3)

            # Background removal
            if remove_bg:
                yield "🧹 Removing background...", None, []
                from modules.background_remover import remove_background
                image = remove_background(image)

            # Construct metadata
            metadata = build_metadata(
                category=category,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=used_seed,
                speed_mode=speed_mode,
                selected_styles=selected_styles,
                colors=colors,
                width=width,
                height=height,
                loras=cat_cfg.get("loras") if cat_cfg else None,
            )

            # Save
            filepath = _save_image(image, config.OUTPUT_DIR, metadata=metadata)
            yield f"✅ Done! Seed: {used_seed}", image, [filepath]

        except Exception as e:
            yield f"❌ Generation failed: {str(e)}", None, []


def _on_palette_preset_change(preset_name):
    """Update 5 color pickers when a curated palette preset is selected."""
    colors = get_preset_colors(preset_name)
    return (
        gr.update(value=colors[0]),
        gr.update(value=colors[1]),
        gr.update(value=colors[2]),
        gr.update(value=colors[3]),
        gr.update(value=colors[4]),
    )


def _on_image_drop_inspect(image_file):
    """Extract metadata from dropped image and return updates for UI components."""
    if not image_file:
        return (
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update()
        )
    
    filepath = image_file if isinstance(image_file, str) else getattr(image_file, "name", None)
    meta = extract_metadata(filepath) if filepath else extract_metadata(image_file)

    if not meta:
        return (
            "⚠️ No Fooocus Designer metadata found in this image.",
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update()
        )

    cat = meta.get("category", "")
    pr = meta.get("prompt", "")
    neg = meta.get("negative_prompt", "")
    spd = "⚡ Fast (~3s)" if meta.get("speed_mode") == "fast" else "🎯 Master (~15s)"
    styles = meta.get("selected_styles", [])
    seed = str(meta.get("seed", -1))
    colors = meta.get("colors", [])
    c1 = colors[0] if len(colors) > 0 else "#000000"
    c2 = colors[1] if len(colors) > 1 else "#000000"
    c3 = colors[2] if len(colors) > 2 else "#000000"
    c4 = colors[3] if len(colors) > 3 else "#000000"
    c5 = colors[4] if len(colors) > 4 else "#000000"

    status_msg = f"📥 Loaded settings from image! (Category: {cat or 'Custom'}, Seed: {seed})"
    return (
        status_msg,
        gr.update(value=cat) if cat else gr.update(),
        gr.update(value=pr),
        gr.update(value=neg),
        gr.update(value=spd),
        gr.update(value=styles),
        gr.update(value=seed),
        gr.update(value=c1),
        gr.update(value=c2),
        gr.update(value=c3),
        gr.update(value=c4),
        gr.update(value=c5),
    )


def _on_category_change(category):
    """Update UI defaults when category changes."""
    if not category:
        return gr.update(), gr.update(), gr.update(), gr.update()
    
    transparent = get_default_transparent(category)
    ar = get_default_aspect_ratio(category)
    default_styles = ["Fooocus V2"] if category in ["Artwork", "Poster"] else []
    lora_label = _get_lora_status(category)
    
    # Find matching aspect ratio label
    ar_labels = config.get_aspect_ratio_labels()
    w, h = ar.split('*')
    target = f'{w}×{h}'
    selected_ar = ar_labels[0]
    for label in ar_labels:
        if target in label:
            selected_ar = label
            break

    return gr.update(value=transparent), gr.update(value=selected_ar), gr.update(value=lora_label), gr.update(value=default_styles)


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
            with gr.Accordion('🤖 Model & Architecture (SDXL)', open=False):
                base_model_dropdown = gr.Dropdown(
                    label='📦 Base SDXL Checkpoint',
                    choices=get_base_model_choices(),
                    value=get_base_model_choices()[0],
                    interactive=True,
                    elem_id='base_model_dropdown',
                )
                lora_status = gr.Markdown(
                    value=_get_lora_status(generate_categories[0] if generate_categories else None),
                    elem_id='lora_status_markdown',
                )
                gr.Markdown(
                    "⚡ **Fast Mode**: ByteDance SDXL-Lightning 4-step LoRA (~3s generation, rapid concepting).\n\n"
                    "🎯 **Master Mode**: Full 28-step DPM++ 2M Karras scheduler (maximum detail & 100% LoRA fidelity)."
                )
            speed_choice = gr.Radio(
                label="⚡ Engine Mode",
                choices=["⚡ Fast (~3s)", "🎯 Master (~15s)"],
                value="⚡ Fast (~3s)",
                interactive=True,
                elem_id="speed_choice_radio",
            )
            styles_selector = gr.Dropdown(
                label="🎨 Fooocus Styles",
                choices=get_available_styles(),
                value=[],
                multiselect=True,
                interactive=True,
                elem_id="styles_dropdown",
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
                palette_preset = gr.Dropdown(
                    label="🎨 Curated Palette Preset",
                    choices=get_palette_presets(),
                    value="Custom / None",
                    interactive=True,
                    elem_id="palette_preset_dropdown"
                )
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
 
            with gr.Accordion('📥 Load Settings from Image', open=False):
                inspect_image = gr.Image(
                    label='Drop a Fooocus Designer PNG to restore prompt & settings',
                    type='filepath',
                    sources=['upload'],
                    height=160,
                    elem_id='inspect_image_input'
                )

            generate_btn = gr.Button('🚀 Generate', variant='primary', elem_id='generate_btn')
 
        # RIGHT PANEL - Output
        with gr.Column(scale=3):
            gr.Markdown(
                "🟢 **Active Engine**: SDXL Juggernaut-XL v9 | **Speed**: ⚡ Fast (~3s Lightning) | **Zero Slider Setup**",
                elem_id="engine_status_header"
            )
            status = gr.Textbox(label='Status', interactive=False, elem_id='status_display')
            preview = gr.Image(label='Preview', type='pil', interactive=False, height=512)
            gallery = gr.Gallery(label='Generated Images', columns=4, height=300,
                                 object_fit='contain', elem_id='output_gallery')
 
    # Wire events
    category.change(
        _on_category_change,
        inputs=[category],
        outputs=[remove_bg, aspect_ratio, lora_status, styles_selector],
    )
 
    palette_preset.change(
        _on_palette_preset_change,
        inputs=[palette_preset],
        outputs=[color1, color2, color3, color4, color5],
    )

    inspect_image.change(
        _on_image_drop_inspect,
        inputs=[inspect_image],
        outputs=[
            status, category, prompt, negative_prompt, speed_choice,
            styles_selector, seed_val, color1, color2, color3, color4, color5
        ],
    )

    generate_btn.click(
        _generate,
        inputs=[
            category, prompt, negative_prompt,
            color1, color2, color3, color4, color5,
            use_master_neg, use_enhancement, remove_bg, vector_mode, concept_grid,
            aspect_ratio, seed_val, speed_choice, styles_selector, base_model_dropdown
        ],
        outputs=[status, preview, gallery]
    )
 
    return category, prompt, negative_prompt, generate_btn, gallery
