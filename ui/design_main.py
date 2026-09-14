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


def _save_image(image, output_dir, fmt='png', metadata=None, stealth_mode=False):
    """Save a PIL image with optional embedded metadata and return the file path."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time() * 1000)
    rand = random.randint(1000, 9999)
    filename = f"design_{timestamp}_{rand}.{fmt}"
    filepath = os.path.join(output_dir, filename)
    if metadata is not None:
        save_image_with_metadata(image, filepath, metadata, fmt=fmt, stealth_mode=stealth_mode)
    else:
        if stealth_mode:
            from modules.metadata_manager import save_sanitized_image
            save_sanitized_image(image, filepath, fmt=fmt)
        else:
            image.save(filepath, quality=95 if fmt == 'jpeg' else None)
    return filepath


def _generate(category, prompt, negative_prompt, color1, color2, color3, color4, color5,
              use_master_neg, use_enhancement, remove_bg, vector_mode, aspect_ratio, seed_val, speed_mode_label,
              selected_styles=None, base_model=None, batch_size=1, stealth_mode=False, concept_grid=False, **kwargs):
    """Core generation function wired to the Generate button with safe batch processing."""

    if not (prompt or "").strip() and not category:
        yield "⚠️ Please enter a prompt.", None, []
        return

    # Parse speed mode and category config
    speed_mode = "fast" if "fast" in str(speed_mode_label).lower() else "master"
    cat_cfg = get_category(category) if category else None

    # Build final prompts
    final_prompt = enhance_prompt(prompt, category, use_enhancement=use_enhancement, selected_styles=selected_styles)
    
    # Inject color palette
    colors = [c for c in [color1, color2, color3, color4, color5] if c and c != '#000000']
    if colors:
        final_prompt = inject_palette_prompt(final_prompt, colors)

    final_negative = build_negative_prompt(negative_prompt, category, use_master_negative=use_master_neg, selected_styles=selected_styles)

    # Parse aspect ratio
    width, height = config.parse_aspect_ratio(aspect_ratio)

    # Parse seed
    seed = int(seed_val) if seed_val and str(seed_val).strip() and str(seed_val).strip() != '-1' else -1

    batch_count = max(1, min(32, int(batch_size or 1)))
    all_output_paths = []
    latest_img = None
    model_label = base_model.split("(")[0].strip() if base_model else "Juggernaut XL v9"

    yield f"🔄 Initializing {model_label}...", None, []

    if vector_mode:
        from modules.starvector_pipeline import image_to_svg
        for i in range(batch_count):
            cur_seed = -1 if seed == -1 else (seed + i)
            step_txt = f" ({i+1}/{batch_count})" if batch_count > 1 else ""
            yield f"🎨 Generating vector design{step_txt} with {model_label}...", latest_img, all_output_paths
            try:
                image, used_seed = sdxl_generate(
                    prompt=final_prompt,
                    negative_prompt=final_negative,
                    width=width,
                    height=height,
                    seed=cur_seed,
                    speed_mode=speed_mode,
                    category_cfg=cat_cfg,
                    base_model=base_model,
                )
                yield f"✏️ Tracing SVG vector paths{step_txt}...", image, all_output_paths
                svg_code = image_to_svg(image)
                os.makedirs(config.OUTPUT_DIR, exist_ok=True)
                timestamp = int(time.time() * 1000)
                svg_path = os.path.join(config.OUTPUT_DIR, f"design_{timestamp}_{i}.svg")
                with open(svg_path, 'w', encoding='utf-8') as f:
                    f.write(svg_code)

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
                raster_path = _save_image(image, config.OUTPUT_DIR, metadata=metadata, stealth_mode=stealth_mode)
                all_output_paths.append(raster_path)
                latest_img = image
                yield f"✅ Done! {i+1} of {batch_count} vector design(s) (Seed: {used_seed})", latest_img, all_output_paths
            except Exception as e:
                yield f"❌ Vector generation failed on item {i+1}: {str(e)}", latest_img, all_output_paths
                break
    else:
        for i in range(batch_count):
            cur_seed = -1 if seed == -1 else (seed + i)
            step_txt = f" ({i+1}/{batch_count})" if batch_count > 1 else ""
            yield f"🎨 Generating design{step_txt} with {model_label} ({speed_mode} mode)...", latest_img, all_output_paths
            try:
                image, used_seed = sdxl_generate(
                    prompt=final_prompt,
                    negative_prompt=final_negative,
                    width=width,
                    height=height,
                    seed=cur_seed,
                    speed_mode=speed_mode,
                    category_cfg=cat_cfg,
                    base_model=base_model,
                )
                if colors:
                    image = apply_palette_post(image, colors, strength=0.3)
                if remove_bg:
                    yield f"🧹 Removing background{step_txt}...", image, all_output_paths
                    from modules.background_remover import remove_background
                    image = remove_background(image)

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
                filepath = _save_image(image, config.OUTPUT_DIR, metadata=metadata, stealth_mode=stealth_mode)
                all_output_paths.append(filepath)
                latest_img = image
                yield f"✅ Done! {i+1} of {batch_count} design(s) (Seed: {used_seed})", latest_img, all_output_paths
            except Exception as e:
                yield f"❌ Generation failed on item {i+1}: {str(e)}", latest_img, all_output_paths
                break


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
                stealth_mode_checkbox = gr.Checkbox(
                    label="🛡️ AI Metadata Cleanser (Sanitize AI tags for Microstock)",
                    value=True,
                    interactive=True,
                    elem_id="stealth_metadata_checkbox",
                )
            speed_choice = gr.Radio(
                label="⚡ Engine Mode",
                choices=["🎯 Master (~15s)", "⚡ Fast (~3s)"],
                value="🎯 Master (~15s)",
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
                placeholder='Describe your design idea (e.g. japanese kimono model, coffee shop logo)...',
                lines=3,
                elem_id='prompt_input'
            )
            with gr.Row():
                ai_enhance_btn = gr.Button("🪄 ✨ Enhance Prompt with AI Copilot", size="sm", variant="secondary", elem_id="ai_enhance_btn")
            
            with gr.Accordion('🤖 AI Prompt Copilot Settings', open=False):
                from modules.ai_copilot import get_default_system_prompt
                ai_mode_choice = gr.Radio(
                    label="Copilot Engine",
                    choices=["Fast Heuristic Engine", "Local Qwen2.5-0.5B (CPU)", "Cloud API (Groq / OpenAI)"],
                    value="Fast Heuristic Engine",
                    interactive=True,
                )
                ai_api_key = gr.Textbox(
                    label="Cloud API Key (Optional)",
                    placeholder="Enter Groq (gsk_...) or OpenAI (sk-...) key...",
                    type="password",
                    max_lines=1
                )
                ai_system_prompt = gr.Textbox(
                    label="Custom Agent Prompt / Directives",
                    value=get_default_system_prompt(),
                    lines=4,
                    placeholder="Enter custom instructions or system prompt for the AI Copilot...",
                    interactive=True,
                    elem_id="ai_system_prompt_input"
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
 
            aspect_ratio = gr.Dropdown(
                label='📐 Aspect Ratio',
                choices=config.get_aspect_ratio_labels(),
                value=config.get_aspect_ratio_labels()[9],  # 1024x1024
                interactive=True
            )
            seed_val = gr.Textbox(label='🎲 Seed (-1 = random)', value='-1', max_lines=1)
            batch_size = gr.Slider(
                label='🔢 Batch Size (Number of Images)',
                minimum=1,
                maximum=32,
                step=1,
                value=1,
                interactive=True,
                elem_id='batch_size_slider'
            )
 
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
                "🟢 **Active Engine**: SDXL Juggernaut-XL v9 | **Speed**: 🎯 Master (~15s) | **Zero Slider Setup**",
                elem_id="engine_status_header"
            )
            status = gr.Textbox(label='Status', interactive=False, elem_id='status_display')
            preview = gr.Image(label='Active Design Canvas', type='pil', interactive=False, height=480, elem_id='main_canvas_preview')
            
            with gr.Row():
                quick_rembg_btn = gr.Button("🧹 Remove Background", size="sm", variant="secondary", elem_id="quick_rembg_btn")
                quick_vector_btn = gr.Button("✏️ Vectorize SVG", size="sm", variant="secondary", elem_id="quick_vector_btn")
            
            svg_download_file = gr.File(label="SVG Vector Asset Download", visible=False, elem_id="svg_download_file")
            
            gallery = gr.Gallery(label='Generated Asset History', columns=4, height=220,
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

    def _quick_remove_bg(preview_img):
        if preview_img is None:
            return "⚠️ No active image to remove background from.", None
        try:
            from modules.background_remover import remove_background
            transparent_img = remove_background(preview_img)
            _save_image(transparent_img, config.OUTPUT_DIR)
            return "✅ Background removed cleanly!", transparent_img
        except Exception as e:
            return f"❌ Background removal error: {e}", preview_img

    def _quick_vectorize(preview_img):
        if preview_img is None:
            return "⚠️ No active image to vectorize.", gr.update(visible=False)
        try:
            from modules.starvector_pipeline import image_to_svg
            svg_str = image_to_svg(preview_img)
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            svg_path = os.path.join(config.OUTPUT_DIR, f"vector_{int(time.time()*1000)}.svg")
            with open(svg_path, 'w', encoding='utf-8') as f:
                f.write(svg_str)
            return f"✅ Vectorized to SVG! ({os.path.basename(svg_path)})", gr.update(value=svg_path, visible=True)
        except Exception as e:
            return f"❌ Vectorization error: {e}", gr.update(visible=False)

    quick_rembg_btn.click(
        _quick_remove_bg,
        inputs=[preview],
        outputs=[status, preview]
    )

    quick_vector_btn.click(
        _quick_vectorize,
        inputs=[preview],
        outputs=[status, svg_download_file]
    )

    def _on_ai_enhance(p_val, cat_val, mode_val, key_val, sys_prompt_val):
        from modules.ai_copilot import enhance_prompt_with_ai
        mode = "local" if "local" in str(mode_val).lower() else "hybrid"
        enh_p, enh_neg = enhance_prompt_with_ai(
            p_val, category=cat_val, api_key=key_val, mode=mode, custom_system_prompt=sys_prompt_val
        )
        return gr.update(value=enh_p), gr.update(value=enh_neg), "✨ Prompt enhanced by AI Copilot for commercial quality!"

    ai_enhance_btn.click(
        _on_ai_enhance,
        inputs=[prompt, category, ai_mode_choice, ai_api_key, ai_system_prompt],
        outputs=[prompt, negative_prompt, status]
    )

    generate_btn.click(
        _generate,
        inputs=[
            category, prompt, negative_prompt,
            color1, color2, color3, color4, color5,
            use_master_neg, use_enhancement, remove_bg, vector_mode,
            aspect_ratio, seed_val, speed_choice, styles_selector, base_model_dropdown, batch_size,
            stealth_mode_checkbox
        ],
        outputs=[status, preview, gallery]
    )
 
    return category, prompt, negative_prompt, generate_btn, gallery
