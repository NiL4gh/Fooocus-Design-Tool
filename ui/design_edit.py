"""
Design Edit Tab — Commercial design transformation and editing.
"""
import gradio as gr
import os, time, random
from modules import config
from modules.sdxl_pipeline import generate as sdxl_generate, load_pipeline
from modules.metadata_manager import build_metadata, save_image_with_metadata


def _edit_generate(edit_image, edit_prompt, edit_negative, edit_strength, model_choice="🎯 Master (~15s)"):
    """Simplified edit generation (inpaint-like using img2img)."""
    if edit_image is None:
        yield "⚠️ Please upload an image to edit.", None, []
        return

    speed_mode = "fast" if "fast" in str(model_choice).lower() else "master"
    yield f"🔄 Initializing pipeline for editing ({speed_mode} mode)...", None, []

    try:
        from PIL import Image
        import numpy as np

        # Prepare image
        if isinstance(edit_image, np.ndarray):
            edit_pil = Image.fromarray(edit_image)
        else:
            edit_pil = edit_image

        edit_pil = edit_pil.convert('RGB').resize((1024, 1024))

        if os.environ.get("MOCK_IMAGE_GEN") == "1":
            image, seed = sdxl_generate(
                prompt=edit_prompt or "same image, improved",
                negative_prompt=edit_negative or "",
                speed_mode=speed_mode,
            )
        else:
            import torch
            from diffusers import AutoPipelineForImage2Image

            pipe = load_pipeline(speed_mode=speed_mode)

            yield f"✏️ Applying design transformation ({speed_mode} mode)...", None, []

            # Load img2img variant
            img2img_pipe = AutoPipelineForImage2Image.from_pipe(pipe)

            seed = random.randint(0, 2**32 - 1)
            generator = torch.Generator(device="cpu").manual_seed(seed)

            steps = 6 if speed_mode == "fast" else 28
            guidance = 1.8 if speed_mode == "fast" else 7.0

            result = img2img_pipe(
                prompt=edit_prompt or "same image, improved",
                negative_prompt=edit_negative or "",
                image=edit_pil,
                strength=edit_strength,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )

            image = result.images[0]

        meta = build_metadata(
            category="Edit Design/Photo",
            prompt=edit_prompt or "same image, improved",
            negative_prompt=edit_negative or "",
            seed=seed,
            speed_mode=speed_mode,
        )
        filepath = os.path.join(config.OUTPUT_DIR, f"edit_{int(time.time()*1000)}.png")
        save_image_with_metadata(image, filepath, meta)

        yield f"✅ Transformation complete! Seed: {seed}", image, [filepath]

    except Exception as e:
        yield f"❌ Edit failed: {str(e)}", None, []


def build_tab():
    """Build the Edit tab."""
    with gr.Row():
        with gr.Column(scale=2):
            edit_image = gr.Image(label='📷 Upload Image to Edit', type='numpy',
                                   sources=['upload'], height=360, elem_id="edit_input_image")
            with gr.Row():
                edit_rembg_btn = gr.Button("🧹 Remove Background", size="sm", variant="secondary")
                edit_vector_btn = gr.Button("✏️ Vectorize SVG", size="sm", variant="secondary")
            
            edit_svg_file = gr.File(label="SVG Vector Asset", visible=False, elem_id="edit_svg_file")

            edit_model_choice = gr.Dropdown(
                label="⚡ Engine Mode",
                choices=["🎯 Master (~15s)", "⚡ Fast (~3s)"],
                value="🎯 Master (~15s)",
                interactive=True,
                elem_id="edit_model_dropdown"
            )
            edit_prompt = gr.Textbox(label='✨ Transformation Prompt',
                                     placeholder='Describe the modifications you want (e.g. change color, add details)...',
                                     lines=2)
            edit_negative = gr.Textbox(label='Negative', placeholder='What to avoid...', lines=1)
            edit_strength = gr.Slider(label='Transformation Strength', minimum=0.1, maximum=1.0,
                                       step=0.05, value=0.6,
                                       info='Lower (0.2-0.4) = subtle touch-ups, Higher (0.6-0.9) = major creative restyling')
            edit_btn = gr.Button('✏️ Apply Transformation', variant='primary', elem_id="edit_generate_btn")

        with gr.Column(scale=3):
            edit_status = gr.Textbox(label='Status', interactive=False, elem_id="edit_status_display")
            edit_preview = gr.Image(label='Result Canvas', type='pil', interactive=False, height=440)
            edit_gallery = gr.Gallery(label='Edit History', columns=4, height=200, elem_id="edit_history_gallery")

    def _on_edit_rembg(img):
        if img is None:
            return "⚠️ Please upload an image first.", None
        from modules.background_remover import remove_background
        from PIL import Image
        import numpy as np
        pil_img = Image.fromarray(img) if isinstance(img, np.ndarray) else img
        res = remove_background(pil_img)
        return "✅ Background removed cleanly!", res

    def _on_edit_vector(img):
        if img is None:
            return "⚠️ Please upload an image first.", gr.update(visible=False)
        from modules.starvector_pipeline import image_to_svg
        from PIL import Image
        import numpy as np
        pil_img = Image.fromarray(img) if isinstance(img, np.ndarray) else img
        svg_code = image_to_svg(pil_img)
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        svg_path = os.path.join(config.OUTPUT_DIR, f"edit_vector_{int(time.time()*1000)}.svg")
        with open(svg_path, 'w', encoding='utf-8') as f:
            f.write(svg_code)
        return f"✅ Vectorized to SVG! ({os.path.basename(svg_path)})", gr.update(value=svg_path, visible=True)

    edit_rembg_btn.click(
        _on_edit_rembg,
        inputs=[edit_image],
        outputs=[edit_status, edit_preview]
    )

    edit_vector_btn.click(
        _on_edit_vector,
        inputs=[edit_image],
        outputs=[edit_status, edit_svg_file]
    )

    edit_btn.click(
        _edit_generate,
        inputs=[edit_image, edit_prompt, edit_negative, edit_strength, edit_model_choice],
        outputs=[edit_status, edit_preview, edit_gallery]
    )
