"""
Design Edit Tab — Simplified inpainting/outpainting for design modification.
"""
import gradio as gr
import os, time, random
from modules import config


def _edit_generate(edit_image, edit_prompt, edit_negative, edit_strength):
    """Simplified edit generation (inpaint-like using img2img)."""
    if edit_image is None:
        yield "⚠️ Please upload an image to edit.", None, []
        return

    yield "🔄 Loading pipeline for editing...", None, []

    try:
        from modules.zimage_pipeline import load_pipeline
        import torch
        from PIL import Image
        import numpy as np

        pipe = load_pipeline()

        # For diffusers-based editing, we use img2img pipeline
        yield "✏️ Editing image...", None, []

        from diffusers import AutoPipelineForImage2Image
        
        # Load img2img variant
        img2img_pipe = AutoPipelineForImage2Image.from_pipe(pipe)

        # Prepare image
        if isinstance(edit_image, np.ndarray):
            edit_pil = Image.fromarray(edit_image)
        else:
            edit_pil = edit_image

        edit_pil = edit_pil.convert('RGB').resize((1024, 1024))

        seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)

        result = img2img_pipe(
            prompt=edit_prompt or "same image, improved",
            negative_prompt=edit_negative or "",
            image=edit_pil,
            strength=edit_strength,
            num_inference_steps=4,
            guidance_scale=0.0,
            generator=generator,
        )

        image = result.images[0]

        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        filepath = os.path.join(config.OUTPUT_DIR, f"edit_{int(time.time()*1000)}.png")
        image.save(filepath)

        yield f"✅ Edit complete! Seed: {seed}", image, [filepath]

    except Exception as e:
        yield f"❌ Edit failed: {str(e)}", None, []


def build_tab():
    """Build the Edit tab."""
    with gr.Row():
        with gr.Column(scale=2):
            edit_image = gr.Image(label='📷 Upload Image to Edit', type='numpy',
                                   sources=['upload'], height=400)
            edit_prompt = gr.Textbox(label='✨ Edit Prompt',
                                     placeholder='Describe the changes you want...',
                                     lines=2)
            edit_negative = gr.Textbox(label='Negative', placeholder='What to avoid...', lines=1)
            edit_strength = gr.Slider(label='Edit Strength', minimum=0.1, maximum=1.0,
                                       step=0.05, value=0.6,
                                       info='Lower = subtle changes, Higher = major changes')
            edit_btn = gr.Button('✏️ Apply Edit', variant='primary')

        with gr.Column(scale=3):
            edit_status = gr.Textbox(label='Status', interactive=False)
            edit_preview = gr.Image(label='Result', type='pil', interactive=False, height=400)
            edit_gallery = gr.Gallery(label='Edit History', columns=4, height=200)

    edit_btn.click(
        _edit_generate,
        inputs=[edit_image, edit_prompt, edit_negative, edit_strength],
        outputs=[edit_status, edit_preview, edit_gallery]
    )
