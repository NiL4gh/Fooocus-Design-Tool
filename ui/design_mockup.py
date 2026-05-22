"""
Logo Mockup Tab — Upload a logo, choose product and mockup layout style, dynamically generate mockup scene.
"""
import gradio as gr
import os
import time
import random
from modules.logo_mockup import get_product_types, get_mockup_styles, generate_mockup
from modules import config


def _gen_mockup(logo_image, product_type, scene_prompt, mockup_style):
    """Generate dynamic mockup with progressive status updates."""
    if logo_image is None:
        yield "⚠️ Please upload a logo image first.", None
        return

    yield "🔄 Initializing mockup pipeline...", None

    def cb(msg):
        print(f"[Mockup] {msg}")

    yield f"🎨 Generating empty {product_type} ({mockup_style}) background scene with AI...", None

    try:
        mockup_img, status_msg = generate_mockup(
            logo_image, 
            product_type, 
            scene_prompt, 
            mockup_style,
            progress_cb=cb
        )
        
        if mockup_img is None:
            yield status_msg, None
            return

        yield "💾 Saving generated mockup design...", mockup_img

        # Save to outputs folder
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        timestamp = int(time.time() * 1000)
        rand = random.randint(1000, 9999)
        filepath = os.path.join(config.OUTPUT_DIR, f"mockup_{timestamp}_{rand}.png")
        mockup_img.save(filepath)

        yield f"{status_msg} | Saved: {os.path.basename(filepath)}", mockup_img

    except Exception as e:
        yield f"❌ Mockup generation failed: {str(e)}", None


def build_tab():
    """Build the Mockup tab."""
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📦 AI-Powered Logo Mockup Generator")
            gr.Markdown("Upload any logo graphic and dynamically place it on a photorealistic product mockup scene.")
            mockup_logo = gr.Image(label='📷 Upload Logo', type='pil', sources=['upload'], height=300)
            
            with gr.Row():
                mockup_product = gr.Dropdown(label='🏷️ Product Type', choices=get_product_types(),
                                              value=get_product_types()[0])
                mockup_style = gr.Dropdown(label='🎨 Mockup Style Layout', choices=get_mockup_styles(),
                                            value=get_mockup_styles()[0])
                                            
            mockup_prompt = gr.Textbox(label='Scene Description (optional)',
                                        placeholder='e.g., resting on a solid concrete desk with warm sunlight, studio lighting...',
                                        lines=2)
            mockup_btn = gr.Button('📦 Generate Mockup', variant='primary')

        with gr.Column(scale=3):
            mockup_status = gr.Textbox(label='Status', interactive=False, elem_id='mockup_status_display')
            mockup_preview = gr.Image(label='Mockup Result', type='pil', interactive=False, height=450)
            gr.Markdown("""
            ### 💡 Custom Mockup Layout Styles:
            *   **Single Product (Centered)**: Generates a standard high-quality clean studio product presentation mockup.
            *   **Bento Knolling Layout**: Generates a flat-lay knolling stationery arrangement with paper/plastic textures.
            *   **Realistic Ambient Scene**: Generates lifestyle, commercial editorial photography in a real environment with soft natural lighting.
            """)

    mockup_btn.click(
        _gen_mockup,
        inputs=[mockup_logo, mockup_product, mockup_prompt, mockup_style],
        outputs=[mockup_status, mockup_preview]
    )
