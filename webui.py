"""
Fooocus Design Tool — Web UI
A specialized graphic design asset generator built on Z-Image-Turbo.
"""
import gradio as gr
from ui.theme import THEME_CSS
from ui import design_main, design_edit, design_variations, design_mockup
from modules.flags import APP_NAME, VERSION


def clean_vram():
    """Clean GPU VRAM memory by unloading active models and collecting garbage."""
    import gc
    import torch
    from modules.zimage_pipeline import unload_pipeline, is_loaded as is_z
    from modules.starvector_pipeline import unload_model, is_loaded as is_sv
    
    freed = []
    try:
        if is_z():
            unload_pipeline()
            freed.append("Z-Image-Turbo")
        if is_sv():
            unload_model()
            freed.append("StarVector-1B")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        
        if freed:
            freed_str = ", ".join(freed)
            return f'<div class="status-msg status-success">🧹 Successfully unloaded: <b>{freed_str}</b>. GPU memory cleared!</div>'
        else:
            return '<div class="status-msg status-info">🧹 GPU memory is already clean. No active models were loaded!</div>'
    except Exception as e:
        return f'<div class="status-msg status-warning">⚠️ Error cleaning VRAM: {str(e)}</div>'


def create_app():
    """Create and configure the Gradio application."""
    
    with gr.Blocks(
        title=f"{APP_NAME} v{VERSION}",
        css=THEME_CSS,
        theme=gr.themes.Base(
            primary_hue="violet",
            secondary_hue="blue",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
    ) as app:
        # Header
        gr.HTML(f'<div class="app-header">🎨 {APP_NAME}</div>')
        gr.HTML(f'<div class="app-subtitle">AI-Powered Design Asset Generator • v{VERSION}</div>')

        # Main tabs
        with gr.Tabs():
            with gr.Tab("🎯 Generate", id="generate_tab"):
                design_main.build_tab()

            with gr.Tab("✏️ Edit", id="edit_tab"):
                design_edit.build_tab()

            with gr.Tab("🔄 Variations", id="variations_tab"):
                design_variations.build_tab()

            with gr.Tab("📦 Mockup", id="mockup_tab"):
                design_mockup.build_tab()

        # Footer / Utilities
        with gr.Row(elem_id="app_footer"):
            with gr.Column(scale=4):
                gr.HTML(f'<div style="text-align: center; color: var(--text-muted); font-size: 0.85em; margin-top: 20px;">'
                        f'🎨 {APP_NAME} v{VERSION} • Optimized for Graphic Design Asset Production</div>')
            with gr.Column(scale=1):
                vram_btn = gr.Button("🧹 Clean GPU VRAM", variant="secondary", elem_id="vram_clean_btn")
                vram_status = gr.HTML(value="", elem_id="vram_status")

        vram_btn.click(clean_vram, inputs=[], outputs=[vram_status])

    return app


# Create and launch the app
app = create_app()

if __name__ == '__main__':
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=7865,
        share=False,
        inbrowser=True,
    )
