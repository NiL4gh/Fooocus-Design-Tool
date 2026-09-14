"""
Fooocus Design Tool — Web UI
A specialized graphic design asset generator built on SDXL.
"""
import gradio as gr
from ui.theme import THEME_CSS
from ui import design_main, design_edit, design_mockup
from modules.flags import APP_NAME, VERSION


def clean_vram():
    """Clean GPU VRAM memory by unloading active models and collecting garbage (utility)."""
    import gc
    import torch
    from modules.sdxl_pipeline import unload_pipeline, is_loaded as is_sdxl
    from modules.starvector_pipeline import unload_model, is_loaded as is_sv
    
    freed = []
    try:
        if is_sdxl():
            unload_pipeline()
            freed.append("Juggernaut-XL (SDXL)")
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
            font=gr.themes.GoogleFont("Plus Jakarta Sans"),
        ),
    ) as app:
        # Header Webapp Bar
        gr.HTML(f'''
        <div class="app-header-container">
            <div class="app-brand">
                <span style="font-size: 1.6rem;">🎨</span>
                <div>
                    <span class="app-brand-title">{APP_NAME}</span>
                    <span class="app-badge">v{VERSION} PRO WORKSTATION</span>
                </div>
            </div>
            <div style="font-size: 0.82rem; color: var(--text-medium); font-weight: 500;">
                Zero-Friction Graphic Design Asset Studio • Powered by SDXL
            </div>
        </div>
        ''')

        # Main tabs
        with gr.Tabs():
            with gr.Tab("🎯 Generate", id="generate_tab"):
                design_main.build_tab()

            with gr.Tab("✏️ Edit", id="edit_tab"):
                design_edit.build_tab()

            with gr.Tab("📦 Mockup", id="mockup_tab"):
                design_mockup.build_tab()

        # Footer
        gr.HTML(f'<div style="text-align: center; color: var(--text-low); font-size: 0.8rem; margin-top: 24px; padding-bottom: 12px;">'
                f'🎨 {APP_NAME} v{VERSION} • Commercial Stock & Graphic Asset Production Studio</div>')

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
