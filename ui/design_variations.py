"""
Design Variations Tab — Generate 2-4 similar outputs from a prompt.
"""
import gradio as gr
import os, time
from modules import config


def _gen_variations(prompt, negative, var_count, var_strength, seed_val, model_choice):
    """Generate variations using seed mixing."""
    if not prompt.strip():
        yield "⚠️ Please enter a prompt.", []
        return

    yield f"🔄 Generating variations using {model_choice}...", []

    try:
        from modules.zimage_pipeline import generate_variations

        seed = int(seed_val) if seed_val and str(seed_val).strip() != '-1' else None
        count = int(var_count)

        results = generate_variations(
            prompt=prompt, negative_prompt=negative,
            base_seed=seed, count=count,
            variation_strength=int(var_strength),
            model_name=model_choice,
        )

        paths = []
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        for img, s in results:
            fp = os.path.join(config.OUTPUT_DIR, f"var_{int(time.time()*1000)}_{s}.png")
            img.save(fp)
            paths.append(fp)

        seeds = [str(s) for _, s in results]
        yield f"✅ Generated {len(results)} variations! Seeds: {', '.join(seeds)}", paths

    except Exception as e:
        yield f"❌ Failed: {str(e)}", []


def build_tab():
    """Build the Variations tab."""
    with gr.Row():
        with gr.Column(scale=2):
            var_prompt = gr.Textbox(label='✨ Prompt', placeholder='Enter the base prompt...', lines=3)
            var_model_choice = gr.Dropdown(
                label="🤖 AI Model",
                choices=["FLUX.1-schnell", "Z-Image-Turbo"],
                value="FLUX.1-schnell",
                interactive=True,
                elem_id="var_model_dropdown"
            )
            var_negative = gr.Textbox(label='Negative', placeholder='What to avoid...', lines=1)
            var_count = gr.Slider(label='Number of Variations', minimum=2, maximum=4, step=1, value=4)
            var_strength = gr.Slider(label='Variation Diversity', minimum=10, maximum=100,
                                      step=5, value=50,
                                      info='Higher = more different from each other')
            var_seed = gr.Textbox(label='🎲 Base Seed (-1 = random)', value='-1')
            var_btn = gr.Button('🔄 Generate Variations', variant='primary')

        with gr.Column(scale=3):
            var_status = gr.Textbox(label='Status', interactive=False)
            var_gallery = gr.Gallery(label='Variations', columns=2, height=500, object_fit='contain')

    var_btn.click(
        _gen_variations,
        inputs=[var_prompt, var_negative, var_count, var_strength, var_seed, var_model_choice],
        outputs=[var_status, var_gallery]
    )
