"""
StarVector Pipeline Module
SVG generation using StarVector-1B. Lazy-loaded on first use.
"""
import torch, gc, os

_model = None
_processor = None
MODEL_ID = "starvector/starvector-1b-im2svg"
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'starvector-1b')

def is_loaded():
    return _model is not None

def load_model(progress_cb=None):
    global _model, _processor
    if _model is not None:
        return _model, _processor
    if progress_cb: progress_cb("Downloading StarVector-1B (first use, ~2-3 min)...")
    from transformers import AutoModelForCausalLM, AutoProcessor
    os.makedirs(CACHE_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
    _processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, torch_dtype=dt, trust_remote_code=True).to(dev)
    _model.eval()
    if progress_cb: progress_cb("StarVector-1B loaded!")
    return _model, _processor

def image_to_svg(image, progress_cb=None):
    model, proc = load_model(progress_cb)
    if progress_cb: progress_cb("Vectorizing to SVG...")
    dev = next(model.parameters()).device
    inp = proc(images=image, return_tensors="pt")
    inp = {k: v.to(dev) for k, v in inp.items()}
    with torch.no_grad():
        ids = model.generate(**inp, max_new_tokens=4096, do_sample=False)
    svg = proc.decode(ids[0], skip_special_tokens=True)
    s, e = svg.find('<svg'), svg.rfind('</svg>')
    return svg[s:e+6] if s != -1 and e != -1 else svg

def unload_model():
    global _model, _processor
    del _model; del _processor; _model = None; _processor = None
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
