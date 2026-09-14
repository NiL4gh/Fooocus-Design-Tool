"""
AI Copilot Module — Fooocus Designer 2.0
Intelligent prompt expansion, negative prompt crafting, and workflow strategy.
Supports local CPU Qwen2.5-0.5B-Instruct, optional Groq/OpenAI API, and
instant zero-latency heuristic design rule-engine fallback.
"""
import os
import re
import json
import urllib.request
import urllib.error
from typing import Dict, Any, Tuple, Optional

_local_llm_model = None
_local_llm_tokenizer = None
LOCAL_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

CONFIG_PROMPT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'copilot_system_prompt.txt')

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert commercial graphic design prompt engineer for Stable Diffusion XL.\n"
    "Your task is to take a raw user idea and return an enhanced positive prompt and a negative prompt.\n"
    "Respond ONLY with a valid JSON object in this format:\n"
    '{"prompt": "enhanced prompt here", "negative_prompt": "negative prompt here"}'
)


def get_default_system_prompt() -> str:
    """Load default system prompt from config file or return built-in standard."""
    try:
        if os.path.exists(CONFIG_PROMPT_PATH):
            with open(CONFIG_PROMPT_PATH, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    return content
    except Exception:
        pass
    return DEFAULT_SYSTEM_PROMPT

# Commercial design expansion dictionary for instant heuristic enhancement
CATEGORY_MODIFIERS = {
    "Adobe Stock Silhouette": {
        "keywords": "solid pitch black silhouette, razor-sharp clean edges, minimalist vector profile, high contrast graphic symbol, perfectly isolated on white background, commercial clip art asset",
        "negative": "color, gradient, grayscale, photographic shading, textures, background noise, 3d render, watermark",
    },
    "Adobe Stock Flat Vector": {
        "keywords": "flat 2d vector illustration, bold clean outlines, vibrant solid color blocks, smooth SVG paths, modern corporate Memphis aesthetic, professional Adobe Stock graphic asset",
        "negative": "photorealistic photo, gradients, 3d depth, realistic textures, grainy, blurry, messy background, text",
    },
    "Adobe Stock Sticker/Clipart": {
        "keywords": "die-cut vinyl sticker design, clean white contour border, cute vibrant clipart illustration, smooth clean edges, vector patch aesthetic, high contrast isolated white background",
        "negative": "photograph, complex busy background, dark background, blurry edges, clipping artifacts, realism",
    },
    "Adobe Stock Seamless Pattern": {
        "keywords": "seamless tileable repeating pattern, continuous textile surface design, symmetrical geometric motif, clean flat vector elements, commercial fabric wallpaper backdrop",
        "negative": "asymmetric, non-repeating, framed edges, borders, photograph, 3d shadows, perspective view",
    },
    "Logo": {
        "keywords": "modern minimalist vector logo, clean iconic emblem, balanced golden ratio geometry, sleek typography spacing, professional corporate branding symbol, centered composition",
        "negative": "complex realistic photograph, messy lines, human face, realistic animal, noisy details, gradient blur",
    },
    "Poster": {
        "keywords": "commercial advertising poster design, dynamic layout composition, striking visual hierarchy, professional graphic typography balance, clean editorial aesthetic, 8k resolution",
        "negative": "amateur composition, blurry, low contrast, distorted text, cluttered layout",
    },
    "Artwork": {
        "keywords": "masterpiece digital concept artwork, rich cinematic lighting, detailed atmospheric composition, artstation trending quality, harmonious color palette, high fidelity",
        "negative": "blurry, low resolution, deformed anatomy, bad proportions, messy sketches, oversaturated",
    },
}


def _heuristic_enhance(prompt: str, category: str = "") -> Tuple[str, str]:
    """
    Zero-latency heuristic prompt expansion specifically tuned for graphic assets.
    """
    clean_p = (prompt or "").strip()
    if not clean_p:
        return "minimalist geometric commercial design asset", "blurry, low resolution, watermark"

    cat_info = CATEGORY_MODIFIERS.get(category, {
        "keywords": "commercial graphic design asset, clean composition, professional studio lighting, high contrast, 8k sharp detail",
        "negative": "blurry, low resolution, bad anatomy, noisy artifacts, watermark, text distortions",
    })

    # Avoid duplicating existing words
    p_lower = clean_p.lower()
    added_keywords = []
    for word in cat_info["keywords"].split(", "):
        if word.lower() not in p_lower:
            added_keywords.append(word)

    enhanced = f"{clean_p}, {', '.join(added_keywords)}" if added_keywords else clean_p
    negative = cat_info["negative"]
    return enhanced, negative


def _call_groq_or_openai(prompt: str, category: str, api_key: str, custom_system_prompt: Optional[str] = None, base_url: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """Call OpenAI-compatible endpoint (Groq, OpenAI, OpenRouter) for prompt expansion."""
    if not api_key:
        return None

    url = base_url or "https://api.groq.com/openai/v1/chat/completions" if "gsk_" in api_key else "https://api.openai.com/v1/chat/completions"
    model = "llama-3.1-8b-instant" if "gsk_" in api_key else "gpt-4o-mini"

    base_sys = custom_system_prompt.strip() if (custom_system_prompt and custom_system_prompt.strip()) else get_default_system_prompt()
    sys_msg = (
        f"{base_sys}\n\n"
        "Return your response ONLY as a valid JSON object in this format: "
        '{"prompt": "enhanced prompt here", "negative_prompt": "negative prompt here"}'
    )
    user_msg = f"Category: {category or 'General Graphic Asset'}\nUser Idea: {prompt}"

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.7,
        "max_tokens": 200,
        "response_format": {"type": "json_object"}
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key.strip()}",
            "User-Agent": "FooocusDesigner/2.0",
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            return parsed.get("prompt", prompt), parsed.get("negative_prompt", "")
    except Exception as e:
        print(f"[AI Copilot] API call notice: {e}. Falling back to local engine.")
        return None


def _call_local_tiny_llm(prompt: str, category: str, custom_system_prompt: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """Run local Qwen2.5-0.5B-Instruct on CPU without consuming GPU VRAM."""
    global _local_llm_model, _local_llm_tokenizer
    if os.environ.get("MOCK_IMAGE_GEN") == "1":
        return _heuristic_enhance(prompt, category)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        if _local_llm_model is None or _local_llm_tokenizer is None:
            print(f"[AI Copilot] Loading lightweight prompt assistant ({LOCAL_MODEL_ID}) on CPU...")
            _local_llm_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_ID)
            _local_llm_model = AutoModelForCausalLM.from_pretrained(
                LOCAL_MODEL_ID,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
            _local_llm_model.eval()

        system_instruction = (
            custom_system_prompt.strip() if (custom_system_prompt and custom_system_prompt.strip())
            else (
                "You are a professional SDXL prompt enhancer. Expand the user's idea into a single line of descriptive, "
                "photorealistic or vector design tags with lighting and clean background. Keep it under 60 words."
            )
        )
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Category: {category}. Idea: {prompt}"},
        ]
        text = _local_llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = _local_llm_tokenizer([text], return_tensors="pt")

        with torch.no_grad():
            outputs = _local_llm_model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.6,
                pad_token_id=_local_llm_tokenizer.eos_token_id,
            )
        response = _local_llm_tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
        enhanced = response.replace("\n", " ")
        _, neg = _heuristic_enhance(prompt, category)
        return enhanced, neg
    except Exception as e:
        print(f"[AI Copilot] Local LLM notice: {e}. Using fast heuristic enhancer.")
        return None


def enhance_prompt_with_ai(
    prompt: str,
    category: str = "",
    api_key: str = "",
    mode: str = "hybrid",
    custom_system_prompt: Optional[str] = None
) -> Tuple[str, str]:
    """
    Main entry point for AI prompt enhancement.
    1. If api_key is provided, attempts fast API call (Groq/OpenAI).
    2. If mode == "local", attempts local tiny LLM.
    3. Always falls back cleanly to the instant heuristic commercial prompt engine.
    """
    if not (prompt or "").strip():
        return _heuristic_enhance(prompt, category)

    if api_key and api_key.strip():
        res = _call_groq_or_openai(prompt, category, api_key.strip(), custom_system_prompt=custom_system_prompt)
        if res:
            return res

    if mode == "local" and os.environ.get("MOCK_IMAGE_GEN") != "1":
        res = _call_local_tiny_llm(prompt, category, custom_system_prompt=custom_system_prompt)
        if res:
            return res

    # Fast, rock-solid heuristic default
    return _heuristic_enhance(prompt, category)
