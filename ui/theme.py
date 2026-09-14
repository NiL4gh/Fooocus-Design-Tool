"""
Premium Dark-Mode Webapp Theme CSS for Fooocus Designer 2.0.
Full-width responsive workstation layout inspired by Meta/Linear/Vercel design systems.
"""

THEME_CSS = """
/* ===== FOOOCUS DESIGNER 2.0 — MODERN WEBAPP DESIGN SYSTEM ===== */

@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-base: #090a10;
    --bg-surface: #10121d;
    --bg-card: rgba(18, 20, 32, 0.75);
    --bg-card-hover: rgba(26, 29, 46, 0.85);
    --bg-input: #151827;
    --bg-input-focus: #1c2033;
    
    --border-subtle: rgba(255, 255, 255, 0.07);
    --border-default: rgba(255, 255, 255, 0.12);
    --border-active: rgba(99, 102, 241, 0.5);
    
    --text-high: #f8fafc;
    --text-medium: #94a3b8;
    --text-low: #64748b;
    
    --brand-primary: #6366f1;
    --brand-secondary: #8b5cf6;
    --brand-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
    --brand-glow: rgba(99, 102, 241, 0.25);
    
    --accent-cyan: #06b6d4;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;
    
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-full: 9999px;
    
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow-card: 0 4px 20px -2px rgba(0, 0, 0, 0.5), 0 0 0 1px var(--border-subtle);
    --shadow-glow: 0 0 25px var(--brand-glow);
    --shadow-glow-strong: 0 0 35px rgba(99, 102, 241, 0.45);
}

/* Base resets & full-width layout */
html, body {
    background-color: var(--bg-base) !important;
    color: var(--text-high) !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow-x: hidden !important;
}

.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    padding: 16px 28px !important;
    margin: 0 !important;
    box-sizing: border-box !important;
    background-color: var(--bg-base) !important;
}

/* Header Webapp Bar */
.app-header-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 20px;
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    margin-bottom: 16px;
    backdrop-filter: blur(12px);
}

.app-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.app-brand-title {
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: var(--brand-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.app-badge {
    display: inline-flex;
    align-items: center;
    padding: 3px 9px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    border-radius: var(--radius-full);
    background: rgba(99, 102, 241, 0.12);
    color: #a5b4fc;
    border: 1px solid rgba(99, 102, 241, 0.25);
}

/* Modern Tab Bar */
.tabs {
    border: none !important;
    background: transparent !important;
}

.tab-nav {
    display: flex !important;
    gap: 8px !important;
    padding: 6px !important;
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    margin-bottom: 20px !important;
}

.tab-nav button {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: var(--text-low) !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: var(--radius-sm) !important;
    padding: 8px 20px !important;
    transition: all 0.18s ease !important;
}

.tab-nav button:hover {
    color: var(--text-high) !important;
    background: var(--bg-card-hover) !important;
}

.tab-nav button.selected {
    color: #ffffff !important;
    background: var(--brand-primary) !important;
    box-shadow: 0 2px 12px var(--brand-glow) !important;
}

/* Card Panels & Containers */
.gr-panel, .gr-box, .gr-form {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-lg) !important;
    backdrop-filter: blur(12px) !important;
}

/* Text Inputs, Textarea, Dropdowns */
.gr-input, .gr-text-input, textarea, input[type="text"], .gr-dropdown {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-high) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.92rem !important;
    transition: all 0.15s ease !important;
}

.gr-input:focus, textarea:focus, input[type="text"]:focus {
    background: var(--bg-input-focus) !important;
    border-color: var(--border-active) !important;
    box-shadow: 0 0 0 3px var(--brand-glow) !important;
    outline: none !important;
}

/* Accordion Component */
.gr-accordion {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    margin-bottom: 10px !important;
}

.gr-accordion > .label-wrap {
    padding: 10px 14px !important;
}

.gr-accordion > .label-wrap span {
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    color: var(--text-medium) !important;
}

/* Buttons */
.gr-button {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    border-radius: var(--radius-md) !important;
    transition: all 0.18s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
}

.gr-button-primary, button.primary, #generate_btn {
    background: var(--brand-gradient) !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 4px 16px var(--brand-glow) !important;
    font-size: 1.02rem !important;
    padding: 12px 24px !important;
}

.gr-button-primary:hover, button.primary:hover, #generate_btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-glow-strong) !important;
}

.gr-button-secondary {
    background: var(--bg-surface) !important;
    color: var(--text-high) !important;
    border: 1px solid var(--border-default) !important;
}

.gr-button-secondary:hover {
    background: var(--bg-card-hover) !important;
    border-color: var(--border-active) !important;
}

/* Radio & Checkbox */
input[type="radio"], input[type="checkbox"] {
    accent-color: var(--brand-primary) !important;
}

/* Engine Status Header Pill */
#engine_status_header {
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.25);
    border-radius: var(--radius-md);
    padding: 8px 14px;
    color: #6ee7b7;
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 12px;
}

/* Gallery & Image Previews */
.gallery, .image_gallery {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-lg) !important;
}

/* Scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-track {
    background: var(--bg-base);
}

::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--brand-primary);
}
"""
