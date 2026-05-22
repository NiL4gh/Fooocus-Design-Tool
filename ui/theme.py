"""Premium dark-mode theme CSS for the Fooocus Design Tool."""

THEME_CSS = """
/* ===== FOOOCUS DESIGN TOOL — PREMIUM DARK THEME ===== */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #12121a;
    --bg-card: #1a1a26;
    --bg-input: #22222e;
    --bg-hover: #2a2a38;
    --border-color: #2d2d3d;
    --border-accent: #4a4a6a;
    --text-primary: #e8e8f0;
    --text-secondary: #9898b0;
    --text-muted: #6a6a80;
    --accent-primary: #7c5cfc;
    --accent-secondary: #5c8cfc;
    --accent-gradient: linear-gradient(135deg, #7c5cfc 0%, #5c8cfc 50%, #5cfcb0 100%);
    --accent-glow: rgba(124, 92, 252, 0.3);
    --success: #4ceb8a;
    --warning: #fcb95c;
    --danger: #fc5c6a;
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --shadow-card: 0 4px 24px rgba(0, 0, 0, 0.4);
    --shadow-glow: 0 0 20px var(--accent-glow);
    --transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Global */
body, .gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}
.gradio-container { max-width: 1400px !important; }

/* Header area */
.app-header {
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.8em;
    font-weight: 700;
    text-align: center;
    padding: 16px 0 8px 0;
    letter-spacing: -0.5px;
}
.app-subtitle {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.9em;
    margin-bottom: 20px;
}

/* Tab styling */
.tabs > .tab-nav { border-bottom: 2px solid var(--border-color) !important; }
.tabs > .tab-nav > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.95em !important;
    color: var(--text-secondary) !important;
    border: none !important;
    padding: 10px 20px !important;
    border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
    transition: var(--transition) !important;
}
.tabs > .tab-nav > button.selected {
    color: var(--accent-primary) !important;
    border-bottom: 2px solid var(--accent-primary) !important;
    background: rgba(124, 92, 252, 0.08) !important;
}
.tabs > .tab-nav > button:hover {
    color: var(--text-primary) !important;
    background: var(--bg-hover) !important;
}

/* Inputs */
.gr-input, .gr-text-input, textarea, input[type="text"],
.gr-box, .prose {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    transition: var(--transition) !important;
}
.gr-input:focus, textarea:focus, input:focus {
    border-color: var(--accent-primary) !important;
    box-shadow: var(--shadow-glow) !important;
}

/* Buttons */
.gr-button, button.primary {
    background: var(--accent-gradient) !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    color: white !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    padding: 10px 24px !important;
    transition: var(--transition) !important;
    box-shadow: var(--shadow-card) !important;
}
.gr-button:hover, button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: var(--shadow-glow), var(--shadow-card) !important;
}
.secondary-btn {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-accent) !important;
    color: var(--text-primary) !important;
}

/* Generate button special */
#generate_btn {
    background: var(--accent-gradient) !important;
    font-size: 1.1em !important;
    padding: 14px 32px !important;
    border-radius: var(--radius-lg) !important;
    letter-spacing: 0.5px;
    min-height: 50px;
}
#generate_btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px var(--accent-glow) !important;
}

/* Gallery */
.image_gallery, .gallery {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-lg) !important;
    min-height: 400px;
}

/* Dropdown */
.gr-dropdown {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
}

/* Sliders */
input[type="range"] {
    accent-color: var(--accent-primary) !important;
}

/* Checkboxes */
.gr-check-radio { accent-color: var(--accent-primary) !important; }

/* Color pickers */
input[type="color"] {
    border: 2px solid var(--border-color) !important;
    border-radius: var(--radius-sm) !important;
    cursor: pointer;
    width: 44px !important;
    height: 44px !important;
}

/* Accordion */
.gr-accordion {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-md) !important;
}

/* Labels */
label, .gr-label {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 0.85em !important;
}

/* Status messages */
.status-msg {
    padding: 12px 16px;
    border-radius: var(--radius-sm);
    font-size: 0.9em;
    margin: 8px 0;
}
.status-info { background: rgba(92, 140, 252, 0.1); border-left: 3px solid var(--accent-secondary); }
.status-success { background: rgba(76, 235, 138, 0.1); border-left: 3px solid var(--success); }
.status-warning { background: rgba(252, 185, 92, 0.1); border-left: 3px solid var(--warning); }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border-accent); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-primary); }

/* Responsive */
@media (max-width: 768px) {
    .gradio-container { padding: 8px !important; }
    .app-header { font-size: 1.3em; }
}
"""
