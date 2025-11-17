import platform
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

BG_GRAD_TOP = "#08142c"
BG_PANEL = "#101f3c"
BG_LIST = "#11244a"
HL_LIST = "#22386a"
TEXT = "#e9f1ff"
MUTED = "#a9b8d6"
ACCENT = "#6ea5ff"
BORDER_ACCENT = "#91bbff"
ACCENT_HOVER = "#84b5ff"
SURFACE_PRIMARY = "#172a52"
SURFACE_ELEVATED = "#1e3561"
SURFACE_HOVER = "#2a4475"
CONTROL_BG = "#1b2f56"
CONTROL_BORDER = "#2f4474"
CRITICAL = "#b84d4d"
CRITICAL_HOVER = "#d85f5f"
SUCCESS = "#58b896"
SUCCESS_HOVER = "#66c7a4"
BASE_UI = 14
BASE_H1 = 22
BASE_H2 = 18

def _get_font_family() -> str:
    system = platform.system().lower()
    if system == "darwin":
        return "Lucida Grande"
    elif system == "windows":
        return "Segoe UI"
    else:
        return "DejaVu Sans"

FONT_FAMILY = _get_font_family()
FONT_UI = (FONT_FAMILY, BASE_UI)
FONT_BOLD = (FONT_FAMILY, BASE_UI, "bold")
FONT_H1 = (FONT_FAMILY, BASE_H1, "bold")
FONT_H2 = (FONT_FAMILY, BASE_H2, "bold")
PLACEHOLDER = "Ask me anything..."
CONFIG_PATH = BASE_DIR / "config" / "settings.json"
MODELS_PATH = BASE_DIR / "models"
OUTPUTS_PATH = BASE_DIR / "outputs"
DISCLAIMER_TEXT = (
    "WARNING — Read Carefully:\n\n"
    "This application runs offline language models that do NOT have access to real-time information.\n"
    "Models were trained on data available up to their release date and may provide inaccurate,\n"
    "outdated, or incomplete answers. They can also reflect biases present in their training data.\n"
    "Under NO circumstances should responses from these models be used for life-critical, medical,\n"
    "legal, or safety-related decisions. Always verify important information with qualified professionals\n"
    "or authoritative sources. The developers provide this tool for research and experimentation only;\n"
    "we do not assume responsibility for decisions made based on model output.\n\n"
    "By checking the box below and clicking 'I Understand', you acknowledge that you have read this\n"
    "disclaimer, understand the limitations, and accept responsibility for how you use the system.\n"
)

RADIUS_LG = 18
RADIUS_MD = 14
RADIUS_SM = 10
BUTTON_RADIUS = 12

TITLE_BAR_HEIGHT = 46
TITLE_BAR_COLOR = "#101a2d"
TITLE_BAR_ACCENT = "#1f2f52"

PANEL_ELEVATED = "#152847"
GLASS_BG = "#1c3158"
