import platform
import sys
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent

BG_GRAD_TOP = "#0a1022"
BG_PANEL = "#101f3c"
BG_LIST = "#11244a"
HL_LIST = "#22386a"
TEXT = "#e9f1ff"
MUTED = "#a9b8d6"
ACCENT = "#6ea5ff"
ACCENT_HOVER = "#84b5ff"
CONTROL_BG = "#1b2f56"
CONTROL_BORDER = "#2f4474"
BORDER_ACCENT = "#91bbff"
SURFACE_PRIMARY = "#172a52"
SURFACE_ELEVATED = "#1e3561"
SURFACE_HOVER = "#2a4475"
CRITICAL = "#b84d4d"
CRITICAL_HOVER = "#d85f5f"
SUCCESS = "#58b896"
SUCCESS_HOVER = "#66c7a4"

TITLE_BAR_COLOR = "#101a2d"
TITLE_BAR_ACCENT = "#1f2f52"
PANEL_ELEVATED = "#152847"
GLASS_BG = "#1c3158"

CONFIG_PATH = BASE_DIR / "config" / "settings.json"
MODELS_PATH = BASE_DIR / "models"
OUTPUTS_PATH = BASE_DIR / "outputs"


THEMES = { 
    "Blue": {
        "BG_GRAD_TOP": "#0a1022",
        "BG_PANEL": "#101f3c",
        "BG_LIST": "#11244a",
        "HL_LIST": "#22386a",
        "TEXT": "#e9f1ff",
        "MUTED": "#a9b8d6",
        "ACCENT": "#6ea5ff",
        "ACCENT_HOVER": "#84b5ff",
        "CONTROL_BG": "#1b2f56",
        "CONTROL_BORDER": "#2f4474",
        "BORDER_ACCENT": "#91bbff",  
        "TITLE_BAR_ACCENT": "#1f2f52", 
        "SURFACE_PRIMARY": "#172a52",
        "SURFACE_ELEVATED": "#1e3561",
        "SURFACE_HOVER": "#2a4475",
        "CRITICAL": "#b84d4d",
        "CRITICAL_HOVER": "#d85f5f",
        "SUCCESS": "#58b896",
        "SUCCESS_HOVER": "#66c7a4",
        "TITLE_BAR_COLOR": "#060b19",
        "TITLE_BAR_ACCENT": "#1f2f52",
        "PANEL_ELEVATED": "#101f3c",
        "GLASS_BG": "#11244a",
    },
    "Green": {
        "BG_GRAD_TOP": "#0e1d17",
        "BG_PANEL": "#14271f",
        "BG_LIST": "#132d21",
        "HL_LIST": "#214836",
        "TEXT": "#e9fbee",
        "MUTED": "#a8cab5",
        "ACCENT": "#3fb97e",
        "BORDER_ACCENT": "#67d6a0",
        "ACCENT_HOVER": "#55c791",
        "SURFACE_PRIMARY": "#183328",
        "SURFACE_ELEVATED": "#1e4333",
        "SURFACE_HOVER": "#265a43",
        "CONTROL_BG": "#19382d",
        "CONTROL_BORDER": "#285c47",
        "CRITICAL": "#c95a5a",
        "CRITICAL_HOVER": "#da6e6e",
        "SUCCESS": "#3fc181",
        "SUCCESS_HOVER": "#4fce8f",
        "TITLE_BAR_COLOR": "#13231d",
        "TITLE_BAR_ACCENT": "#1e3a2f",
        "PANEL_ELEVATED": "#18352b",
        "GLASS_BG": "#1d4031",
    },
     "Black": {
        "BG_GRAD_TOP": "#000000",
        "BG_PANEL":   "#0c0c0c",
        "BG_LIST":    "#101522",
        "HL_LIST":    "#22386a",
        "TEXT":  "#ffffff",
        "MUTED": "#cfcfcf",
        "ACCENT":        "#6ea5ff",
        "ACCENT_HOVER":  "#84b5ff",
        "BORDER_ACCENT": "#91bbff",
        "SURFACE_PRIMARY":  "#151926",
        "SURFACE_ELEVATED": "#1b1f2e",
        "SURFACE_HOVER":    "#22273a",
        "CONTROL_BG":     "#11151f",
        "CONTROL_BORDER": "#2f4474",
        "CRITICAL":        "#e15c5c",
        "CRITICAL_HOVER":  "#ef7070",
        "SUCCESS":         "#4cd964",
        "SUCCESS_HOVER":   "#6fe584",
        "TITLE_BAR_COLOR":  "#05070c",
        "TITLE_BAR_ACCENT": "#1f2f52",
        "PANEL_ELEVATED":   "#151926",
        "GLASS_BG":         "#1d2233",
    },
    "White":{
        "BG_GRAD_TOP":"#F3F4F6",
        "BG_PANEL":"#F9FAFB",
        "BG_LIST":"#D9DEE7",
        "HL_LIST":"#C7CEDA",
        "TEXT":"#111827",
        "MUTED":"#374151",
        "ACCENT":"#2563EB",
        "ACCENT_HOVER":"#3B82F6",
        "BORDER_ACCENT":"#93C5FD",
        "SURFACE_PRIMARY":"#C9D0DC",
        "SURFACE_ELEVATED":"#C3CAD7",
        "PANEL_ELEVATED":"#C9D0DC",
        "SURFACE_HOVER":"#B7BFCE",
        "GLASS_BG":"#D0D6E3",
        "CONTROL_BG":"#FFFFFF",
        "CONTROL_BORDER":"#9CA3AF",
        "CRITICAL":"#B91C1C",
        "CRITICAL_HOVER":"#DC2626",
        "SUCCESS":"#15803D",
        "SUCCESS_HOVER":"#16A34A",
        "TITLE_BAR_COLOR":"#E3E7EF",
        "TITLE_BAR_ACCENT":"#D0D6E3"
    },
    "Purple": {
        "BG_GRAD_TOP": "#1b1030",
        "BG_PANEL": "#22183e",
        "BG_LIST": "#281d48",
        "HL_LIST": "#3a2e60",
        "TEXT": "#f2ebff",
        "MUTED": "#b7a8d6",
        "ACCENT": "#a679ff",
        "BORDER_ACCENT": "#b893ff",
        "ACCENT_HOVER": "#bb8fff",
        "SURFACE_PRIMARY": "#2c1e52",
        "SURFACE_ELEVATED": "#372869",
        "SURFACE_HOVER": "#493a7d",
        "CONTROL_BG": "#2e1f56",
        "CONTROL_BORDER": "#4a3882",
        "CRITICAL": "#d35f5f",
        "CRITICAL_HOVER": "#e36f6f",
        "SUCCESS": "#58b896",
        "SUCCESS_HOVER": "#66c7a4",
        "TITLE_BAR_COLOR": "#20143a",
        "TITLE_BAR_ACCENT": "#3a2b6a",
        "PANEL_ELEVATED": "#2d1e50",
        "GLASS_BG": "#3b2a6b",
    },
    "Amber" : {
        "BG_GRAD_TOP": "#2c1a08",
        "BG_PANEL": "#3f2510",
        "BG_LIST": "#462b13",
        "HL_LIST": "#5b391c",
        "TEXT": "#fff4e9",
        "MUTED": "#c8b09a",
        "ACCENT": "#f5a623",
        "BORDER_ACCENT": "#f7b64f",
        "ACCENT_HOVER": "#f8b93a",
        "SURFACE_PRIMARY": "#4c2e13",
        "SURFACE_ELEVATED": "#5b3718",
        "SURFACE_HOVER": "#6d441f",
        "CONTROL_BG": "#4a2d12",
        "CONTROL_BORDER": "#7a4b22",
        "CRITICAL": "#d35454",
        "CRITICAL_HOVER": "#e36c6c",
        "SUCCESS": "#58b896",
        "SUCCESS_HOVER": "#66c7a4",
        "TITLE_BAR_COLOR": "#3a1f0d",
        "TITLE_BAR_ACCENT": "#5a3318",
        "PANEL_ELEVATED": "#4b2913",
        "GLASS_BG": "#553017",
    }
}

def _read_theme_name() -> str:
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return data.get("prefs", {}).get("theme", "Blue")
    except Exception:
        return "Blue"

def _write_theme_name(name: str) -> None:
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    prefs = data.setdefault("prefs", {})
    prefs["theme"] = name
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def _apply_theme_globals(theme: dict):
    g = globals()
    for k, v in theme.items():
        g[k] = v

_apply_theme_globals(THEMES.get(_read_theme_name(), THEMES["Blue"]))

def switch_theme(theme_name: str):
    name = theme_name if theme_name in THEMES else "Blue"
    _write_theme_name(name)
    _apply_theme_globals(THEMES[name])

_apply_theme_globals(THEMES.get(_read_theme_name(), THEMES["Blue"]))

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
