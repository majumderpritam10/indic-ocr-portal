"""Central paths. Everything is resolved relative to the folder that holds app.py,
so the app runs the same on your laptop and on Streamlit Cloud (no hard-coded paths)."""
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = APP_ROOT / "data"
REPORTS_DIR = APP_ROOT / "reports"
CORRECTED_DIR = DATA_DIR / "corrected"
RAW_IMAGES_DIR = DATA_DIR / "raw" / "indic_ocr_dataset" / "images"
PREP_IMAGES_DIR = DATA_DIR / "preprocessed" / "images"
FINAL_IMAGES_DIR = DATA_DIR / "final" / "images"
AUG_DIR = DATA_DIR / "augmented"

LANG_NAMES = {"bn": "Bengali", "hi": "Hindi", "en": "English"}


def norm_rel(p) -> str:
    """The project was built on Windows, so manifests contain 'data\\final\\images\\x.png'.
    Convert to forward slashes so the same path works everywhere."""
    return str(p).replace("\\", "/")


def resolve(rel) -> Path:
    return APP_ROOT / norm_rel(rel)


def rel_to_root(p: Path) -> str:
    try:
        return p.relative_to(APP_ROOT).as_posix()
    except ValueError:
        return str(p)
