from pathlib import Path
import os
import fastf1

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directory references
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Locate raw data (existing FastF1 cache & external datasets live under backend/data/raw)
BACKEND_RAW_DIR = PROJECT_ROOT / "backend" / "data" / "raw"
ROOT_RAW_DIR = DATA_DIR / "raw"
RAW_DATA_DIR = BACKEND_RAW_DIR if BACKEND_RAW_DIR.exists() else ROOT_RAW_DIR

EXTERNAL_DATA_DIR = RAW_DATA_DIR / "external"
CACHE_DIR = str(RAW_DATA_DIR)

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to PROJECT_ROOT if it is not already absolute."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


fastf1.Cache.enable_cache(CACHE_DIR)

EVENTS = [
    {"year": 2024, "gp": "Bahrain", "sessions": ["FP1", "FP2", "FP3", "R"]},
    {"year": 2024, "gp": "Canada", "sessions": ["FP1", "FP2", "FP3", "R"]},
]