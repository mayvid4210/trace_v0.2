import fastf1
import os

CACHE_DIR = "data/raw"

os.makedirs(CACHE_DIR, exist_ok=True)

fastf1.Cache.enable_cache(CACHE_DIR)

EVENTS = [
    {"year": 2024, "gp": "Bahrain", "sessions": ["FP1", "FP2", "FP3", "R"]},
    {"year": 2024, "gp": "Canada", "sessions": ["FP1", "FP2", "FP3", "R"]},
]