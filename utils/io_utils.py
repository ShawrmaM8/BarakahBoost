
 # utils/io_utils.py
from __future__ import annotations
import json, os
from typing import Any, Dict, List
import pandas as pd

ROOT = r"C:\Users\muzam\OneDrive\Desktop\PROJECTS\Passion Projects\BarakahBoost"

def ensure_dirs():
    for rel in [
    "data/raw", "data/raw/screen_time", "data/processed", "models",
    "config", "data/reference"
    ]:
        os.makedirs(os.path.join(ROOT, rel), exist_ok=True)

def load_json(path: str, default: Any):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, data: Any):
 os.makedirs(os.path.dirname(path), exist_ok=True)
 with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

def append_daily_log(entry: Dict[str, Any], path: str):
 data = load_json(path, [])
 data.append(entry)
 save_json(path, data)

def read_config()-> Dict[str, Any]:
 cfg_path = os.path.join(ROOT, "config", "config.json")
 return load_json(cfg_path, {})

def write_config(new_cfg: Dict[str, Any]):
 cfg_path = os.path.join(ROOT, "config", "config.json")
 save_json(cfg_path, new_cfg)

def daily_logs_df()-> pd.DataFrame:
 path = os.path.join(ROOT, "data", "raw", "daily_logs.json")
 logs = load_json(path, [])
 if not logs:
    return pd.DataFrame()
 return pd.DataFrame(logs)

def save_df_csv(df: pd.DataFrame, rel_path: str):
 path = os.path.join(ROOT, rel_path)
 os.makedirs(os.path.dirname(path), exist_ok=True)
 df.to_csv(path, index=False)

def list_screen_time_files()-> List[str]:
 d = os.path.join(ROOT, "data", "raw", "screen_time")
 if not os.path.exists(d):
    return []
 return [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith((".json", ".csv"))]