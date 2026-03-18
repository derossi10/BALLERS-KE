import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

def load_config(file_name):
    with open(BASE_DIR / "config" / file_name, "r") as f:
        return yaml.safe_load(f)