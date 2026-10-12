from pathlib import Path
import tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_config(filename="config.toml"):
    config_path = PROJECT_ROOT / "config" / filename

    with open(config_path, "rb") as file:
        return tomllib.load(file)

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)