"""Environment variable and .env file loader for CAMIR."""

import os
from pathlib import Path
from typing import Optional


def load_env(env_path: Optional[Path | str] = None) -> None:
    """Load key-value pairs from a .env file into os.environ if not already present.
    
    Searches workspace root by default. Pure Python with no external dependencies.
    """
    if env_path is None:
        # Defaults to repository root .env
        env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    else:
        env_path = Path(env_path)

    if not env_path.exists():
        return

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = val
    except Exception:
        pass
