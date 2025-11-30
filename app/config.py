# config.py
from pathlib import Path
import os


class Config:
    # Base dir
    BASE_DIR = Path.cwd()

    # Upload roots (can be overridden by env vars)
    UPLOAD_TRAINING_ROOT = Path(
        os.environ.get("UPLOAD_TRAINING_ROOT", BASE_DIR / "uploaded_training")
    )
    UPLOAD_DATASET_ROOT = Path(
        os.environ.get("UPLOAD_DATASET_ROOT", BASE_DIR / "uploaded_datasets")
    )

    # Ensure dirs exist
    UPLOAD_TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
    UPLOAD_DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    MAX_CONTENT_LENGTH = 1024 * 1024 * 1024  # 1GB

    # Allowed file types
    TRAINING_ALLOWED_EXTS = {
        ".py", ".ipynb", ".zip", ".txt", ".md",
        ".json", ".yaml", ".yml", ".ini", ".cfg",
    }
    DATASET_ALLOWED_EXTS = {
        ".csv", ".tsv", ".parquet", ".xlsx", ".zip",
        ".txt", ".md",
        ".png", ".jpg", ".jpeg", ".bmp",
    }

    # Entry point candidates for training scripts (add ipynb variants)
    ENTRYPOINT_CANDIDATES = {
        "train.py", "main.py", "finetune.py",
        "train.ipynb", "main.ipynb", "finetune.ipynb",
    }

    # GPU options for v1 estimator
    GPU_OPTIONS = [
        {"Item": "NVIDIA H200", "vRAM": 141, "vCPUs": 30, "RAM_GB": 375, "Price_per_Hour": 3.49},
        {"Item": "NVIDIA H100", "vRAM": 80,  "vCPUs": 26, "RAM_GB": 250, "Price_per_Hour": 2.90},
    ]

    # Logging
    LOG_DIR = BASE_DIR / "logs"
    LOG_FILE = LOG_DIR / "app.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
