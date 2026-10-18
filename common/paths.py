"""Где лежат данные. Данные и веса в репозиторий не попадают, путь задаётся на каждой машине свой."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def platform_name() -> str:
    if Path("/kaggle/input").exists():
        return "kaggle"
    return "windows" if os.name == "nt" else "linux"


def data_dir() -> Path:
    """Корень данных.

    Порядок: переменная окружения EPI_DATA_DIR -> /kaggle/input -> <репозиторий>/data (она в .gitignore).
    Windows (PowerShell):  $env:EPI_DATA_DIR = "D:\\epi_data"
    Ubuntu:                export EPI_DATA_DIR=/mnt/data/epi
    """
    env = os.environ.get("EPI_DATA_DIR")
    if env:
        return Path(env)
    if Path("/kaggle/input").exists():
        return Path("/kaggle/input")
    return REPO_ROOT / "data"


def weights_dir() -> Path:
    """Куда кладём скачанные веса моделей (вне git)."""
    env = os.environ.get("EPI_WEIGHTS_DIR")
    if env:
        return Path(env)
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/weights")
    return REPO_ROOT / "weights"
