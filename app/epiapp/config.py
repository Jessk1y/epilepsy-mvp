"""Настройки приложения. Всё задаётся переменными окружения EPI_* (или ключами запуска, см. __main__.py)."""
import os
from dataclasses import dataclass, field
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]            # epilepsy-mvp/app
REPO_DIR = APP_DIR.parent                                 # epilepsy-mvp
WEB_DIR = APP_DIR / "web"


def _env(name, default=""):
    return os.environ.get(name, default)


def _path(name, default):
    return Path(os.path.expanduser(_env(name, str(default)))).resolve()


@dataclass
class Settings:
    mode: str = "demo"                                    # demo | real
    var_dir: Path = APP_DIR / "var"                       # загруженные файлы, база, отчёты — вне git
    clinic: str = "Клиника"                               # название в отчёте
    # --- Г1
    g1_model: str = "demo"                                # demo | reve_dino | tuep_pair
    notch_hz: float = 50.0                                # режекция сети: 50 Гц (Россия), у TUH было 60
    device: str = "cuda:0"
    reve_dir: Path = Path("~/data/weights/h1-reve-v4")
    dino_dir: Path = Path("~/data/weights/h1-eegdino-v2")
    dino_src: str = ""                                    # папка с models/ авторов EEG-DINO, если нет доступа к HF
    tuep_final_dir: Path = Path("~/data/tuep_work/ft/final")
    # --- Г2
    g2_adapter: str = "demo"                              # demo | external
    g2_cmd: str = ""                                      # команда по контракту (см. epiapp/g2/contract.py)
    g2_title: str = "MS-DSA-NET"
    g2_card: str = ""                                     # путь к паспорту модели Г2 (json); пусто — встроенный
    extra: dict = field(default_factory=dict)

    @property
    def demo(self):
        return self.mode == "demo"


def load_settings(mode=None):
    mode = mode or _env("EPI_MODE", "demo")
    assert mode in ("demo", "real"), "EPI_MODE: demo или real"
    s = Settings(mode=mode)
    s.var_dir = _path("EPI_VAR_DIR", s.var_dir)
    s.clinic = _env("EPI_CLINIC", s.clinic)
    s.g1_model = _env("EPI_G1_MODEL", "demo" if mode == "demo" else "reve_dino")
    s.notch_hz = float(_env("EPI_NOTCH_HZ", "50"))
    s.device = _env("EPI_DEVICE", s.device)
    s.reve_dir = _path("EPI_REVE_DIR", s.reve_dir)
    s.dino_dir = _path("EPI_DINO_DIR", s.dino_dir)
    s.dino_src = _env("EPI_DINO_SRC", "")
    s.tuep_final_dir = _path("EPI_TUEP_FINAL_DIR", s.tuep_final_dir)
    s.g2_cmd = _env("EPI_G2_CMD", "")
    s.g2_adapter = _env("EPI_G2_ADAPTER", "external" if (mode == "real" and s.g2_cmd) else "demo")
    s.g2_title = _env("EPI_G2_TITLE", s.g2_title)
    s.g2_card = _env("EPI_G2_CARD", "")
    for sub in ("uploads", "g1", "g2", "reports"):
        (s.var_dir / sub).mkdir(parents=True, exist_ok=True)
    return s
