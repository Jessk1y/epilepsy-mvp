"""Г1: рутинная ЭЭГ -> вероятность эпилепсии по записи."""
import sys
from ..config import REPO_DIR

# Код Г1 не копируется, а подключается из h1_eeg (см. app/README.md)
for _p in (REPO_DIR / "h1_eeg" / "tuep", REPO_DIR / "h1_eeg" / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
