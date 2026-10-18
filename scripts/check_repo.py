"""Проверка репозитория перед коммитом / в pull request.

Что проверяет:
  1. В git не попали данные и веса (по расширению и по папкам) — это ОШИБКА.
  2. Любой файл больше 5 МБ (кроме блокнотов) — ОШИБКА.
  3. Блокнот больше 10 МБ — ПРЕДУПРЕЖДЕНИЕ (выводы сохраняем, но следим за размером).
  4. Блокнот без сохранённых выводов (ни у одной ячейки нет outputs) — ПРЕДУПРЕЖДЕНИЕ.

Запуск:
  py scripts/check_repo.py --staged   # только то, что в индексе (используется в git-хуке)
  py scripts/check_repo.py --all      # все файлы под контролем git (используется в GitHub Actions)
Код возврата 1 — только при ошибках; предупреждения коммит не блокируют.
"""
import json
import subprocess
import sys
from pathlib import Path

NB_WARN_MB = 10
FILE_MAX_MB = 5

FORBIDDEN_EXT = {
    ".edf", ".bdf", ".gdf", ".set", ".fdt", ".fif", ".vhdr", ".vmrk", ".eeg",
    ".mat", ".h5", ".hdf5", ".npy", ".npz", ".parquet", ".feather", ".pkl",
    ".pickle", ".joblib", ".tfrecord",
    ".nii", ".mgz", ".mgh", ".dcm", ".mnc", ".nrrd", ".img", ".hdr",
    ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".bin", ".weights", ".keras",
}
FORBIDDEN_DIRS = {"data", "datasets", "raw", "weights", "checkpoints", "outputs"}

ROOT = Path(__file__).resolve().parents[1]


def git_files(staged: bool) -> list[Path]:
    cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"] if staged else ["git", "ls-files"]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True).stdout
    return [ROOT / line for line in out.splitlines() if line.strip()]


def is_forbidden(rel: Path) -> bool:
    name = rel.name.lower()
    if name.endswith(".nii.gz") or name.endswith(".fif.gz"):
        return True
    if rel.suffix.lower() in FORBIDDEN_EXT:
        return True
    # папки с данными в корне или в h1_eeg / h2_mri
    parts = rel.parts[:-1]
    return any(p in FORBIDDEN_DIRS for p in parts)


def main() -> int:
    staged = "--staged" in sys.argv
    errors, warnings = [], []
    for path in git_files(staged):
        if not path.exists():
            continue
        rel = path.relative_to(ROOT)
        size_mb = path.stat().st_size / 1024 / 1024
        if is_forbidden(rel):
            errors.append(f"{rel}: данные/веса в git запрещены")
            continue
        if path.suffix == ".ipynb":
            if size_mb > NB_WARN_MB:
                warnings.append(
                    f"{rel}: блокнот {size_mb:.1f} МБ (> {NB_WARN_MB} МБ). "
                    "Уменьшите картинки (dpi=100), не печатайте большие массивы, разбейте блокнот."
                )
            try:
                nb = json.loads(path.read_text(encoding="utf-8"))
                code = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
                if rel.parts[0] != "templates" and code and not any(c.get("outputs") for c in code):
                    warnings.append(f"{rel}: нет сохранённых выводов — блокнот является отчётом, выводы нужны.")
            except Exception as e:  # noqa: BLE001
                errors.append(f"{rel}: не читается как JSON ({e}) — вероятно, остались маркеры конфликта")
        elif size_mb > FILE_MAX_MB:
            errors.append(f"{rel}: файл {size_mb:.1f} МБ (> {FILE_MAX_MB} МБ)")

    for w in warnings:
        print("ПРЕДУПРЕЖДЕНИЕ:", w)
    for e in errors:
        print("ОШИБКА:", e)
    if errors:
        print("\nКоммит остановлен. Уберите файлы из индекса: git restore --staged <файл>")
        return 1
    print("Проверка пройдена." if not warnings else "Проверка пройдена (есть предупреждения).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
