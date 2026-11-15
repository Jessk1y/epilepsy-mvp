"""Запуск приложения из любой папки:  py app/run.py --demo   |   python app/run.py --real   (то же, что python -m epiapp)."""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from epiapp.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()
