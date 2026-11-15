"""Запуск:  py -m epiapp --demo        (ноутбук: синтетика, без моделей)
            python -m epiapp --real       (рабочая машина: настоящие модели; настройки — переменные EPI_*, см. README)"""
import argparse, os


def main():
    ap = argparse.ArgumentParser(prog="epiapp", description="ЭпиАссистент: второе мнение для врача (ЭЭГ и МРТ)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--demo", action="store_true", help="демо-режим: синтетические данные, без моделей и видеокарты")
    g.add_argument("--real", action="store_true", help="боевой режим: настоящие модели")
    ap.add_argument("--host", default="127.0.0.1", help="по умолчанию только эта машина; доступ с другой — проброс порта по SSH")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--g1-model", default="", help="demo | reve_dino | tuep_pair (иначе EPI_G1_MODEL)")
    a = ap.parse_args()
    if a.g1_model:
        os.environ["EPI_G1_MODEL"] = a.g1_model
    import uvicorn
    from .config import load_settings
    from .server import create_app
    s = load_settings("real" if a.real else "demo" if a.demo else None)
    print(f"ЭпиАссистент · режим: {'ДЕМО' if s.demo else 'боевой'} · модель Г1: {s.g1_model} · Г2: {s.g2_adapter} · данные: {s.var_dir}")
    print(f"Откройте в браузере: http://{a.host}:{a.port}")
    uvicorn.run(create_app(s), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
