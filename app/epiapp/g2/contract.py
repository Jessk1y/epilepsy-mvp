"""Контракт модели Г2 и адаптеры.

Модели Г2 живут в своих окружениях (MS-DSA-NET: monai + FSL; nnU-Net: nnunetv2), поэтому приложение не импортирует их код,
а вызывает ВНЕШНЮЮ КОМАНДУ и читает файлы. Любая модель, которая выполняет этот контракт, подключается без переделки интерфейса.

    <команда> --t1 T1.nii.gz --flair FLAIR.nii.gz --out ПАПКА

    ПАПКА/prob.nii.gz       обязательно. Карта вероятности ФКД, значения 0..1, с правильной матрицей положения (affine).
    ПАПКА/candidates.csv    желательно. Столбцы: peak_prob, volume_ml, x, y, z — кандидаты при рабочей точке модели (мм, мир карты).
    ПАПКА/qc.json           желательно. {"t1_mm": 1.0, "flair_mm": 1.0, "mni_corr": 0.6, "flags": ["…"], "ok": true}
    ПАПКА/t1.nii.gz         необязательно. T1 и FLAIR в сетке карты (после совмещения) — тогда наложение точное.
    ПАПКА/flair.nii.gz        Если их нет, приложение берёт исходные файлы и совмещает по матрицам положения.
    ПАПКА/brain_mask.nii.gz необязательно. Маска мозга: кандидаты вне мозга отбрасываются.
    код возврата 0 — успех; иначе текст ошибки из stderr показывается врачу.

Команда задаётся настройкой EPI_G2_CMD (например: «~/venvs/mri/bin/python ~/epilepsy-mvp/app/tools/g2_msdsa_cli.py»).
"""
import csv, json, os, shlex, shutil, subprocess, time
from dataclasses import dataclass, field
from pathlib import Path

from . import candidates as CAND, phantom, volume as VOL

NB = "h2_mri/notebooks/ms_dsa_net_stage2b_cv.ipynb"
# Открытые данные, MS-DSA-NET после дообучения (этап 2b), источник — блокнот NB, раздел 9 «Рабочая точка».
# sens — доля пациентов с ФКД, у которых очаг найден (34 пациента IDEAS, каждый — моделью, которая его не видела);
# fp — ложных находок на человека без ФКД, итоговый ансамбль, отчётная половина отложенных (23 человека); clean — доля людей без ложных находок.
MSDSA_TABLE = [dict(thr=0.3, sens=0.882, fp=5.261, clean=0.348), dict(thr=0.4, sens=0.824, fp=2.565, clean=0.391),
               dict(thr=0.5, sens=0.706, fp=0.870, clean=0.435), dict(thr=0.6, sens=0.706, fp=0.391, clean=0.739),
               dict(thr=0.7, sens=0.676, fp=0.261, clean=0.826), dict(thr=0.8, sens=0.588, fp=0.174, clean=0.870),
               dict(thr=0.9, sens=0.559, fp=0.130, clean=0.870), dict(thr=0.95, sens=0.471, fp=0.130, clean=0.870)]
MSDSA_CARD = dict(
    title="MS-DSA-NET", demo=False,
    trained_on="Bonn ds004199 (85 ФКД II типа, 40 здоровых) + IDEAS ds005602 (34 ФКД, 60 без ФКД), 5 фолдов; ансамбль 5 моделей, Nyúl + TTA",
    default_thr=0.9, table=MSDSA_TABLE, table_source=NB + ", раздел 9",
    table_note="Открытые данные: находимость — 34 пациента с ФКД из IDEAS (проверка по фолдам); ложные находки — 23 отложенных человека без ФКД.",
    flair5_note="При FLAIR со срезами 5 мм очаг найден у 47 % пациентов против 94 % при срезах до 3,5 мм (IDEAS, порог 0,5; " + NB + ", раздел 8).",
    stages=["Приведение снимков к стандартному виду", "Совмещение T1 и FLAIR", "Карта вероятности", "Выделение кандидатов"],
    eta="Время зависит от модели и машины.",
    limits=["Возможны ложные находки и пропуски очагов.", "Модель обучалась на ФКД; другую патологию она не ищет."])
DEMO_CARD = dict(MSDSA_CARD, title="Демо (синтетическая карта)", demo=True, eta="Несколько секунд.",
                 trained_on="не обучалась: снимки и карта — схема",
                 table_note=MSDSA_CARD["table_note"] + " В демо-режиме таблица показана для примера: к синтетическим снимкам она не относится.")
NO_NUMBERS_CARD = dict(MSDSA_CARD, title="Модель Г2", table=[], table_source="", table_note="Для этой модели в репозитории нет таблицы рабочих точек.",
                       flair5_note="Толстые срезы FLAIR снижают находимость очагов.", default_thr=0.5, trained_on="—")


@dataclass
class G2Result:
    pack_dir: str                                  # папка с объёмами для просмотра (volume.save_pack)
    info: dict                                     # сетка: shape, zooms, affine
    candidates: list = field(default_factory=list)  # при рабочей точке по умолчанию
    default_thr: float = 0.5
    qc: dict = field(default_factory=dict)
    seconds: float = 0.0
    candidates_from: str = "app"                   # app — посчитаны приложением; model — таблица модели


class G2Adapter:
    id = "base"

    def __init__(self, settings):
        self.s = settings
        self.card = self._card()

    def _card(self):
        return dict(NO_NUMBERS_CARD)

    def available(self):
        return True, ""

    def predict(self, t1, flair, workdir, progress=lambda stage, frac: None, demo_kind="ok") -> G2Result:
        raise NotImplementedError


class DemoAdapter(G2Adapter):
    """Синтетические снимки и карта. Файлы врача в демо-режиме не обрабатываются моделью."""
    id = "demo"

    def _card(self):
        return dict(DEMO_CARD)

    def demo_pack(self, demo_kind):
        """Синтетические снимки создаются один раз и лежат в рабочем каталоге (var/demo)."""
        d = Path(self.s.var_dir) / "demo" / f"g2_{demo_kind}"
        if not (d / "info.json").exists():
            ph = phantom.make_phantom(thick_flair=demo_kind == "flair5")
            VOL.save_pack(VOL.pack_from_arrays(ph["t1"], ph["flair"], ph["prob"], ph["affine"], ph["brain"]), d)
        return d

    def predict(self, t1, flair, workdir, progress=lambda stage, frac: None, demo_kind="ok"):
        t0 = time.time()
        progress(0, 0.1)
        src = self.demo_pack(demo_kind)
        for k, fr in ((1, 0.4), (2, 0.7)):
            progress(k, fr); time.sleep(0.5)
        dst = Path(workdir) / "pack"
        shutil.copytree(src, dst, dirs_exist_ok=True)
        pack = VOL.load_pack(dst)
        progress(3, 0.9)
        thr = self.card["default_thr"]
        cands = CAND.cluster_table(pack["prob"] / 255.0, thr, pack["affine"], pack["brain"])
        return G2Result(str(dst), pack["info"], cands, thr, dict(ok=demo_kind != "flair5", flags=[]), round(time.time() - t0, 1))


class ExternalAdapter(G2Adapter):
    """Любая модель по контракту «команда + файлы» (см. описание модуля)."""
    id = "external"
    TIMEOUT_S = 6 * 3600

    def _card(self):
        if self.s.g2_card:
            return dict(NO_NUMBERS_CARD, **json.loads(Path(self.s.g2_card).read_text(encoding="utf-8")))
        if "ms-dsa" in self.s.g2_title.lower() or "msdsa" in self.s.g2_title.lower():
            return dict(MSDSA_CARD, title=self.s.g2_title)
        return dict(NO_NUMBERS_CARD, title=self.s.g2_title)

    def available(self):
        if not self.s.g2_cmd:
            return False, "не задана команда модели Г2 (настройка EPI_G2_CMD)"
        return True, ""

    def predict(self, t1, flair, workdir, progress=lambda stage, frac: None, demo_kind="ok"):
        t0 = time.time()
        out = Path(workdir) / "model_out"; out.mkdir(parents=True, exist_ok=True)
        cmd = shlex.split(self.s.g2_cmd) + ["--t1", str(t1), "--flair", str(flair), "--out", str(out)]
        progress(0, 0.05)
        log = open(Path(workdir) / "model.log", "w", encoding="utf-8")
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                             env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
        while p.poll() is None:                              # ход внешней команды неизвестен: показываем этап по файлу progress.txt, если модель его пишет
            try:
                st, fr = (out / "progress.txt").read_text().split()[:2]; progress(int(st), float(fr))
            except Exception:  # noqa: BLE001
                pass
            if time.time() - t0 > self.TIMEOUT_S:
                p.kill(); raise RuntimeError("модель Г2 не ответила за отведённое время")
            time.sleep(1.0)
        err = p.stderr.read(); log.write(err); log.close()
        if p.returncode != 0:
            raise RuntimeError("модель Г2 завершилась с ошибкой: " + (err.strip().splitlines() or ["нет текста ошибки"])[-1][:300])
        if not (out / "prob.nii.gz").exists():
            raise RuntimeError("модель Г2 не создала prob.nii.gz (контракт: epiapp/g2/contract.py)")
        progress(3, 0.9)
        pick = lambda name, fallback: out / name if (out / name).exists() else fallback
        brain = out / "brain_mask.nii.gz"
        pack = VOL.build_pack(pick("t1.nii.gz", t1), pick("flair.nii.gz", flair), out / "prob.nii.gz", brain if brain.exists() else None)
        info = VOL.save_pack(pack, Path(workdir) / "pack")
        qc = json.loads((out / "qc.json").read_text(encoding="utf-8")) if (out / "qc.json").exists() else {}
        thr = float(self.card.get("default_thr", 0.5))
        cands = CAND.cluster_table(pack["prob"] / 255.0, thr, pack["affine"], pack["brain"])
        src = "app"
        if (out / "candidates.csv").exists():                # таблица модели — главнее: в ней постобработка самой модели
            import numpy as np
            rows = list(csv.DictReader(open(out / "candidates.csv", encoding="utf-8")))
            inv, cands, src = np.linalg.inv(pack["affine"]), [], "model"
            mid = float((pack["affine"] @ np.array([*(np.array(pack["t1"].shape) / 2), 1.0]))[0])
            for i, r in enumerate(sorted(rows, key=lambda r: -float(r["peak_prob"]))):
                xyz = [float(r["x"]), float(r["y"]), float(r["z"])]
                ijk = [int(round(v)) for v in (inv @ np.array([*xyz, 1.0]))[:3]]
                ijk = [min(max(v, 0), s - 1) for v, s in zip(ijk, pack["t1"].shape)]
                cands.append(dict(n=i + 1, peak_prob=float(r["peak_prob"]), volume_ml=float(r["volume_ml"]), xyz=xyz, ijk=ijk,
                                  hemi="левое" if xyz[0] < mid else "правое"))
        return G2Result(str(Path(workdir) / "pack"), info, cands, thr, qc, round(time.time() - t0, 1), src)


def get_adapter(settings):
    return {"demo": DemoAdapter, "external": ExternalAdapter}[settings.g2_adapter](settings)
