"""Действия приложения: то, что стоит за маршрутами сервера. Вынесено отдельно, чтобы проверять тестами без сети."""
import json, shutil, time
from pathlib import Path
import numpy as np

from .config import Settings
from .store import Store
from .jobs import JobQueue
from .g1 import prepare as PREP, synth as SYNTH, threshold as THR
from .g1.models import get_model, list_models
from .g2 import contract as G2C, volume as VOL, candidates as CAND, phantom as PH


class AppError(Exception):
    """Ошибка, которую показываем пользователю (код ответа 400/404/409/422)."""
    def __init__(self, msg, code=400):
        super().__init__(msg); self.code = code


def _date(ts):
    return time.strftime("%d.%m.%Y", time.localtime(ts))


class Service:
    def __init__(self, settings: Settings):
        self.s = settings
        self.store = Store(settings.var_dir / "epiapp.sqlite")
        self.jobs = JobQueue()
        self.g2 = G2C.get_adapter(settings)
        self._cand_cache = {}

    def warm_demo(self):
        """Демо-режим: синтетические записи и снимки готовятся заранее в фоне, чтобы первый показ не ждал."""
        import threading
        def work():
            try:
                d = self.s.var_dir / "demo"; d.mkdir(exist_ok=True)
                for kind in SYNTH.DEMO:
                    if not (d / f"g1_{kind}.edf").exists():
                        tmp = d / f"g1_{kind}.tmp"; SYNTH.make_demo_edf(str(tmp), kind); tmp.replace(d / f"g1_{kind}.edf")
                for kind in ("ok", "flair5"):
                    G2C.DemoAdapter(self.s).demo_pack(kind)
            except Exception as e:  # noqa: BLE001
                print("демо-данные не подготовлены заранее:", e)
        threading.Thread(target=work, daemon=True).start()

    # ================================================================= общее
    def meta(self):
        m = self.model()
        ok, why = m.available()
        g2ok, g2why = self.g2.available()
        return dict(mode=self.s.mode, demo=self.s.demo, clinic=self.s.clinic,
                    g1=dict(model=dict(id=m.id, title=m.title, card=m.card, available=ok, reason=why), models=list_models(self.s),
                            threshold=self.threshold_info(), notch_hz=self.s.notch_hz),
                    g2=dict(adapter=self.g2.id, card=self.g2.card, available=g2ok, reason=g2why, upload=self.g2.id != "demo"))

    def model(self):
        return get_model(self.s)

    def job(self, jid):
        j = self.jobs.get(jid)
        if not j:
            raise AppError("задача не найдена", 404)
        return j.as_dict()

    def _study(self, sid, kind):
        st = self.store.study(sid)
        if not st or st["kind"] != kind:
            raise AppError("исследование не найдено", 404)
        return st

    # ================================================================= Г1: порог
    def threshold_info(self):
        """Действующий порог для текущей модели: настроенный под клинику или порог модели по умолчанию (с открытых данных)."""
        m = self.model()
        t = self.store.threshold(m.id, self.s.demo)
        if t:
            d = t["details"] or {}
            src = {"youden": f"по {d.get('n_labeled', '?')} размеченным записям вашей клиники (индекс Юдена)",
                   "sens80": f"по {d.get('n_labeled', '?')} размеченным записям вашей клиники (чувствительность не ниже 80 %)",
                   "median": f"по медиане оценок {d.get('n_total', '?')} записей вашей клиники (без разметки)",
                   "manual": "вручную"}[t["method"]]
            return dict(value=t["value"], method=t["method"], source=src, date=_date(t["created"]), clinic=True, details=d)
        return dict(value=float(m.card["default_threshold"]), method="default", clinic=False, date="",
                    source="не настроен под клинику — используется порог модели: " + m.card["default_threshold_source"], details={})

    def clinic(self):
        m = self.model()
        recs = self.store.clinic_records(m.id, self.s.demo)
        prop = THR.propose([r["score"] for r in recs], [r["label"] for r in recs])
        return dict(model=dict(id=m.id, title=m.title), records=recs, proposal=prop, threshold=self.threshold_info(), demo=self.s.demo,
                    need=dict(labeled=THR.MIN_LABELED, per_class=THR.MIN_PER_CLASS, unlabeled=THR.MIN_UNLABELED))

    def clinic_demo_fill(self):
        """Демо-набор «записей клиники»: 20 размеченных и 44 без разметки. Это выдуманные оценки (без сигнала), только для демо-режима."""
        if not self.s.demo:
            raise AppError("демо-набор доступен только в демо-режиме")
        m = self.model()
        self.store.clear_clinic(m.id, True)
        r = np.random.default_rng(3)
        pos = list(0.55 + r.random(10) * 0.40); neg = list(0.08 + r.random(10) * 0.45)
        pos[3], neg[4] = 0.42, 0.52                                       # два пересекающихся случая, как в макете
        for i, v in enumerate(pos):
            self.store.add_clinic_record(m.id, f"демо_клиника_Э{i + 1:02d}.edf", v, 1, "demo", demo=True)
        for i, v in enumerate(neg):
            self.store.add_clinic_record(m.id, f"демо_клиника_Н{i + 1:02d}.edf", v, 0, "demo", demo=True)
        for i, v in enumerate(np.clip(0.39 + (r.random(44) + r.random(44) + r.random(44) - 1.5) * 0.45, 0.02, 0.98)):
            self.store.add_clinic_record(m.id, f"демо_клиника_{i + 1:02d}.edf", float(v), None, "demo", demo=True)
        return self.clinic()

    def clinic_set_threshold(self, method, value=None):
        m = self.model()
        recs = self.store.clinic_records(m.id, self.s.demo)
        prop = THR.propose([r["score"] for r in recs], [r["label"] for r in recs])
        det = dict(n_total=len(recs), n_labeled=prop["labeled"]["n"] if prop["labeled"] else 0, model=m.id)
        if method in ("youden", "sens80"):
            if not prop["labeled"]:
                raise AppError("нет размеченных записей обоих классов", 422)
            v = prop["labeled"][method]["threshold"]
            det.update({k: prop["labeled"][method][k] for k in ("tp", "tn", "fp", "fn", "sensitivity", "specificity")})
        elif method == "median":
            if not prop["unlabeled"]:
                raise AppError("нет записей клиники", 422)
            v = prop["unlabeled"]["threshold"]
        elif method == "manual":
            if value is None or not 0 <= float(value) <= 1:
                raise AppError("порог должен быть числом от 0 до 1", 422)
            v = float(value)
        else:
            raise AppError("неизвестный способ: " + str(method), 422)
        self.store.save_threshold(m.id, method, v, self.s.demo, det)
        return self.threshold_info()

    def clinic_reset(self, records=False):
        m = self.model()
        if records:
            self.store.clear_clinic(m.id, self.s.demo)
        else:
            self.store.run("DELETE FROM thresholds WHERE model_id = ? AND demo = ?", (m.id, int(self.s.demo)))
        return self.clinic()

    def clinic_label(self, rid, label):
        if label not in (0, 1, None):
            raise AppError("метка: 1 (эпилепсия), 0 (без эпилепсии) или пусто", 422)
        self.store.set_label(rid, label)

    # ================================================================= Г1: исследования
    def g1_ingest(self, edf_path, name, demo=False, code=""):
        """EDF -> подготовка и проверка входа -> исследование. Исходный EDF после подготовки удаляется (остаётся подготовленный сигнал)."""
        rec, check = PREP.prepare_edf(str(edf_path), self.s.notch_hz)
        sid = self.store.add_study("g1", name, demo or self.s.demo, check, status="checked" if check["can_run"] else "rejected", code=code)
        if rec is not None:
            PREP.save_record(rec, self.s.var_dir / "g1" / f"{sid}.npy")
        return self.g1_view(sid)

    def g1_demo(self, kind):
        if kind not in SYNTH.DEMO:
            raise AppError("демо-запись: ok или bad", 422)
        d = self.s.var_dir / "demo"; d.mkdir(exist_ok=True)
        p = d / f"g1_{kind}.edf"
        if not p.exists():
            SYNTH.make_demo_edf(str(p), kind)
        return self.g1_ingest(p, SYNTH.DEMO[kind]["file"], demo=True)

    def _g1_record(self, st):
        p = self.s.var_dir / "g1" / f"{st['id']}.npy"
        if not p.exists():
            raise AppError("подготовленный сигнал не найден", 404)
        return PREP.load_record(p, st["check"])

    def g1_view(self, sid):
        """Исследование Г1 для интерфейса: проверка, результат и вывод относительно действующего порога."""
        st = self._study(sid, "g1")
        out = dict(id=st["id"], name=st["name"], code=st["code"], created=st["created"], date=_date(st["created"]), demo=bool(st["demo"]),
                   status=st["status"], check=st["check"], decision=st["decision"], error=st["error"], result=None,
                   report=self.store.last_report_id(st["id"]))
        if st["result"]:
            r, m, t = dict(st["result"]), get_model(self.s, st["model_id"]), None
            if st["model_id"] == self.model().id:
                t = self.threshold_info()
            else:                                               # запись оценена другой моделью — её порог по умолчанию
                t = dict(value=float(m.card["default_threshold"]), method="default", clinic=False, date="", source="порог модели " + m.title, details={})
            sc = [p["score"] for p in r.get("parts", [])]
            r.update(threshold=t, above=bool(r["score"] >= t["value"]), model_title=m.title, model_card=m.card,
                     verdict=m.card["verdict_above"] if r["score"] >= t["value"] else m.card["verdict_below"],
                     spread=[min(sc), max(sc)] if sc else None)
            out["result"] = r
        return out

    def g1_list(self):
        out = []
        for st in self.store.studies("g1"):
            c = (st["check"] or {}).get("summary", {})
            out.append(dict(id=st["id"], name=st["name"], date=_date(st["created"]), created=st["created"], status=st["status"], demo=bool(st["demo"]),
                            dur_text=c.get("dur_text", ""), low_quality=(st["check"] or {}).get("low_quality", False),
                            score=(st["result"] or {}).get("score"), decision=st["decision"], report=self.store.last_report_id(st["id"])))
        return out

    def g1_signal(self, sid, t0, sec, montage, hp, lp):
        st = self._study(sid, "g1")
        rec = self._g1_record(st)
        sec = float(min(max(sec, 1), 60)); t0 = float(min(max(t0, 0), max(0, rec.dur_s - sec)))
        names, data, fs, bad = PREP.view_chunk(rec.x, t0, sec, montage, hp, lp, rec.flat)
        arts = [[max(a, t0) - t0, min(b, t0 + sec) - t0] for a, b in rec.artifacts if b > t0 and a < t0 + sec]
        return dict(names=names, fs=fs, t0=t0, sec=sec, dur=rec.dur_s, bad=bad, artifacts=arts, data=data.tolist(), unit=0.1)

    def g1_run(self, sid):
        st = self._study(sid, "g1")
        if not st["check"]["can_run"]:
            raise AppError("запись не прошла проверку входа — оценка невозможна", 409)
        m = self.model()
        ok, why = m.available()
        if not ok:
            raise AppError(f"модель «{m.title}» недоступна: {why}", 409)

        def work(job):
            if not m.loaded:
                job.progress(0, 0.02); m.load()
            out = m.predict(self._g1_record(st), job.progress)
            res = dict(score=out.score, parts=out.parts, components=out.components, n_windows=out.n_windows, seconds=out.seconds)
            self.store.update_study(sid, result=res, status="done", model_id=m.id)
            self.store.run("DELETE FROM clinic_records WHERE study_id = ?", (sid,))
            self.store.add_clinic_record(m.id, st["name"], out.score, None, "study", sid, demo=bool(st["demo"]) or self.s.demo,
                                         low_quality=st["check"]["low_quality"])
            return dict(study=sid)
        self.store.update_study(sid, status="running")
        return self.jobs.submit("Оценка записи ЭЭГ", m.card["stages"], work).as_dict()

    def clinic_add_record(self, edf_path, name, label):
        """Запись клиники для настройки порога: подготовка -> оценка моделью -> в список записей клиники (с меткой или без)."""
        m = self.model()
        ok, why = m.available()
        if not ok:
            raise AppError(f"модель «{m.title}» недоступна: {why}", 409)
        rec, check = PREP.prepare_edf(str(edf_path), self.s.notch_hz)
        if rec is None:
            bad = next((i for i in check["items"] if i["status"] == "bad"), {"note": "запись не подходит"})
            raise AppError(f"{name}: {bad['note']}", 422)

        def work(job):
            if not m.loaded:
                m.load()
            out = m.predict(rec, job.progress)
            rid = self.store.add_clinic_record(m.id, name, out.score, label, "calib", demo=self.s.demo, low_quality=check["low_quality"])
            return dict(record=rid, score=out.score, name=name)
        return self.jobs.submit(f"Запись клиники: {name}", m.card["stages"], work).as_dict()

    def decision(self, sid, kind, body):
        st = self._study(sid, kind)
        d = dict(st["decision"] or {}); d.update(body); d["time"] = time.time()
        self.store.update_study(sid, decision=d)
        return d

    # ================================================================= Г2
    def g2_demo(self, kind):
        if kind not in ("ok", "flair5"):
            raise AppError("демо-снимки: ok или flair5", 422)
        z = PH.ZOOM
        check = VOL.check_input(dict(zooms=[z, z, z], ndim=3), dict(zooms=[0.9, 0.9, 5.0] if kind == "flair5" else [z, z, z], ndim=3))
        sid = self.store.add_study("g2", "Демо-пациент" + (" (FLAIR 5 мм)" if kind == "flair5" else ""), True, check, files=dict(demo_kind=kind))
        return self.g2_view(sid)

    def g2_create(self, code=""):
        if self.g2.id == "demo":
            raise AppError("в демо-режиме модель МРТ не подключена: используйте демо-снимки", 409)
        return self.g2_view(self.store.add_study("g2", "T1 + FLAIR", False, None, files={}, status="upload", code=code))

    def g2_file(self, sid, kind, path, name):
        st = self._study(sid, "g2")
        if kind not in ("t1", "flair"):
            raise AppError("серия: t1 или flair", 422)
        try:
            info = VOL.header_info(path)
        except Exception as e:  # noqa: BLE001
            raise AppError(f"{name}: файл не читается как NIfTI ({str(e)[:120]}). Папку DICOM сначала преобразуйте в NIfTI (dcm2niix).", 422)
        files = dict(st["files"] or {}); files[kind] = dict(path=str(path), name=name, info=info)
        kw = dict(files=files)
        if "t1" in files and "flair" in files:
            kw.update(check=VOL.check_input(files["t1"]["info"], files["flair"]["info"]), status="checked",
                      name=f"{files['t1']['name']} + {files['flair']['name']}")
        self.store.update_study(sid, **kw)
        return self.g2_view(sid)

    def g2_view(self, sid):
        st = self._study(sid, "g2")
        f = st["files"] or {}
        return dict(id=st["id"], name=st["name"], code=st["code"], date=_date(st["created"]), created=st["created"], demo=bool(st["demo"]), status=st["status"],
                    check=st["check"], result=st["result"], decision=st["decision"], error=st["error"], card=self.g2.card,
                    files={k: v["name"] for k, v in f.items() if isinstance(v, dict)}, demo_kind=f.get("demo_kind"),
                    report=self.store.last_report_id(st["id"]))

    def g2_list(self):
        out = []
        for st in self.store.studies("g2"):
            r = st["result"] or {}
            out.append(dict(id=st["id"], name=st["name"], date=_date(st["created"]), created=st["created"], status=st["status"], demo=bool(st["demo"]),
                            decision=st["decision"], default_thr=r.get("default_thr"), report=self.store.last_report_id(st["id"]),
                            flair5=(st["check"] or {}).get("flair5", False), n_cand=len(r.get("candidates", [])) if r else None,
                            top=max([c["peak_prob"] for c in r.get("candidates", [])], default=None) if r else None))
        return out

    def g2_run(self, sid):
        st = self._study(sid, "g2")
        if not st["check"] or not st["check"]["can_run"]:
            raise AppError("снимки не прошли проверку — запуск невозможен", 409)
        ok, why = self.g2.available()
        if not ok:
            raise AppError("модель Г2 недоступна: " + why, 409)
        f = st["files"] or {}
        if not st["demo"] and self.g2.id == "demo":
            raise AppError("в демо-режиме модель МРТ не подключена", 409)
        work_dir = self.s.var_dir / "g2" / sid

        def work(job):
            r = self.g2.predict(f.get("t1", {}).get("path"), f.get("flair", {}).get("path"), work_dir, job.progress, demo_kind=f.get("demo_kind", "ok"))
            res = dict(info=r.info, candidates=r.candidates, default_thr=r.default_thr, qc=r.qc, seconds=r.seconds, candidates_from=r.candidates_from)
            self.store.update_study(sid, result=res, status="done", model_id=self.g2.id)
            return dict(study=sid)
        self.store.update_study(sid, status="running")
        return self.jobs.submit("Поиск очагов ФКД", self.g2.card["stages"], work).as_dict()

    def g2_thumb(self, sid, kind):
        """Уменьшенный аксиальный срез для экрана проверки (до запуска модели)."""
        st = self._study(sid, "g2")
        f = st["files"] or {}
        if kind not in ("t1", "flair"):
            raise AppError("серия: t1 или flair", 404)
        if f.get("demo_kind"):
            pack = VOL.load_pack(self.g2.demo_pack(f["demo_kind"]) if hasattr(self.g2, "demo_pack") else G2C.DemoAdapter(self.s).demo_pack(f["demo_kind"]))
            t = VOL.thumb_from_volume(pack[kind]); t["aspect"] = pack["t1"].shape[0] * pack["zooms"][0] / (pack["t1"].shape[1] * pack["zooms"][1])
            return dict(t, demo=True)
        if kind not in f:
            raise AppError("файл не загружен", 404)
        return dict(VOL.thumb_from_file(f[kind]["path"]), demo=False)

    def g2_volume_path(self, sid, name):
        self._study(sid, "g2")
        if name not in ("t1", "flair", "prob"):
            raise AppError("объём: t1, flair или prob", 404)
        p = self.s.var_dir / "g2" / sid / "pack" / f"{name}.u8"
        if not p.exists():
            raise AppError("объём ещё не готов", 404)
        return p

    def g2_candidates(self, sid, thr):
        """Кандидаты при рабочей точке thr. При рабочей точке модели и готовой таблице модели — она; иначе пересчёт приложением."""
        st = self._study(sid, "g2")
        r = st["result"]
        if not r:
            raise AppError("расчёт ещё не выполнен", 409)
        thr = float(thr)
        if abs(thr - r["default_thr"]) < 1e-6:
            return dict(thr=thr, candidates=r["candidates"], source=r.get("candidates_from", "app"))
        key = (sid, round(thr, 3))
        if key not in self._cand_cache:
            p = VOL.load_pack(self.s.var_dir / "g2" / sid / "pack")
            self._cand_cache[key] = CAND.cluster_table(p["prob"] / 255.0, thr, p["affine"], p["brain"])
        return dict(thr=thr, candidates=self._cand_cache[key], source="app")

    def cleanup_upload(self, path):
        try:
            Path(path).unlink()
        except OSError:
            pass
