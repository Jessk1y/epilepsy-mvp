"""Отчёт для врача. Выдаётся только с заключением врача; содержимое собирает сервер из сохранённого исследования
(а не из того, что прислал браузер), снимок отчёта хранится в базе. Печать и PDF — средствами браузера."""
import html, time

from .service import AppError

MIN_CONCLUSION = 3
DISCLAIMER = "Результат программы — второе мнение и не является диагнозом. Окончательное решение принимает врач."


def _f(x, d=2):
    return f"{float(x):.{d}f}".replace(".", ",")


def issue(svc, study_id, doctor, conclusion, thr=None):
    """Выдать отчёт. Без заключения врача — ошибка 422."""
    conclusion, doctor = (conclusion or "").strip(), (doctor or "").strip()
    if len(conclusion) < MIN_CONCLUSION:
        raise AppError("Без заключения врача отчёт не выдаётся: заполните поле «Заключение врача».", 422)
    st = svc.store.study(study_id)
    if not st:
        raise AppError("исследование не найдено", 404)
    if not st["result"]:
        raise AppError("расчёт ещё не выполнен — отчёт выдать нельзя", 409)
    rows = g1_rows(svc, st) if st["kind"] == "g1" else g2_rows(svc, st, thr)
    payload = dict(kind="ЭЭГ" if st["kind"] == "g1" else "МРТ", file=st["name"], code=st["code"], demo=bool(st["demo"]) or svc.s.demo,
                   clinic=svc.s.clinic, rows=rows, date=time.strftime("%d.%m.%Y"), disclaimer=DISCLAIMER)
    rid = svc.store.add_report(study_id, st["kind"], doctor, conclusion, payload)
    return svc.store.report(rid)


def g1_rows(svc, st):
    v = svc.g1_view(st["id"])
    c, r = v["check"], v["result"]
    s, t = c["summary"], r["threshold"]
    rows = [("Длительность / каналы", f"{s['dur_text']} · {s['n_good']} из 19" + (f" (без сигнала: {', '.join(c['flat'])})" if c["flat"] else "")),
            ("Артефакты", f"≈ {s['art_pct']} % записи"),
            ("Оценка модели", f"<b>{_f(r['score'])}</b> (порог {_f(t['value'])}) — {'выше' if r['above'] else 'ниже'} порога"),
            ("Порог", html.escape(t["source"]) + (f", {t['date']}" if t["date"] else "")),
            ("Вывод модели", html.escape(r["verdict"]))]
    if r.get("spread"):
        rows.append(("Устойчивость", f"оценки по частям записи — от {_f(r['spread'][0])} до {_f(r['spread'][1])}"))
    if c["low_quality"]:
        rows.append(("Качество входа", "ниже нормы: " + ", ".join(c.get("low_reasons", [])) + " — доверие к оценке снижено"))
    rows.append(("Модель", html.escape(r["model_title"]) + (" — ДЕМО, оценка не имеет клинического смысла" if r["model_card"].get("demo") else "")))
    rows.append(("Ограничения", "Модель не локализует события: оценка относится ко всей записи. " + " ".join(html.escape(x) for x in r["model_card"]["limits"])))
    return rows


def g2_rows(svc, st, thr):
    r, c, card = st["result"], st["check"], svc.g2.card
    thr = float(thr if thr is not None else r["default_thr"])
    cands = svc.g2_candidates(st["id"], thr)["candidates"]
    s = c["summary"]
    rows = [("Снимки", f"T1 {s['t1']} · FLAIR {s['flair']}" + (" (толстые срезы — низкое качество)" if c.get("flair5") else ""))]
    if cands:
        txt = "<br>".join(f"№{k['n']}: вероятность {_f(k['peak_prob'])}, {_f(k['volume_ml'], 1)} см³, "
                          f"({'; '.join(f'{round(x):+d}'.replace('-', '−') for x in k['xyz'])}) мм, {k['hemi']} полушарие" for k in cands)
    else:
        txt = "нет"
    rows.append(("Кандидаты выше рабочей точки", txt))
    row = next((x for x in card.get("table", []) if abs(x["thr"] - thr) < 1e-6), None)
    op = f"порог {_f(thr)}"
    if row and not card.get("demo"):
        op += f" — на открытых данных: очаг найден у {round(100 * row['sens'])} % пациентов с ФКД, {_f(row['fp'])} ложных находок на человека без ФКД"
    rows.append(("Рабочая точка", op))
    if c.get("flair5"):
        rows.append(("Качество входа", "FLAIR со срезами 5 мм: часть очагов могла быть пропущена. " + html.escape(card.get("flair5_note", ""))))
    flags = (r.get("qc") or {}).get("flags") or []
    if flags:
        rows.append(("Замечания модели к снимкам", html.escape("; ".join(map(str, flags)))))
    rows.append(("Модель", html.escape(card["title"]) + (" — ДЕМО, снимки и карта — схема" if card.get("demo") else "")))
    rows.append(("Ограничения", " ".join(html.escape(x) for x in card["limits"])))
    return rows


PAGE = """<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Отчёт для врача · {kind} · {date}</title>
<style>
body{{margin:0;font:15px/1.55 system-ui,"Segoe UI",sans-serif;background:#f5f3fa;color:#2b3346}}
.report{{max-width:760px;margin:24px auto;background:#fff;border:1px solid #e7e3f0;padding:36px 44px;box-shadow:0 2px 10px rgba(0,0,0,.05)}}
h1{{font-size:20px;margin:0 0 4px}}h3{{margin:18px 0 8px;font-size:16px}}.small{{font-size:13px;color:#6b7387}}
dl{{display:grid;grid-template-columns:220px 1fr;gap:6px 12px;margin:14px 0}}dt{{color:#6b7387}}dd{{margin:0}}
.concl{{border:1px solid #e7e3f0;border-radius:8px;padding:10px 12px;white-space:pre-wrap;min-height:60px}}
.demo{{background:#fff3cd;color:#7a5600;border:1px solid #efd27f;border-radius:4px;padding:2px 8px;font-size:12px;font-weight:700}}
.sign{{display:flex;justify-content:space-between;margin-top:26px}}.sign span{{border-top:1px solid #2b3346;padding-top:4px;min-width:220px;text-align:center;font-size:13px;color:#6b7387}}
.btn{{font:inherit;border:1px solid #5f72c9;background:#5f72c9;color:#fff;border-radius:12px;padding:9px 16px;cursor:pointer;font-weight:600}}
.bar{{max-width:760px;margin:16px auto 0;display:flex;justify-content:flex-end;gap:10px}}
@media print{{.noprint{{display:none!important}}body{{background:#fff}}.report{{border:0;box-shadow:none;margin:0 auto}}}}
</style></head><body>
<div class="bar noprint"><button class="btn" onclick="print()">Печать / PDF</button></div>
<div class="report"><div class="small">ЭпиАссистент · {kind} · {clinic} {demo}</div><h1>Отчёт для врача</h1>
<div class="small">{who} · {file} · {date} · № {rid}</div>
<dl>{rows}</dl>
<h3>Заключение врача</h3><div class="concl">{conclusion}</div>
<div class="sign"><span>{doctor}</span><span>подпись</span></div>
<p class="small">{disclaimer}</p></div></body></html>"""


def render(rep):
    p, e = rep["payload"], html.escape
    demo = '<span class="demo">ДЕМО · данные выдуманы</span>' if p["demo"] else ""
    who = "Пациент: Демо-пациент (данные выдуманы)" if p["demo"] else ("Код пациента: " + e(p["code"]) if p["code"] else "Код пациента: не указан")
    rows = "".join(f"<dt>{e(k)}</dt><dd>{v}</dd>" for k, v in p["rows"])
    return PAGE.format(kind=e(p["kind"]), date=e(p["date"]), clinic=e(p["clinic"]), demo=demo, who=who, file=e(p["file"]), rid=e(rep["id"]), rows=rows,
                       conclusion=e(rep["conclusion"]), doctor="врач: " + e(rep["doctor"]) if rep["doctor"] else "врач", disclaimer=e(p["disclaimer"]))
