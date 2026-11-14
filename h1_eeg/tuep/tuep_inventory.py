"""Задача 1.3 роадмапа: опись TUEP — «пациент / сеанс / файл / группа / длительность / частота / каналы / монтаж / пересечение с TUSZ».

    py tuep_inventory.py --root /data/tuep/v3.1.0 --out /data/tuep_work/inventory

Читает только заголовки EDF (сигнал не читает), на Ubuntu это минуты. Выход:
  inventory_files.csv     — строка на файл
  inventory_sessions.csv  — строка на сеанс (сумма по файлам, пересечение с TUSZ, число отчётов .txt, таблица метаданных)
  inventory_summary.md    — сводка (пациенты, сеансы, часы, частоты, монтажи, наборы каналов, неполные записи)
  metadata_columns.txt    — столбцы metadata_v00r.xlsx (формат таблицы не проверен до получения данных)
"""
import argparse, glob, os, re, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tuep_common as C


def find_tusz_list(root):
    hits = [p for p in glob.glob(os.path.join(root, "**", "*.list"), recursive=True) if "tusz" in os.path.basename(p).lower()]
    return sorted(hits)


def parse_tusz_keys(list_path):
    """Из строк списка достаём пары (пациент, сеанс). Формат строки до данных не известен — берём любые «aaaaaaaa … sNNN»."""
    keys = set()
    for line in open(list_path, encoding="utf-8", errors="replace"):
        m = re.search(r"([a-z]{8})[/_\\ ].*?(s\d{3})", line.strip().lower())
        if m:
            keys.add((m.group(1), m.group(2)))
    return keys


def load_metadata(root, out_dir):
    # в DOCS две версии: metadata_v00.xlsx (585 сеансов) и metadata_v00r.xlsx (698, полная) — берём «r»
    hits = sorted(glob.glob(os.path.join(root, "**", "metadata*.xlsx"), recursive=True), key=lambda p: ("v00r" not in os.path.basename(p), p))
    if not hits:
        return None, None
    x = pd.read_excel(hits[0], sheet_name=None)
    with open(os.path.join(out_dir, "metadata_columns.txt"), "w", encoding="utf-8") as f:
        f.write(f"файл: {hits[0]}\n")
        for name, df in x.items():
            f.write(f"\nлист «{name}»: {df.shape}\nстолбцы: {list(df.columns)}\n{df.head(5).to_string()}\n")
    return hits[0], x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="папка v3.1.0 (внутри — edf/…)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(a.root, "**", "*.edf"), recursive=True))
    print("EDF-файлов:", len(paths))
    assert paths, "EDF не найдены — проверьте --root"

    def one(p):
        r = C.parse_path(p); r["path"] = os.path.relpath(p, a.root)
        r["year"] = C.session_year(r["session"])
        r["n_seiz"], r["seiz_s"] = C.read_seizures(p)
        try:
            fs, n, labels = C.read_header(p)
            idx = C.channel_index(labels)
            r.update(sfreq=fs, n_samples=n, dur_s=n / fs, n_channels=len(labels), has19=idx is not None, ref=C.ref_kind(labels),
                     channels="|".join(labels), size_mb=os.path.getsize(p) / 1e6, error="")
        except Exception as e:
            r.update(sfreq=float("nan"), n_samples=0, dur_s=0.0, n_channels=0, has19=False, ref="", channels="", size_mb=os.path.getsize(p) / 1e6,
                     error=f"{type(e).__name__}: {str(e)[:120]}")
        return r

    with ThreadPoolExecutor(a.threads) as ex:
        F = pd.DataFrame(list(ex.map(one, paths)))

    # пересечение с TUSZ
    lists = find_tusz_list(a.root)
    tusz = set()
    for l in lists:
        tusz |= parse_tusz_keys(l)
    F["skey"] = F.patient.astype(str) + "/" + F.session.astype(str)
    F["in_tusz"] = [(p, s[:4]) in tusz for p, s in zip(F.patient, F.session.astype(str))] if tusz else pd.NA

    # группа: из пути; если в пути нет — из таблицы метаданных (проверить руками по metadata_columns.txt)
    meta_path, meta = load_metadata(a.root, a.out)
    if F.group.isna().any():
        print("ВНИМАНИЕ: группа не определилась по пути у", int(F.group.isna().sum()), "файлов — заполните по таблице метаданных (см. metadata_columns.txt)")

    # отчёты врачей и разметка приступов рядом с сеансами
    def sdir(p):
        q = os.path.join(a.root, p)
        return os.path.dirname(os.path.dirname(q)) if C.parse_path(p)["montage"] else os.path.dirname(q)
    n_txt = Counter(); n_ann = Counter()
    for d in {sdir(p) for p in F.path}:
        n_txt[d] = len(glob.glob(os.path.join(d, "**", "*.txt"), recursive=True))
        n_ann[d] = len(glob.glob(os.path.join(d, "**", "*.csv_bi"), recursive=True)) + len(glob.glob(os.path.join(d, "**", "*.tse*"), recursive=True))
    F["n_txt_in_session"] = [n_txt[sdir(p)] for p in F.path]
    F["n_seizure_ann_in_session"] = [n_ann[sdir(p)] for p in F.path]
    F["skey"] = F.patient.astype(str) + "/" + F.session.astype(str)
    F.drop(columns=["skey"]).to_csv(os.path.join(a.out, "inventory_files.csv"), index=False)

    S = (F.groupby(["patient", "session"], dropna=False)
          .agg(group=("group", "first"), montage=("montage", lambda x: "|".join(sorted(set(map(str, x))))), n_files=("file", "size"),
               dur_min=("dur_s", lambda x: x.sum() / 60), sfreq=("sfreq", lambda x: "|".join(str(int(v)) for v in sorted(set(x.dropna())))),
               ref=("ref", lambda x: "|".join(sorted(set(x)))), all_have19=("has19", "all"), in_tusz=("in_tusz", "first"),
               n_txt=("n_txt_in_session", "max"), n_seizure_ann=("n_seizure_ann_in_session", "max"),
               year=("year", "first"), n_seiz=("n_seiz", "sum"), seiz_files=("n_seiz", lambda x: int((x > 0).sum()))).reset_index())
    S.to_csv(os.path.join(a.out, "inventory_sessions.csv"), index=False)

    P = F.groupby("patient").agg(group=("group", "first"), n_groups=("group", "nunique"))
    lines = [f"# Опись TUEP ({a.root})", "",
             f"- EDF-файлов: **{len(F)}**; сеансов: **{len(S)}**; пациентов: **{F.patient.nunique()}**; часов: **{F.dur_s.sum()/3600:.1f}**; объём EDF: {F.size_mb.sum()/1e3:.1f} ГБ",
             f"- Не разобрались имена по шаблону aaaaaaaa_sNNN_tNNN.edf: {int(F.patient.isna().sum())} файлов; не прочитался заголовок: {int((F.error != '').sum())}",
             f"- Файлов без полного набора 19 каналов 10–20: **{int((~F.has19).sum())}** ({F.loc[~F.has19, 'patient'].nunique()} пациентов)",
             f"- Пациенты, попавшие в обе группы (ошибка разбора пути): {int((P.n_groups > 1).sum())}", ""]
    for title, col in [("Группа", "group"), ("Монтаж", "montage"), ("Суффикс референта", "ref")]:
        t = F.groupby(col, dropna=False).agg(файлов=("file", "size"), сеансов=("skey", "nunique"), часов=("dur_s", lambda x: x.sum() / 3600),
                                             пациентов=("patient", "nunique"))
        lines += [f"## {title}", "", t.round(1).to_markdown(), ""]
    lines += ["## Частоты дискретизации, Гц (файлов)", "", pd.Series(F.sfreq.value_counts(dropna=False)).to_markdown(), "",
              "## Длительность файла, мин", "", (F.dur_s / 60).describe().round(1).to_markdown(), "",
              "## Наборы каналов (самые частые 5)", ""]
    for k, v in Counter(F.channels).most_common(5):
        lines.append(f"- {v} файлов: `{k[:300]}`")
    g = F.groupby("group", dropna=False)
    sess_txt = F.drop_duplicates("skey").assign(has_txt=lambda d: d.n_txt_in_session > 0).groupby("group", dropna=False).has_txt.sum()
    lines += ["", "## Пациенты и сеансы по группам", "",
              pd.DataFrame({"пациентов": g.patient.nunique(), "сеансов": g.skey.nunique(), "файлов": g.size(), "часов": g.dur_s.sum() / 3600,
                            "сеансов_с_отчётом_txt": sess_txt}).round(1).to_markdown(), ""]
    # протокол (PROTOCOL.md): приступы, сеансы на пациента, монтаж и год по группам — проверка, не совпадают ли они с классом
    Sx = S.assign(has_seiz=S.n_seiz > 0)
    sz = pd.DataFrame({"сеансов": Sx.groupby("group").size(), "сеансов_с_приступами": Sx.groupby("group").has_seiz.sum(),
                       "пациентов_с_приступами": Sx[Sx.has_seiz].groupby("group").patient.nunique()}).fillna(0)
    lines += ["## Приступы (разметка .csv_bi)", "", sz.to_markdown(), "",
              "## Сеансов на пациента", "", S.groupby(["group", "patient"]).size().groupby("group").describe().round(1).to_markdown(), "",
              "## Монтаж по группам (сеансы)", "", pd.crosstab(S.montage, S.group, margins=True).to_markdown(), "",
              "## Год записи по группам (сеансы)", "", S.groupby("group").year.describe().round(1).to_markdown(), ""]
    if tusz:
        s = S[S.in_tusz == True]
        lines += [f"## Пересечение с TUSZ (по списку `{[os.path.basename(l) for l in lists]}`)", "",
                  f"- сеансов в списке: {len(tusz)}; найдено среди сеансов описи: {len(s)}; пациентов: {s.patient.nunique()}; групп: {s.group.value_counts(dropna=False).to_dict()}", ""]
    else:
        lines += ["## Пересечение с TUSZ", "", "Список `*tusz*.list` не найден или формат строк не распознан — открыть файл и поправить `parse_tusz_keys`.", ""]
    lines += ["## Метаданные", "", f"Таблица: `{meta_path}`; листы и столбцы — в `metadata_columns.txt` (связать с описью по пациенту/сеансу руками после просмотра)." if meta_path else "Таблица `metadata*.xlsx` не найдена в --root."]
    open(os.path.join(a.out, "inventory_summary.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines[:12]))
    print("→", a.out)


if __name__ == "__main__":
    main()
