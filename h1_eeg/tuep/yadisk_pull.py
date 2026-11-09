"""Скачать папку с публичного Яндекс Диска по файлам, с докачкой и проверкой md5.

Только стандартная библиотека Python, ничего ставить не нужно.

Обычный режим (без входа в Яндекс):
    py yadisk_pull.py "https://disk.yandex.ru/d/XXXX" "C:/.../TUEP" --list      # только список и объём
    py yadisk_pull.py "https://disk.yandex.ru/d/XXXX" "C:/.../TUEP"             # скачать всё, чего ещё нет
    py yadisk_pull.py URL DEST --max-gb 5                                        # не больше 5 ГБ за запуск
    py yadisk_pull.py URL DEST --only "*.edf" --only "*.csv"                     # фильтр по имени/пути

Запускать повторно можно сколько угодно: готовые файлы (размер и md5 совпали) пропускаются,
недокачанные (*.part) докачиваются с места обрыва, новые файлы на диске подхватываются.

Если Яндекс отвечает «превышен лимит скачивания» для публичной ссылки (бывает у популярных
публичных файлов), есть обходной режим --via-own-disk: файл копируется на ваш Яндекс Диск,
скачивается оттуда и затем удаляется с вашего диска (без корзины). Нужен OAuth-токен
Яндекс Диска в переменной окружения YADISK_TOKEN (получаете сами; в код и в чаты не вставлять).
На вашем диске должно быть свободно больше, чем самый крупный файл.
"""
import argparse, fnmatch, hashlib, json, os, sys, time, urllib.error, urllib.parse, urllib.request

API = "https://cloud-api.yandex.net/v1/disk"
UA = {"User-Agent": "yadisk-pull/1.0"}


def call(url, method="GET", token=None, tries=6):
    h = dict(UA)
    if token:
        h["Authorization"] = f"OAuth {token}"
    for k in range(tries):
        try:
            req = urllib.request.Request(url, method=method, headers=h)
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
                return r.status, (json.loads(body) if body else {})
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            if e.code == 429 and "/download" in url:       # ссылку на скачивание ждёт основной цикл
                pass
            elif e.code in (429, 500, 502, 503, 504) and k < tries - 1:
                time.sleep(min(300, 10 * 2 ** k)); continue
            return e.code, (json.loads(body) if body.startswith("{") else {"message": body})
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if k < tries - 1:
                time.sleep(min(300, 10 * 2 ** k)); continue
            raise
    raise RuntimeError("не удалось обратиться к API")


def q(**kw):
    return urllib.parse.urlencode({k: v for k, v in kw.items() if v is not None})


def walk(public_key, path="/"):
    """Все файлы публичной папки: (путь внутри папки, размер, md5)."""
    off = 0
    while True:
        st, d = call(f"{API}/public/resources?" + q(public_key=public_key, path=path, limit=1000, offset=off))
        if st != 200:
            sys.exit(f"ошибка списка {path}: {st} {d.get('message') or d}")
        if d.get("type") == "file":                       # ссылка ведёт на один файл
            yield d.get("path") or "/" + d["name"], d.get("size", 0), d.get("md5"); return
        items = d["_embedded"]["items"]
        for it in items:
            if it["type"] == "dir":
                yield from walk(public_key, it["path"])
            else:
                yield it["path"], it.get("size", 0), it.get("md5")
        off += len(items)
        if off >= d["_embedded"].get("total", 0) or not items:
            return


def md5_of(fn):
    h = hashlib.md5()
    with open(fn, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(href, part, size, limit_mbps=None):
    """Скачать по прямой ссылке в .part с докачкой (Range)."""
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if have > size:
        os.remove(part); have = 0
    if have == size:
        return
    h = dict(UA); h["Range"] = f"bytes={have}-"
    req = urllib.request.Request(href, headers=h)
    with urllib.request.urlopen(req, timeout=120) as r:
        if have and r.status != 206:                       # сервер не дал докачку — начинаем заново
            have = 0; mode = "wb"
        else:
            mode = "ab"
        t0, got = time.time(), 0
        with open(part, mode) as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b); got += len(b)
                if limit_mbps:                              # ограничение скорости, Мбит/с
                    need = got * 8 / (limit_mbps * 1e6) - (time.time() - t0)
                    if need > 0:
                        time.sleep(need)


def direct_href(public_key, path, token, own_tmp):
    """Прямая ссылка: из публичной папки, либо через копию на своём диске."""
    if not own_tmp:
        st, d = call(f"{API}/public/resources/download?" + q(public_key=public_key, path=path))
        if st == 200:
            return d["href"], None
        return None, f"{st} {d.get('error', '')} {d.get('message', '')}"
    name = path.strip("/").replace("/", "__")
    dst = f"{own_tmp}/{name}"
    call(f"{API}/resources?" + q(path=dst, permanently="true"), method="DELETE", token=token)   # остаток прошлого запуска
    st, d = call(f"{API}/public/resources/save-to-disk?" + q(public_key=public_key, path=path, save_path=own_tmp, name=name),
                 method="POST", token=token)
    if st not in (200, 201, 202):
        return None, f"копирование на свой диск: {st} {d.get('message', '')}"
    if st == 202 and "href" in d:                          # асинхронная операция — ждём
        for _ in range(720):
            s2, op = call(d["href"], token=token)
            if op.get("status") == "success":
                break
            if op.get("status") == "failed":
                return None, "копирование на свой диск не удалось"
            time.sleep(5)
    st, d = call(f"{API}/resources/download?" + q(path=dst), token=token)
    if st != 200:
        return None, f"ссылка со своего диска: {st} {d.get('message', '')}"
    return d["href"], dst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url", help="публичная ссылка на папку Яндекс Диска")
    ap.add_argument("dest", help="куда класть (папка TUEP)")
    ap.add_argument("--list", action="store_true", help="только показать список и объём")
    ap.add_argument("--only", action="append", help="шаблон имени/пути (можно несколько), напр. *.edf")
    ap.add_argument("--max-gb", type=float, help="не скачивать больше стольких ГБ за запуск")
    ap.add_argument("--limit-mbps", type=float, help="ограничить скорость, Мбит/с")
    ap.add_argument("--pause", type=float, default=1.0, help="пауза между файлами, с")
    ap.add_argument("--no-md5", action="store_true", help="не сверять md5 (быстрее)")
    ap.add_argument("--via-own-disk", action="store_true", help="через свой Яндекс Диск (нужен YADISK_TOKEN)")
    a = ap.parse_args()
    a.url = a.url.strip().replace("disk.360.yandex.ru", "disk.yandex.ru")   # ссылки Яндекс 360 → обычные

    token = os.environ.get("YADISK_TOKEN") if a.via_own_disk else None
    if a.via_own_disk and not token:
        sys.exit("для --via-own-disk задайте переменную окружения YADISK_TOKEN")
    own_tmp = "disk:/_yadisk_pull_tmp" if a.via_own_disk else None
    if own_tmp:
        call(f"{API}/resources?" + q(path=own_tmp), method="PUT", token=token)   # папка; если есть — ок

    os.makedirs(a.dest, exist_ok=True)
    files = [f for f in walk(a.url)
             if not a.only or any(fnmatch.fnmatch(f[0], p) or fnmatch.fnmatch(os.path.basename(f[0]), p) for p in a.only)]
    total = sum(s for _, s, _ in files)
    done = [f for f in files if os.path.exists(os.path.join(a.dest, f[0].lstrip("/"))) and
            os.path.getsize(os.path.join(a.dest, f[0].lstrip("/"))) == f[1]]
    print(f"на диске: {len(files)} файлов, {total / 1e9:.2f} ГБ; уже скачано {len(done)} файлов, "
          f"{sum(s for _, s, _ in done) / 1e9:.2f} ГБ")
    with open(os.path.join(a.dest, "_yadisk_manifest.json"), "w", encoding="utf-8") as f:
        json.dump([{"path": p, "size": s, "md5": m} for p, s, m in files], f, ensure_ascii=False, indent=0)
    if a.list:
        for p, s, _ in files[:50]:
            print(f"{s / 1e6:10.1f} МБ  {p}")
        if len(files) > 50:
            print(f"... и ещё {len(files) - 50}")
        return

    budget = a.max_gb * 1e9 if a.max_gb else float("inf")
    got_bytes, errors, t0 = 0, [], time.time()
    log = open(os.path.join(a.dest, "_yadisk_pull.log"), "a", encoding="utf-8")
    for i, (path, size, md5) in enumerate(files, 1):
        out = os.path.join(a.dest, path.lstrip("/"))
        if os.path.exists(out) and os.path.getsize(out) == size:
            continue
        if got_bytes + size > budget:
            print(f"достигнут лимит {a.max_gb} ГБ за запуск — остановка"); break
        os.makedirs(os.path.dirname(out), exist_ok=True)
        part = out + ".part"
        ok = False
        attempt, wait429 = 0, 60
        while attempt < 4:
            attempt += 1
            href, info = direct_href(a.url, path, token, own_tmp)
            if not href and info.startswith("429"):        # слишком много запросов: ждём дольше, попытку не считаем
                print(f"[{i}/{len(files)}] Яндекс просит притормозить (429), жду {wait429} с")
                time.sleep(wait429); wait429 = min(wait429 * 2, 900); attempt -= 1
                a.pause = min(max(a.pause, 1.0) * 1.5, 10)  # и дальше идём медленнее
                continue
            if not href:
                msg = f"нет ссылки: {info}"
                if "Limit" in info or "limit" in info or "лимит" in info:
                    msg += " (лимит Яндекса: ждём 10 мин; если повторяется — запустите с --via-own-disk)"
                    print(msg); time.sleep(600); continue
                print(f"[{i}/{len(files)}] {path}: {msg}"); time.sleep(30); continue
            try:
                fetch(href, part, size, a.limit_mbps)
            except Exception as e:
                print(f"[{i}/{len(files)}] {path}: обрыв ({e}), повтор"); time.sleep(30); continue
            finally:
                if info and own_tmp:                       # info = путь копии на своём диске
                    call(f"{API}/resources?" + q(path=info, permanently="true"), method="DELETE", token=token)
            if os.path.getsize(part) != size:
                print(f"[{i}/{len(files)}] {path}: размер не совпал, докачка"); continue
            if md5 and not a.no_md5 and md5_of(part) != md5:
                print(f"[{i}/{len(files)}] {path}: md5 не совпал, качаю заново"); os.remove(part); continue
            os.replace(part, out); ok = True; break
        if ok:
            got_bytes += size
            sp = got_bytes / max(time.time() - t0, 1) / 1e6
            print(f"[{i}/{len(files)}] ok {size / 1e6:.1f} МБ  {path}   (за запуск {got_bytes / 1e9:.2f} ГБ, {sp:.1f} МБ/с)")
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\tok\t{size}\t{path}\n")
        else:
            errors.append(path); log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\tfail\t{size}\t{path}\n")
        log.flush(); time.sleep(a.pause)
    print(f"готово: скачано {got_bytes / 1e9:.2f} ГБ, ошибок {len(errors)}"
          + (" — запустите ещё раз, недокачанные продолжатся" if errors else ""))


if __name__ == "__main__":
    main()
