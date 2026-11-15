"""Объёмы МРТ для просмотра. Всё приводится к одной сетке (T1 в ориентации RAS): так три плоскости, карта вероятности и
координаты кандидатов согласованы при любом адаптере модели. Чтение NIfTI — nibabel (нужен только для настоящих файлов)."""
import json
import numpy as np
from scipy import ndimage

MAX_DIM = 256                 # больше — прореживаем, чтобы объём быстро передавался в браузер
T1_WARN_MM, FLAIR_WARN_MM, FLAIR_BAD_MM = 1.5, 3.5, 5.0


def _nib():
    try:
        import nibabel
        return nibabel
    except ImportError as e:
        raise RuntimeError("для чтения NIfTI нужен пакет nibabel (pip install nibabel)") from e


def header_info(path):
    """Размер вокселя и сетки по заголовку (без чтения данных)."""
    img = _nib().load(str(path))
    z = [float(v) for v in img.header.get_zooms()[:3]]
    return dict(zooms=z, shape=[int(v) for v in img.shape[:3]], ndim=len(img.shape))


def fmt_mm(z):
    return " × ".join(f"{v:.1f}".replace(".", ",") for v in z) + " мм"


def check_input(t1_info, flair_info):
    """Проверка снимков до запуска — по заголовкам. Формат как у проверки ЭЭГ: карточки со статусом ok / warn / bad."""
    items, notes, flair5 = [], [], False
    zt, zf = t1_info["zooms"], flair_info["zooms"]
    st = "ok" if max(zt) <= T1_WARN_MM else "warn"
    items.append(dict(key="t1", title="T1", value=fmt_mm(zt), status=st,
                      note="3D, тонкие срезы" if st == "ok" else "Срезы толще 1,5 мм — качество поиска ниже"))
    if max(zf) >= FLAIR_BAD_MM:
        st, note, flair5 = "bad", "Толстые срезы", True
    elif max(zf) > FLAIR_WARN_MM:
        st, note = "warn", "Срезы толще 3,5 мм"
    else:
        st, note = "ok", "Тонкие срезы"
    items.append(dict(key="flair", title="FLAIR", value=fmt_mm(zf), status=st, note=note))
    items.append(dict(key="reg", title="Совмещение T1 и FLAIR", value="выполнится автоматически", status="ok", note=""))
    for name, info in (("T1", t1_info), ("FLAIR", flair_info)):
        if info.get("ndim", 3) != 3:
            notes.append(f"{name}: в файле {info['ndim']} измерения — нужен один трёхмерный объём.")
    return dict(items=items, can_run=not notes, low_quality=flair5 or any(i["status"] != "ok" for i in items), flair5=flair5, notes=notes,
                summary=dict(t1=fmt_mm(zt), flair=fmt_mm(zf), t1_mm=max(zt), flair_mm=max(zf)))


def _to_u8(v, mask=None, lo_p=0.5, hi_p=99.7):
    """Окно/уровень «авто»: по процентилям ненулевых значений."""
    v = np.nan_to_num(np.asarray(v, np.float32))
    ref = v[v > 0] if (v > 0).any() else v.ravel()
    lo, hi = np.percentile(ref, [lo_p, hi_p])
    return np.clip((v - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)


def _canonical(path):
    nib = _nib()
    img = nib.as_closest_canonical(nib.load(str(path)))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim > 3:
        data = data[..., 0]
    return data, np.array(img.affine, float)


def _resample(mov, aff_mov, shape, aff_fix, order=1):
    """Объём mov в сетку (shape, aff_fix) по мировым координатам."""
    M = np.linalg.inv(aff_mov) @ aff_fix
    return ndimage.affine_transform(mov, M[:3, :3], offset=M[:3, 3], output_shape=shape, order=order, mode="constant", cval=0.0)


def build_pack(t1_path, flair_path, prob_path, brain_path=None):
    """Набор для просмотра из файлов NIfTI: T1 — опорная сетка (RAS), остальное пересчитывается в неё.
    Возвращает dict(t1, flair, prob uint8; brain bool | None; affine; zooms)."""
    t1, aff = _canonical(t1_path)
    step = int(np.ceil(max(t1.shape) / MAX_DIM))
    if step > 1:
        t1 = t1[::step, ::step, ::step]
        aff = aff @ np.diag([step, step, step, 1.0])
    fl, aff_f = _canonical(flair_path)
    pr, aff_p = _canonical(prob_path)
    same = lambda a, s: a.shape == t1.shape and np.allclose(s, aff, atol=1e-3)
    fl = fl if same(fl, aff_f) else _resample(fl, aff_f, t1.shape, aff)
    pr = pr if same(pr, aff_p) else _resample(pr, aff_p, t1.shape, aff)
    brain = None
    if brain_path:
        b, aff_b = _canonical(brain_path)
        brain = (b if same(b, aff_b) else _resample(b, aff_b, t1.shape, aff, order=0)) > 0.5
    return dict(t1=_to_u8(t1), flair=_to_u8(fl), prob=np.clip(np.nan_to_num(pr) * 255, 0, 255).astype(np.uint8), brain=brain, affine=aff,
                zooms=[float(v) for v in np.sqrt((aff[:3, :3] ** 2).sum(0))])


def pack_from_arrays(t1, flair, prob, affine, brain=None):
    """То же из массивов (демо-режим)."""
    return dict(t1=_to_u8(t1), flair=_to_u8(flair), prob=np.clip(prob * 255, 0, 255).astype(np.uint8), brain=brain, affine=np.array(affine, float),
                zooms=[float(v) for v in np.sqrt((np.array(affine)[:3, :3] ** 2).sum(0))])


def save_pack(pack, folder):
    """На диск: три объёма uint8 в порядке «i быстрее всех» (срез k — подряд идущий кусок) + описание."""
    folder.mkdir(parents=True, exist_ok=True)
    for k in ("t1", "flair", "prob"):
        (folder / f"{k}.u8").write_bytes(pack[k].tobytes(order="F"))
    if pack.get("brain") is not None:
        (folder / "brain.u8").write_bytes(pack["brain"].astype(np.uint8).tobytes(order="F"))
    info = dict(shape=[int(v) for v in pack["t1"].shape], zooms=pack["zooms"], affine=np.array(pack["affine"]).tolist(), has_brain=pack.get("brain") is not None)
    (folder / "info.json").write_text(json.dumps(info), encoding="utf-8")
    return info


def load_pack(folder):
    info = json.loads((folder / "info.json").read_text(encoding="utf-8"))
    rd = lambda n: np.frombuffer((folder / f"{n}.u8").read_bytes(), np.uint8).reshape(info["shape"], order="F")
    return dict(t1=rd("t1"), flair=rd("flair"), prob=rd("prob"), brain=rd("brain").astype(bool) if info.get("has_brain") else None,
                affine=np.array(info["affine"]), zooms=info["zooms"], info=info)


def thumb_from_volume(v):
    """Аксиальный срез (на 0,6 высоты) в радиологической ориентации: правое полушарие слева, перед сверху. v — (i, j, k) в RAS."""
    import base64
    a = np.asarray(v[:, :, int(v.shape[2] * 0.6)])
    step = int(np.ceil(max(a.shape) / 192))
    img = np.ascontiguousarray(a[::step, ::step][::-1, ::-1].T)
    if img.dtype != np.uint8:
        img = _to_u8(img)
    return dict(w=int(img.shape[1]), h=int(img.shape[0]), data=base64.b64encode(img.tobytes()).decode("ascii"))


def thumb_from_file(path):
    data, aff = _canonical(path)
    t = thumb_from_volume(data)
    z = np.sqrt((aff[:3, :3] ** 2).sum(0))
    t["aspect"] = float((data.shape[0] * z[0]) / (data.shape[1] * z[1]))
    return t
