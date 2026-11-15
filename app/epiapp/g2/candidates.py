"""Кандидаты в очаги по карте вероятности: связные области выше порога (рабочей точки)."""
import numpy as np
from scipy import ndimage

MIN_ML = 0.1          # области меньше 0,1 см³ отбрасываются (у MS-DSA-NET — не меньше 100 вокселей)


def cluster_table(prob, thr, affine, brain=None, min_ml=MIN_ML, top=None, midline_x=None):
    """prob (i, j, k) 0..1 в сетке RAS; affine — воксель -> мм. Возвращает список кандидатов по убыванию пиковой вероятности:
    {"n", "peak_prob", "volume_ml", "xyz": [x, y, z] мм (пик), "ijk": [i, j, k] (пик), "hemi": "левое" | "правое"}."""
    m = prob >= thr
    if brain is not None:
        m &= brain
    lab, n = ndimage.label(m, structure=np.ones((3, 3, 3)))
    if not n:
        return []
    vox_ml = abs(float(np.linalg.det(affine[:3, :3]))) / 1000.0
    idx = np.arange(1, n + 1)
    size = ndimage.sum(m, lab, idx) * vox_ml
    peaks = ndimage.maximum_position(prob, lab, idx)
    if midline_x is None:
        midline_x = float((affine @ np.array([prob.shape[0] / 2, prob.shape[1] / 2, prob.shape[2] / 2, 1.0]))[0])
    out = []
    for k in range(n):
        if size[k] < min_ml:
            continue
        ijk = [int(v) for v in peaks[k]]
        xyz = (affine @ np.array([*ijk, 1.0]))[:3]
        out.append(dict(peak_prob=float(prob[tuple(ijk)]), volume_ml=float(size[k]), xyz=[float(v) for v in xyz], ijk=ijk,
                        hemi="левое" if xyz[0] < midline_x else "правое"))
    out.sort(key=lambda c: -c["peak_prob"])
    for i, c in enumerate(out):
        c["n"] = i + 1
    return out[:top] if top else out
