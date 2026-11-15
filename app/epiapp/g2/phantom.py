"""Синтетические «снимки» для демо-режима: схема головы и мозга, НЕ снимок человека. Создаются формулами (эллипсоиды + сглаженный шум),
поэтому вопросов лицензии и персональных данных нет. В просмотрщике подписаны «СХЕМА · ДЕМО»."""
import numpy as np
from scipy import ndimage

SHAPE, ZOOM = (144, 176, 144), 1.25                      # сетка RAS: i — вправо, j — вперёд, k — вверх; воксель 1,25 мм
# демо-очаги: положение (доля размера по i, j, k), радиус в вокселях, пик вероятности
LESIONS = [dict(pos=(0.30, 0.60, 0.66), r=7, p=0.96), dict(pos=(0.70, 0.35, 0.50), r=5, p=0.74), dict(pos=(0.56, 0.72, 0.38), r=4, p=0.41)]


def make_phantom(thick_flair=False, seed=1):
    """Возвращает dict(t1, flair, prob — (i, j, k) float32 0..1; affine; zooms_t1, zooms_flair)."""
    rng = np.random.default_rng(seed)
    nx, ny, nz = SHAPE
    I, J, K = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny), np.linspace(-1, 1, nz), indexing="ij")
    ell = lambda cx, cy, cz, rx, ry, rz: ((I - cx) / rx) ** 2 + ((J - cy) / ry) ** 2 + ((K - cz) / rz) ** 2
    head = ell(0, 0, -0.02, 0.80, 0.92, 0.93) < 1
    skull = ell(0, 0, 0.0, 0.72, 0.84, 0.84) < 1
    brain = (ell(0, 0, 0.06, 0.64, 0.77, 0.72) < 1) & (K > -0.62)
    cereb = (ell(0, -0.52, -0.56, 0.42, 0.26, 0.20) < 1)
    noise = ndimage.gaussian_filter(rng.normal(size=SHAPE), 3.2); noise /= noise.std()
    fine = ndimage.gaussian_filter(rng.normal(size=SHAPE), 1.6); fine /= fine.std()
    depth = ndimage.distance_transform_edt(brain) * ZOOM                         # мм от поверхности мозга
    fissure = (np.abs(I) < 0.018) & (K > -0.05) & brain                           # межполушарная щель
    sulci = brain & (depth < 9) & (noise + 0.5 * fine > 1.15)
    wm = brain & (depth > 5.5 + 2.6 * noise + 1.2 * fine) & ~fissure
    vent = ((ell(-0.13, -0.02, 0.10, 0.07, 0.36, 0.13) < 1) | (ell(0.13, -0.02, 0.10, 0.07, 0.36, 0.13) < 1)) & brain
    csf = (skull & ~brain & ~cereb) | fissure | sulci | vent
    gm = brain & ~wm & ~csf
    tissue = lambda scalp, bone, c, g, w: np.select([vent | csf, wm, gm, cereb, skull, head], [c, w, g, (g + w) / 2, bone, scalp], 0.0)
    t1 = tissue(0.62, 0.08, 0.10, 0.52, 0.86)
    fl = tissue(0.50, 0.06, 0.05, 0.74, 0.56)
    prob = np.zeros(SHAPE, np.float32)
    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    for L in LESIONS:
        c = [int(L["pos"][a] * SHAPE[a]) for a in range(3)]
        d2 = (ii - c[0]) ** 2 + (jj - c[1]) ** 2 + (kk - c[2]) ** 2
        g = np.exp(-d2 / (2 * (L["r"] * 0.75) ** 2)) * brain
        blur = ndimage.gaussian_filter(t1, 2.5)
        t1 = t1 * (1 - 0.8 * g) + blur * 0.8 * g                                 # размытая граница серого и белого вещества
        fl = fl + 0.22 * g * (gm | wm)                                           # повышенный сигнал на FLAIR
        prob = np.maximum(prob, (L["p"] * np.clip(2.2 * np.exp(-d2 / (2 * (L["r"] * 0.62) ** 2)), 0, 1)).astype(np.float32))   # плато, как у настоящих карт
    prob *= brain
    prob[prob < 0.02] = 0
    bias = 1 + 0.12 * (0.6 * J - 0.4 * K)
    fin = lambda v: np.clip(ndimage.gaussian_filter(v * bias, 0.7) + rng.normal(0, 0.012, SHAPE), 0, 1).astype(np.float32)
    t1, fl = fin(t1), fin(fl)
    zf = (ZOOM, ZOOM, ZOOM)
    if thick_flair:                                                              # срезы 5 мм: усреднение по 4 слоя
        k4 = (nz // 4) * 4
        blk = fl[:, :, :k4].reshape(nx, ny, k4 // 4, 4).mean(-1)
        fl = np.concatenate([np.repeat(blk, 4, axis=2), fl[:, :, k4:]], axis=2).astype(np.float32)
        zf = (0.9, 0.9, 5.0)
    aff = np.diag([ZOOM, ZOOM, ZOOM, 1.0]); aff[:3, 3] = [-nx * ZOOM / 2, -ny * ZOOM / 2 - 12, -nz * ZOOM / 2 + 8]
    return dict(t1=t1, flair=fl, prob=prob, brain=brain, affine=aff, zooms_t1=(ZOOM, ZOOM, ZOOM), zooms_flair=zf)
