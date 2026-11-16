// Просмотрщик срезов МРТ на canvas: три плоскости, наложение карты вероятности, контуры кандидатов, перекрестие.
// Объёмы приходят с сервера в одной сетке (RAS: i — вправо, j — вперёд, k — вверх), по байту на воксель.
// Показ — в радиологической ориентации: правое полушарие пациента слева на экране.
import { api } from './api.js';

export const PLANES = { ax: 'Аксиальный', cor: 'Корональный', sag: 'Сагиттальный' };
const LAB = { ax: ['П', 'Л'], cor: ['П', 'Л'], sag: ['Перед', 'Зад'] };

export class Mri {
  constructor(info, vols) {
    this.info = info; this.v = vols;
    [this.nx, this.ny, this.nz] = info.shape; this.z = info.zooms;
    this._buf = {}; this._profile = null;
  }
  static async load(studyId, info) {
    const [t1, flair, prob] = await Promise.all(['t1', 'flair', 'prob'].map(n => api.bytes(`/api/g2/studies/${studyId}/volume/${n}`)));
    return new Mri(info, { T1: t1, FLAIR: flair, prob });
  }
  // размеры плоскости в вокселях и в мм; оси экрана -> оси объёма
  geom(plane) {
    const { nx, ny, nz, z } = this;
    if (plane === 'ax') return { w: nx, h: ny, pw: nx * z[0], ph: ny * z[1], n: nz };
    if (plane === 'cor') return { w: nx, h: nz, pw: nx * z[0], ph: nz * z[2], n: ny };
    return { w: ny, h: nz, pw: ny * z[1], ph: nz * z[2], n: nx };
  }
  slice(plane, pos) { return plane === 'ax' ? pos.k : plane === 'cor' ? pos.j : pos.i; }
  // положение вокселя на экране плоскости (в вокселях плоскости)
  toScreen(plane, p) {
    const { nx, ny, nz } = this;
    if (plane === 'ax') return [nx - 1 - p.i, ny - 1 - p.j];
    if (plane === 'cor') return [nx - 1 - p.i, nz - 1 - p.k];
    return [ny - 1 - p.j, nz - 1 - p.k];
  }
  fromScreen(plane, x, y, pos) {
    const { nx, ny, nz } = this, c = (v, n) => Math.max(0, Math.min(n - 1, Math.round(v)));
    if (plane === 'ax') return { i: c(nx - 1 - x, nx), j: c(ny - 1 - y, ny), k: pos.k };
    if (plane === 'cor') return { i: c(nx - 1 - x, nx), j: pos.j, k: c(nz - 1 - y, nz) };
    return { i: pos.i, j: c(ny - 1 - x, ny), k: c(nz - 1 - y, nz) };
  }
  // картинка среза (серый снимок + карта) в отдельном canvas размером w × h вокселей
  render(plane, pos, o) {
    const { nx, ny, nz } = this, g = this.geom(plane), base = this.v[o.seq], prob = this.v.prob;
    const key = plane + (o.zoom ? 'z' : ''); let off = this._buf[key];
    if (!off) { off = this._buf[key] = document.createElement('canvas'); off.width = g.w; off.height = g.h; }
    const ctx = off.getContext('2d'), img = ctx.createImageData(g.w, g.h), d = img.data, op = o.overlay ? o.op : 0;
    for (let y = 0; y < g.h; y++) for (let x = 0; x < g.w; x++) {
      let i, j, k;
      if (plane === 'ax') { i = nx - 1 - x; j = ny - 1 - y; k = pos.k; } else if (plane === 'cor') { i = nx - 1 - x; j = pos.j; k = nz - 1 - y; } else { i = pos.i; j = ny - 1 - x; k = nz - 1 - y; }
      const q = i + nx * (j + ny * k), b = base[q], p = prob[q] / 255, t = (y * g.w + x) * 4;
      let r = b, gr = b, bl = b;
      if (op > 0 && p > 0.08) {                              // цвет карты — янтарный «цвет программы»: чем выше вероятность, тем светлее
        const a = op * Math.min(1, (p - 0.05) / 0.45), u = Math.min(1, p / 0.9);
        const cr = 255, cg = 150 + 90 * u, cb = 20 + 60 * u * u;
        r = r * (1 - a) + cr * a; gr = gr * (1 - a) + cg * a; bl = bl * (1 - a) + cb * a;
      }
      d[t] = r; d[t + 1] = gr; d[t + 2] = bl; d[t + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    return off;
  }
  // рисование плоскости на видимом canvas. o: {seq, op, overlay, cands, sel, cross, zoom}
  // zoom > 1 — увеличение с центром в перекрестии (окно «под перекрестием»). Подписи плоскости и среза — в разметке, не на canvas.
  draw(cv, plane, pos, o) {
    const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight; if (!W || !H) return;
    cv.width = W * dpr; cv.height = H * dpr;
    const c = cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.fillStyle = '#000'; c.fillRect(0, 0, W, H);
    const g = this.geom(plane), pad = o.zoom ? 0 : 14, s = Math.min((W - 2 * pad) / g.pw, (H - 2 * pad) / g.ph) * (o.zoom || 1), dw = g.pw * s, dh = g.ph * s;
    const [px, py] = this.toScreen(plane, pos);
    let ox = (W - dw) / 2, oy = (H - dh) / 2;
    if (o.zoom) { ox = W / 2 - (px + .5) * dw / g.w; oy = H / 2 - (py + .5) * dh / g.h; }
    cv._t = { ox, oy, sx: dw / g.w, sy: dh / g.h, plane };
    c.imageSmoothingEnabled = true; c.imageSmoothingQuality = 'high';
    c.drawImage(this.render(plane, pos, o), ox, oy, dw, dh);
    const X = x => ox + (x + .5) * dw / g.w, Y = y => oy + (y + .5) * dh / g.h;
    const zs = plane === 'ax' ? this.z[2] : plane === 'cor' ? this.z[1] : this.z[0];
    let gap = 8;
    (o.cands || []).forEach(L => {                            // контур кандидата: виден на срезах, проходящих через него
      const p = { i: L.ijk[0], j: L.ijk[1], k: L.ijk[2] }, rmm = Math.cbrt(3 * L.volume_ml * 1000 / (4 * Math.PI)) + 2;
      const dmm = Math.abs(this.slice(plane, pos) - this.slice(plane, p)) * zs; if (dmm > rmm) return;
      const [x, y] = this.toScreen(plane, p), rr = Math.max(6, Math.sqrt(rmm * rmm - dmm * dmm) * s * 1.2), on = L.n === o.sel;
      const col = L.below ? '#95a1ac' : '#f3bd4b';
      c.strokeStyle = col; c.lineWidth = on ? 1.8 : 1.1; c.setLineDash(on ? [] : [4, 3]); c.beginPath(); c.arc(X(x), Y(y), rr, 0, 6.2832); c.stroke(); c.setLineDash([]);
      if (!o.zoom) { c.fillStyle = col; c.font = '600 12px Consolas,monospace'; c.fillText(String(L.n), X(x) + rr * .72 + 4, Y(y) - rr * .72 - 2); }
      if (on && Math.hypot(x - px, y - py) < 2) gap = rr + 5;
    });
    if (o.cross) {                                            // перекрестие с разрывом в центре: не закрывает то, на что указывает
      const x = X(px) + .5, y = Y(py) + .5;
      c.strokeStyle = 'rgba(255,255,255,.45)'; c.lineWidth = 1; c.beginPath();
      c.moveTo(x, Math.max(0, oy)); c.lineTo(x, y - gap); c.moveTo(x, y + gap); c.lineTo(x, Math.min(H, oy + dh));
      c.moveTo(Math.max(0, ox), y); c.lineTo(x - gap, y); c.moveTo(x + gap, y); c.lineTo(Math.min(W, ox + dw), y); c.stroke();
    }
    if (!o.zoom) {                                            // стороны: в радиологической ориентации правое полушарие — слева
      c.fillStyle = '#95a1ac'; c.font = '11px "Segoe UI",system-ui,sans-serif'; c.textBaseline = 'middle';
      c.textAlign = 'left'; c.fillText(LAB[plane][0], 8, H / 2); c.textAlign = 'right'; c.fillText(LAB[plane][1], W - 8, H / 2);
      c.textAlign = 'left'; c.textBaseline = 'alphabetic';
    }
  }
  // щелчок по плоскости -> новое положение перекрестия
  pick(cv, ev, pos) {
    const t = cv._t; if (!t) return pos;
    const r = cv.getBoundingClientRect();
    return this.fromScreen(t.plane, (ev.clientX - r.left - t.ox) / t.sx - .5, (ev.clientY - r.top - t.oy) / t.sy - .5, pos);
  }
  // наибольшая вероятность на каждом аксиальном срезе (считается один раз)
  profile() {
    if (this._profile) return this._profile;
    const { nx, ny, nz } = this, p = new Float32Array(nz), n = nx * ny, prob = this.v.prob;
    for (let k = 0; k < nz; k++) { let m = 0; const a = k * n; for (let q = a; q < a + n; q++) if (prob[q] > m) m = prob[q]; p[k] = m / 255; }
    return (this._profile = p);
  }
  center() { return { i: this.nx >> 1, j: this.ny >> 1, k: Math.round(this.nz * 0.55) }; }
  clampPos(p) { const c = (v, n) => Math.max(0, Math.min(n - 1, v)); return { i: c(p.i, this.nx), j: c(p.j, this.ny), k: c(p.k, this.nz) }; }
}

// Кандидаты при рабочей точке wp + (по желанию) более слабые: «ниже рабочей точки», серым.
export async function loadCands(studyId, wp, lo) {
  const above = (await api.get(`/api/g2/studies/${studyId}/candidates?thr=${wp}`)).candidates.map(c => ({ ...c, below: false }));
  let below = [];
  if (lo != null && lo < wp) {
    const all = (await api.get(`/api/g2/studies/${studyId}/candidates?thr=${lo}`)).candidates;
    below = all.filter(c => c.peak_prob < wp).map((c, i) => ({ ...c, n: above.length + i + 1, below: true }));
  }
  return { above, below };
}
// ключ кандидата для отметок врача: координаты, округлённые до миллиметра (как на сервере, report.cand_key)
export const candKey = (c) => c.xyz.map(v => { const r = Math.round(v); return String(r === 0 ? 0 : r); }).join(';');
export const xyzText = (c) => c.xyz.map(v => (v >= 0 ? '+' : '−') + Math.abs(Math.round(v))).join('; ');
export function opText(card, thr, fmt) {
  const row = (card.table || []).find(x => Math.abs(x.thr - thr) < 1e-6);
  if (!row) return '';
  return `на открытых данных: очаг найден у ${Math.round(row.sens * 100)} % пациентов с ФКД, в среднем ${fmt(row.fp)} ложных находок на человека без ФКД`;
}
