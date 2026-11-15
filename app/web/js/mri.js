// Просмотрщик срезов МРТ на canvas: три плоскости, наложение карты вероятности, контуры кандидатов, перекрестие.
// Объёмы приходят с сервера в одной сетке (RAS: i — вправо, j — вперёд, k — вверх), по байту на воксель.
// Показ — в радиологической ориентации, как в макете: правое полушарие пациента слева на экране.
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
    const key = plane; let off = this._buf[key];
    if (!off) { off = this._buf[key] = document.createElement('canvas'); off.width = g.w; off.height = g.h; }
    const ctx = off.getContext('2d'), img = ctx.createImageData(g.w, g.h), d = img.data, op = o.overlay ? o.op : 0;
    for (let y = 0; y < g.h; y++) for (let x = 0; x < g.w; x++) {
      let i, j, k;
      if (plane === 'ax') { i = nx - 1 - x; j = ny - 1 - y; k = pos.k; } else if (plane === 'cor') { i = nx - 1 - x; j = pos.j; k = nz - 1 - y; } else { i = pos.i; j = ny - 1 - x; k = nz - 1 - y; }
      const q = i + nx * (j + ny * k), b = base[q], p = prob[q] / 255, t = (y * g.w + x) * 4;
      let r = b, gr = b, bl = b;
      if (op > 0 && p > 0.08) {                              // цвет карты: красный -> оранжевый -> жёлтый, как в макете
        const a = op * Math.min(1, (p - 0.05) / 0.45), u = Math.min(1, p / 0.9);
        const cr = 255, cg = 32 + (243 - 32) * u * u, cb = 32 + (90 - 32) * u;
        r = r * (1 - a) + cr * a; gr = gr * (1 - a) + cg * a; bl = bl * (1 - a) + cb * a;
      }
      d[t] = r; d[t + 1] = gr; d[t + 2] = bl; d[t + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    return off;
  }
  // рисование плоскости на видимом canvas. o: {seq, op, overlay, cands, wp, cross, demo, caption, sel}
  draw(cv, plane, pos, o) {
    const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight; if (!W || !H) return;
    cv.width = W * dpr; cv.height = H * dpr;
    const c = cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.fillStyle = '#000'; c.fillRect(0, 0, W, H);
    const g = this.geom(plane), s = Math.min(W / g.pw, H / g.ph), dw = g.pw * s, dh = g.ph * s, ox = (W - dw) / 2, oy = (H - dh) / 2;
    cv._t = { ox, oy, sx: dw / g.w, sy: dh / g.h, plane };
    c.imageSmoothingEnabled = true; c.imageSmoothingQuality = 'high';
    c.drawImage(this.render(plane, pos, o), ox, oy, dw, dh);
    const X = x => ox + (x + .5) * dw / g.w, Y = y => oy + (y + .5) * dh / g.h;
    const zs = plane === 'ax' ? this.z[2] : plane === 'cor' ? this.z[1] : this.z[0];
    (o.cands || []).forEach(L => {                            // контур кандидата: виден на срезах, проходящих через него
      const p = { i: L.ijk[0], j: L.ijk[1], k: L.ijk[2] }, rmm = Math.cbrt(3 * L.volume_ml * 1000 / (4 * Math.PI)) + 2;
      const dmm = Math.abs(this.slice(plane, pos) - this.slice(plane, p)) * zs; if (dmm > rmm) return;
      const [x, y] = this.toScreen(plane, p), rr = Math.max(5, Math.sqrt(rmm * rmm - dmm * dmm) * s * 1.2);
      const col = L.below ? '#8899aa' : '#4fd1ff';
      c.strokeStyle = col; c.lineWidth = L.n === o.sel && !L.below ? 2 : 1.2; c.setLineDash([4, 3]); c.beginPath(); c.arc(X(x), Y(y), rr, 0, 6.2832); c.stroke(); c.setLineDash([]);
      c.fillStyle = col; c.font = '700 12px system-ui,sans-serif'; c.fillText(String(L.n), X(x) + rr + 3, Y(y) - rr + 4);
    });
    if (o.cross) {
      const [x, y] = this.toScreen(plane, pos);
      c.strokeStyle = 'rgba(63,169,245,.75)'; c.lineWidth = .8; c.beginPath(); c.moveTo(X(x), oy); c.lineTo(X(x), oy + dh); c.moveTo(ox, Y(y)); c.lineTo(ox + dw, Y(y)); c.stroke();
    }
    c.fillStyle = '#8fa3b5'; c.font = '11px system-ui,sans-serif'; c.textAlign = 'left'; c.fillText(LAB[plane][0], ox + 4, oy + 13);
    c.textAlign = 'right'; c.fillText(LAB[plane][1], ox + dw - 4, oy + 13);
    if (o.demo) { c.fillStyle = '#c9a227'; c.font = '10px system-ui,sans-serif'; c.fillText('СХЕМА · ДЕМО', ox + dw - 4, oy + dh - 6); }
    c.textAlign = 'left';
    if (o.caption) { c.fillStyle = '#8fa3b5'; c.font = '11px system-ui,sans-serif'; c.fillText(o.caption, ox + 4, oy + dh - 6); }
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
export const xyzText = (c) => c.xyz.map(v => (v >= 0 ? '+' : '−') + Math.abs(Math.round(v))).join('; ');
export function opText(card, thr, fmt) {
  const row = (card.table || []).find(x => Math.abs(x.thr - thr) < 1e-6);
  if (!row) return '';
  return `на открытых данных: очаг найден у ${Math.round(row.sens * 100)} % пациентов с ФКД, в среднем ${fmt(row.fp)} ложных находок на человека без ФКД`;
}
