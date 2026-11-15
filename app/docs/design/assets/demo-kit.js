// Общие помощники макетов: рисование сигнала ЭЭГ и срезов МРТ из demo-data.js, переключение экранов.
// Оформление (цвета, толщины, шрифты) задаёт каждое направление само — здесь только геометрия.
(function () {
  const D = window.DEMO, [NX, NY, NZ] = D.shape, [I0, J0, K0] = D.pos;
  const cache = {};
  function pixels(seq, plane) {                       // Promise<{w,h,data:Uint8ClampedArray (серый)}>
    const key = seq + plane;
    if (!cache[key]) cache[key] = new Promise(res => {
      const im = new Image();
      im.onload = () => {
        const c = document.createElement('canvas'); c.width = im.width; c.height = im.height;
        const g = c.getContext('2d'); g.drawImage(im, 0, 0);
        res({ w: im.width, h: im.height, data: g.getImageData(0, 0, im.width, im.height).data });
      };
      im.src = D.img[seq][plane];
    });
    return cache[key];
  }
  const RAMPS = {
    // жёлто-красная «тепловая» (как сейчас в приложении)
    heat: u => [255, 32 + 211 * u * u, 32 + 58 * u],
    // янтарная одного тона: яркость растёт с вероятностью
    amber: u => [255, 150 + 90 * u, 20 + 60 * u * u],
    // холодная бирюзовая — не спорит с красным «плохо»
    cyan: u => [40 + 60 * u, 200 + 55 * u, 230],
    // пурпурная — хорошо отделяется от серого снимка и от статусов
    magenta: u => [235 + 20 * u, 60 + 90 * u * u, 190 - 40 * u],
  };
  // положение вокселя на экране плоскости (в вокселях плоскости)
  function toScreen(plane, ijk) {
    const [i, j, k] = ijk;
    if (plane === 'ax') return [NX - 1 - i, NY - 1 - j];
    if (plane === 'cor') return [NX - 1 - i, NZ - 1 - k];
    return [NY - 1 - j, NZ - 1 - k];
  }
  // o: {seq:'flair'|'t1', overlay:true, op:.75, ramp:'heat', cross:'#…'|null, ring:'#…'|null, ringLabel:true,
  //     label:'#…', font:'11px …', bg:'#000', pad:0, sides:true, crossGap:10}
  async function slice(cv, plane, o = {}) {
    const base = await pixels(o.seq || 'flair', plane), prob = await pixels('prob', plane);
    const { w, h } = base, off = document.createElement('canvas'); off.width = w; off.height = h;
    const g0 = off.getContext('2d'), img = g0.createImageData(w, h), d = img.data, ramp = RAMPS[o.ramp || 'heat'], op = o.overlay === false ? 0 : (o.op ?? .75);
    for (let q = 0; q < w * h; q++) {
      const b = base.data[q * 4], p = prob.data[q * 4] / 255; let r = b, gr = b, bl = b;
      if (op > 0 && p > 0.08) { const a = op * Math.min(1, (p - .05) / .45), c = ramp(Math.min(1, p / .9)); r = r * (1 - a) + c[0] * a; gr = gr * (1 - a) + c[1] * a; bl = bl * (1 - a) + c[2] * a; }
      d[q * 4] = r; d[q * 4 + 1] = gr; d[q * 4 + 2] = bl; d[q * 4 + 3] = 255;
    }
    g0.putImageData(img, 0, 0);
    const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight; if (!W || !H) return;
    cv.width = W * dpr; cv.height = H * dpr;
    const c = cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.fillStyle = o.bg || '#000'; c.fillRect(0, 0, W, H);
    const pad = o.pad || 0, s = Math.min((W - 2 * pad) / w, (H - 2 * pad) / h) * (o.zoom || 1), dw = w * s, dh = h * s;
    const [cx, cy] = toScreen(plane, D.pos);
    let ox = (W - dw) / 2, oy = (H - dh) / 2;
    if (o.zoom && o.zoom > 1) { ox = W / 2 - (cx + .5) * s; oy = H / 2 - (cy + .5) * s; }
    c.imageSmoothingEnabled = true; c.imageSmoothingQuality = 'high'; c.drawImage(off, ox, oy, dw, dh);
    const X = ox + (cx + .5) * s, Y = oy + (cy + .5) * s, vol = D.cands[0].volume_ml, rmm = Math.cbrt(3 * vol * 1000 / (4 * Math.PI)) + 2, rr = rmm / 1.25 * s * 1.15;
    if (o.cross) {
      const gap = o.crossGap ?? rr + 6; c.strokeStyle = o.cross; c.lineWidth = 1;
      c.beginPath(); c.moveTo(X + .5, Math.max(0, oy)); c.lineTo(X + .5, Y - gap); c.moveTo(X + .5, Y + gap); c.lineTo(X + .5, Math.min(H, oy + dh));
      c.moveTo(Math.max(0, ox), Y + .5); c.lineTo(X - gap, Y + .5); c.moveTo(X + gap, Y + .5); c.lineTo(Math.min(W, ox + dw), Y + .5); c.stroke();
    }
    if (o.ring) {
      c.strokeStyle = o.ring; c.lineWidth = o.ringW || 1.5; if (o.ringDash) c.setLineDash(o.ringDash);
      c.beginPath(); c.arc(X, Y, rr, 0, 6.2832); c.stroke(); c.setLineDash([]);
      if (o.ringLabel !== false) { c.fillStyle = o.ring; c.font = o.ringFont || '700 12px system-ui,sans-serif'; c.fillText(o.ringText || '1', X + rr * .75 + 4, Y - rr * .75 - 2); }
    }
    if (o.sides !== false) {
      const lab = plane === 'sag' ? ['Перед', 'Зад'] : ['П', 'Л'];
      c.fillStyle = o.label || '#8fa3b5'; c.font = o.font || '11px system-ui,sans-serif'; c.textBaseline = 'middle';
      c.textAlign = 'left'; c.fillText(lab[0], Math.max(0, ox) + 8, H / 2); c.textAlign = 'right'; c.fillText(lab[1], Math.min(W, ox + dw) - 8, H / 2); c.textAlign = 'left'; c.textBaseline = 'alphabetic';
    }
  }
  // o: {color, bad, label, grid, art, labelW, font, lw, gain (доля высоты строки на типичный размах), scale:'#…'|null}
  function eeg(cv, sig, o = {}) {
    const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight; if (!W || !H) return;
    cv.width = W * dpr; cv.height = H * dpr;
    const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
    const ch = sig.names, L = o.labelW ?? 44, rowH = H / ch.length, pw = W - L, sec = sig.sec;
    const a = []; sig.data.forEach((d, i) => { if (!sig.bad.includes(ch[i])) for (let k = 0; k < d.length; k += 5) a.push(Math.abs(d[k])); });
    a.sort((x, y) => x - y); const px = rowH * (o.gain ?? .2) / ((a[a.length >> 1] || 1) * 1.48);
    (sig.artifacts || []).forEach(([s, e]) => { g.fillStyle = o.art || 'rgba(240,160,40,.18)'; g.fillRect(L + pw * s / sec, 0, pw * (e - s) / sec, H); if (o.artEdge) { g.fillStyle = o.artEdge; g.fillRect(L + pw * s / sec, 0, pw * (e - s) / sec, 2); } });
    g.strokeStyle = o.grid || 'rgba(128,128,128,.2)'; g.lineWidth = 1;
    for (let s = 0; s <= sec; s++) { const x = Math.round(L + pw * s / sec) + .5; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H); g.stroke(); }
    g.save(); g.beginPath(); g.rect(L, 0, pw, H); g.clip(); g.lineJoin = 'round';
    ch.forEach((name, i) => {
      const y0 = rowH * (i + .5), bad = sig.bad.includes(name), d = sig.data[i], n = d.length;
      g.strokeStyle = bad ? (o.bad || '#d33') : (o.color || '#23374d'); g.lineWidth = o.lw || 1; g.beginPath();
      for (let k = 0; k < n; k++) { const x = L + pw * k / (n - 1), y = y0 - Math.max(-rowH * 1.6, Math.min(rowH * 1.6, d[k] * px)); k ? g.lineTo(x, y) : g.moveTo(x, y); }
      g.stroke();
    });
    g.restore();
    g.font = o.font || '10px system-ui,sans-serif'; g.textBaseline = 'middle';
    ch.forEach((name, i) => { g.fillStyle = sig.bad.includes(name) ? (o.bad || '#d33') : (o.label || '#667'); g.fillText(name, o.labelX ?? 6, rowH * (i + .5)); });
    if (o.scale) {
      const bx = W - 14, by = H - 10, bh = 100 / sig.unit * px;
      g.strokeStyle = o.scale; g.fillStyle = o.scale; g.lineWidth = 1.5; g.beginPath(); g.moveTo(bx, by - bh); g.lineTo(bx, by); g.lineTo(bx - pw / sec, by); g.stroke();
      g.textAlign = 'right'; g.fillText('100 мкВ · 1 с', bx - 6, by - bh / 2 - 6); g.textAlign = 'left';
    }
  }
  // экраны макета: <section data-screen="home|eeg|mri">, переключение по адресу после #
  function screens(onShow) {
    const all = [...document.querySelectorAll('[data-screen]')];
    const show = () => {
      const h = location.hash.slice(1) || all[0].dataset.screen;
      all.forEach(s => s.hidden = s.dataset.screen !== h);
      document.querySelectorAll('[data-go]').forEach(b => b.classList.toggle('on', b.dataset.go === h));
      document.documentElement.dataset.screen = h;
      requestAnimationFrame(() => onShow && onShow(h)); scrollTo(0, 0);
    };
    document.addEventListener('click', e => { const b = e.target.closest('[data-go]'); if (b) { e.preventDefault(); location.hash = b.dataset.go; } });
    addEventListener('hashchange', show); let t; addEventListener('resize', () => { clearTimeout(t); t = setTimeout(show, 80); }); show();
  }
  const fmt = (x, d = 2) => Number(x).toFixed(d).replace('.', ',');
  window.Kit = { slice, eeg, screens, toScreen, fmt, D };
})();
