// Рабочее место МРТ: одна страница на все состояния — проверка снимков, поиск, кандидаты, решение врача.
// Три плоскости с перекрестием + окно «под перекрестием без карты» (карта не должна заслонять анатомию).
// Каждого кандидата выше рабочей точки врач отмечает сам: очаг / не очаг / не уверен. Без этого заключение закрыто.
import { $, $$, go, fmt, esc, state, keys, tabs, toast, progressHTML, progressTick, checkItems, plural, C } from './ui.js';
import { api, waitJob } from './api.js';
import { Mri, PLANES, loadCands, xyzText, candKey } from './mri.js';

const S = { seq: 'FLAIR', op: .7, map: true };
const MARK = { lesion: 'очаг', not: 'не очаг', unsure: 'не уверен' };

async function thumb(cv, sid, kind) {
  const t = await api.get(`/api/g2/studies/${sid}/thumb/${kind}`), bin = atob(t.data), off = document.createElement('canvas'); off.width = t.w; off.height = t.h;
  const c = off.getContext('2d'), img = c.createImageData(t.w, t.h);
  for (let q = 0; q < bin.length; q++) { const v = bin.charCodeAt(q); img.data[q * 4] = img.data[q * 4 + 1] = img.data[q * 4 + 2] = v; img.data[q * 4 + 3] = 255; }
  c.putImageData(img, 0, 0);
  const dpr = devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight, a = t.aspect || t.w / t.h, s = Math.min((W - 40) / a, H - 40), w = s * a;
  cv.width = W * dpr; cv.height = H * dpr; const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.imageSmoothingQuality = 'high';
  g.drawImage(off, (W - w) / 2, (H - s) / 2, w, s);
}

export async function mriScreen(root, id) {
  if (!id) return go('list');
  let s = await api.get('/api/g2/studies/' + id);
  if (s.status === 'upload' || !s.check) return go('new');
  tabs.open('mri', id, s.name);
  const c = s.check, card = s.card, demoA = card.demo || s.demo ? '<div class="alert demo-a"><b>ДЕМО.</b> «Снимки» и карта — схема; модель не запускалась.</div>' : '';
  const flairA = c.flair5 ? `<div class="alert"><b>▲ FLAIR 5 мм.</b> Толстые срезы: часть очагов могла быть пропущена.${card.flair5_note ? `<details><summary>Насколько это важно</summary>${esc(card.flair5_note)}</details>` : ''}</div>` : '';
  const alive = () => location.hash.includes(id);

  // ---------------------------------------------------------------- до расчёта: проверка снимков
  if (!s.result) {
    let running = false;
    const render = () => {
      root.innerHTML = `<div class="center"><div class="tool"><span class="lab">Снимки до обработки · средний аксиальный срез</span></div>
        <div class="grid two"><div class="vp"><canvas data-k="t1"></canvas><div class="c"><b>T1</b> · ${esc(c.summary.t1)}</div>${s.demo ? '<div class="dm">СХЕМА · ДЕМО</div>' : ''}</div>
        <div class="vp"><canvas data-k="flair"></canvas><div class="c"><b>FLAIR</b> · ${esc(c.summary.flair)}</div></div></div></div>
        <aside class="right">${demoA}${flairA}<div class="scroll">
          <div class="blk"><h3>${!c.can_run ? 'Поиск запустить нельзя' : c.flair5 ? 'Снимки с замечанием' : 'Снимки подходят'}</h3>
            <p>${!c.can_run ? 'Снимки не прошли проверку. Загрузите другие серии.' : c.flair5 ? 'Поиск запустить можно, но при толстых срезах FLAIR программа находит заметно меньше очагов. Если есть 3D FLAIR с тонкими срезами — лучше загрузить его.' : 'Проверка входа пройдена. Можно запускать поиск очагов.'}</p></div>
          <div class="blk"><div class="lab">Проверка снимков <span class="${c.low_quality ? 'w' : 'o'}">${c.low_quality ? 'есть замечания' : 'без замечаний'}</span></div>${checkItems(c.items)}
            ${(c.notes || []).map(n => `<p class="quiet w">${esc(n)}</p>`).join('')}</div>
        </div><div class="decide" id="runBox">${c.can_run
          ? `<div class="lab">Следующий шаг</div><p class="hint">${esc(card.eta || '')}</p><button class="btn pri lg wide" id="runBtn">${c.flair5 ? 'Всё равно найти очаги' : 'Найти очаги'} <kbd>⏎</kbd></button>`
          : `<button class="btn pri lg wide" onclick="go('new')">Загрузить другие снимки</button>`}</div></aside>`;
      $$('canvas[data-k]', root).forEach(cv => thumb(cv, id, cv.dataset.k).catch(() => {}));
      const rb = $('#runBtn', root); if (rb) rb.onclick = run;
    };
    const run = async () => {
      if (running || !c.can_run) return; running = true;
      try {
        const job = await api.post(`/api/g2/studies/${id}/run`);
        $('#runBox', root).innerHTML = `<div class="lab">Идёт поиск очагов</div>${progressHTML(job.stages)}`;
        const j = await waitJob(job.id, x => { const b = $('#runBox', root); if (b) progressTick(b, x); }, alive);
        if (j && alive()) go('mri/' + id);
      } catch (e) {
        running = false;
        if ($('#runBox', root)) { $('#runBox', root).innerHTML = `<div class="lab x">Поиск не выполнен</div><p class="hint">${esc(e.message)}</p><button class="btn lg wide" id="runBtn">Повторить</button>`; $('#runBtn', root).onclick = run; }
      }
    };
    render();
    const off = keys({ Enter: run }), onResize = () => render(); addEventListener('resize', onResize);
    state.leave = () => { off(); removeEventListener('resize', onResize); };
    return;
  }

  // ---------------------------------------------------------------- результат: просмотр и кандидаты
  const r = s.result;
  root.innerHTML = '<div class="loading">Загрузка снимков…</div>';
  const mri = await Mri.load(id, r.info);
  const table = card.table || [], thrs = table.length ? table.map(x => x.thr) : null, lo = thrs ? thrs[0] : 0.3;
  let wp = (s.decision && s.decision.thr) || r.default_thr, marks = { ...((s.decision && s.decision.marks) || {}) };
  let { above, below } = await loadCands(id, wp, lo);
  let sel = above.length ? 1 : 0, lastSave = Promise.resolve();
  const at = L => ({ i: L.ijk[0], j: L.ijk[1], k: L.ijk[2] });
  let pos = above.length ? at(above[0]) : mri.center();
  const all = () => above.concat(below), left = () => above.filter(L => !marks[candKey(L)]).length;
  // отметки и рабочая точка сохраняются сразу и по очереди; переход к заключению ждёт последнего сохранения
  const save = () => { lastSave = lastSave.then(() => api.post(`/api/g2/studies/${id}/decision`, { thr: wp, marks })).catch(e => toast('Отметки не сохранились: ' + e.message, true)); };
  const toReport = async () => { await lastSave; go('report/mri/' + id); };

  const candList = () => all().map(L => {
    const k = candKey(L), m = marks[k], on = L.n === sel;
    return `<button class="cand ${on ? 'on' : ''} ${L.below ? 'dim' : ''}" data-n="${L.n}" aria-pressed="${on}"><span class="n">${L.n}</span><span>${L.hemi} полушарие</span><span class="p">${fmt(L.peak_prob)}</span>
      <small>${fmt(L.volume_ml, 1)} см³ · ${xyzText(L)}${L.below ? ' · ниже рабочей точки' : ''}</small>${m && !on ? `<span class="mark">врач: ${MARK[m]}</span>` : ''}</button>
      ${on && !L.below ? `<div class="mk">${Object.entries(MARK).map(([v, t], i) => `<button class="btn ${m === v ? 'on' : ''}" data-mark="${v}" aria-pressed="${m === v}">${t[0].toUpperCase() + t.slice(1)}${i < 2 ? ` <kbd>${i ? 'Н' : 'О'}</kbd>` : ''}</button>`).join('')}</div>` : ''}`;
  }).join('') || '<div class="blk"><p style="margin:0">Выше рабочей точки кандидатов нет. Это не исключает очаг: часть очагов программа не находит.</p></div>';
  const opRows = () => {
    const row = table.find(x => Math.abs(x.thr - wp) < 1e-6);
    return row ? `<div class="rows" style="margin-top:10px"><span>Очаг найден у пациентов с ФКД</span><span>${Math.round(row.sens * 100)} %</span><span>Ложных находок на человека без ФКД</span><span>${fmt(row.fp)}</span></div>
      <details class="more"><summary>Откуда цифры</summary><p>${esc(card.table_note || '')}${card.table_source ? ' Источник: ' + esc(card.table_source) + '.' : ''}</p></details>` : '<p class="quiet">Для этой рабочей точки цифр на открытых данных нет.</p>';
  };
  const decideBox = () => {
    const n = left();
    return `<div class="lab">Решение врача</div><p class="hint">${n ? `Отметьте ${n === 1 && above.length === 1 ? 'кандидата №1' : `ещё ${n} ${plural(n, 'кандидата', 'кандидатов', 'кандидатов')}`} — после этого откроется заключение.` : above.length ? 'Все кандидаты выше рабочей точки отмечены.' : 'Кандидатов нет — заключение за вами.'}</p>
      <button class="btn pri lg wide" id="toRep" ${n ? 'disabled' : ''}>${s.report ? 'Заключение (отчёт уже выдан)' : 'Заключение'} <kbd>⏎</kbd></button>`;
  };

  const render = () => {
    root.innerHTML = `<div class="center"><div class="tool">
        <div class="seg" role="group" aria-label="Серия"><button class="${S.seq === 'T1' ? 'on' : ''}" data-seq="T1">T1</button><button class="${S.seq === 'FLAIR' ? 'on' : ''}" data-seq="FLAIR">FLAIR</button></div>
        <span class="gap"></span><button class="btn ${S.map ? 'on' : ''}" id="map" style="height:26px" aria-pressed="${S.map}">Карта <kbd>К</kbd></button>
        <label>прозрачность <input type="range" id="op" min="0.2" max="1" step=".05" value="${S.op}" style="width:110px" aria-label="Прозрачность карты" ${S.map ? '' : 'disabled'}></label>
        <span class="note2"><kbd>1</kbd>–<kbd>9</kbd> кандидат · <kbd>↑</kbd><kbd>↓</kbd> срезы · колесо — срезы · щелчок — перекрестие</span></div>
      <div class="grid">${['ax', 'cor', 'sag'].map(p => `<div class="vp"><canvas data-plane="${p}" role="img" aria-label="${PLANES[p]} срез"></canvas><div class="c"><b>${PLANES[p]}</b> · <span data-seqlab></span></div><div class="s" data-sl="${p}"></div></div>`).join('')}
        <div class="vp"><canvas data-zoom role="img" aria-label="Увеличение под перекрестием без карты"></canvas><div class="c"><b>Под перекрестием без карты</b> · <span data-seqlab></span> ×2,5</div>${card.demo ? '<div class="dm">СХЕМА · ДЕМО</div>' : ''}</div></div></div>
      <aside class="right">${demoA}${flairA}<div class="scroll">
        <div class="blk" style="border:0;padding-bottom:6px"><div class="lab" style="margin:0">Кандидаты в очаги <span>${above.length} выше рабочей точки</span></div></div>
        <div id="cl">${candList()}</div>
        <div class="blk"><div class="lab">Рабочая точка <span>порог вероятности</span></div>
          ${thrs ? `<div class="wp" role="group" aria-label="Рабочая точка">${thrs.map(t => `<button class="${Math.abs(t - wp) < 1e-6 ? 'on' : ''}" data-wp="${t}">${fmt(t, t * 100 % 10 ? 2 : 1)}</button>`).join('')}</div>` : `<input type="range" id="wpr" min="0.1" max="0.95" step="0.05" value="${wp}" style="width:100%"><div class="rows"><span>порог</span><span>${fmt(wp)}</span></div>`}
          <div id="opr">${opRows()}</div></div>
        <div class="blk prof"><div class="lab">Вероятность по срезам <span>щелчок — к срезу</span></div><div id="prof"></div></div>
        <div class="blk"><div class="lab">Модель <span><a href="#models">подробнее</a></span></div><div>${esc(card.title)}</div>
          <p class="quiet">${esc((card.limits || []).join(' '))} Это второе мнение: решение принимает рентгенолог.</p></div>
      </div><div class="decide" id="dec">${decideBox()}</div></aside>`;
    $$('[data-seq]', root).forEach(b => b.onclick = () => { S.seq = b.dataset.seq; render(); });
    $('#map', root).onclick = toggleMap;
    $('#op', root).oninput = e => { S.op = +e.target.value; draw(); };
    $$('canvas[data-plane]', root).forEach(cv => {
      cv.onclick = e => { pos = mri.pick(cv, e, pos); draw(); };
      cv.onwheel = e => { e.preventDefault(); step(cv.dataset.plane, e.deltaY < 0 ? 1 : -1); };
    });
    $$('[data-wp]', root).forEach(b => b.onclick = () => setWp(+b.dataset.wp));
    const wr = $('#wpr', root); if (wr) wr.onchange = e => setWp(+e.target.value);
    bindSide(); requestAnimationFrame(draw);
  };
  const bindSide = () => {
    $$('.cand', root).forEach(el => el.onclick = () => pick(+el.dataset.n));
    $$('[data-mark]', root).forEach(b => b.onclick = () => mark(b.dataset.mark));
    const tr = $('#toRep', root); if (tr) tr.onclick = toReport;
  };
  const side = () => { $('#cl', root).innerHTML = candList(); $('#dec', root).innerHTML = decideBox(); bindSide(); };
  const pick = (n) => { const L = all().find(x => x.n === n); if (!L) return; sel = n; pos = at(L); side(); draw(); };
  const mark = (v) => {
    const L = above.find(x => x.n === sel); if (!L) return;
    const k = candKey(L); if (marks[k] === v) delete marks[k]; else marks[k] = v;
    save(); side();
    if (marks[k]) { const nx = above.find(x => !marks[candKey(x)]); if (nx) pick(nx.n); }      // к следующему неотмеченному
  };
  const setWp = async (v) => {
    wp = v; const want = wp, res = await loadCands(id, wp, lo); if (want !== wp || !alive()) return;
    ({ above, below } = res); if (!all().some(x => x.n === sel)) sel = above.length ? 1 : 0;
    save(); render();
  };
  const toggleMap = () => { S.map = !S.map; render(); };
  const step = (plane, d) => { pos = mri.clampPos({ i: pos.i + (plane === 'sag' ? d : 0), j: pos.j + (plane === 'cor' ? d : 0), k: pos.k + (plane === 'ax' ? d : 0) }); draw(); };
  const draw = () => {
    const o = { seq: S.seq, op: S.op, overlay: S.map, cands: all(), sel, cross: true };
    $$('canvas[data-plane]', root).forEach(cv => mri.draw(cv, cv.dataset.plane, pos, o));
    const z = $('canvas[data-zoom]', root); if (z) mri.draw(z, 'ax', pos, { seq: S.seq, op: 0, overlay: false, cands: all().filter(L => L.n === sel), sel: -1, cross: false, zoom: 2.5 });
    $$('[data-seqlab]', root).forEach(e => e.textContent = S.seq);
    $$('[data-sl]', root).forEach(e => { const p = e.dataset.sl; e.textContent = `${mri.slice(p, pos) + 1}/${mri.geom(p).n}`; });
    // профиль: наибольшая вероятность по аксиальным срезам; пунктир — рабочая точка
    const pr = mri.profile(), n = pr.length, W = 306, H = 50, box = $('#prof', root); if (!box) return;
    let d = ''; for (let k = 0; k < n; k++) d += (k ? 'L' : 'M') + (k / (n - 1) * W).toFixed(1) + ',' + (H - 3 - pr[k] * (H - 6)).toFixed(1);
    const yt = H - 3 - wp * (H - 6), xk = pos.k / (n - 1) * W;
    box.innerHTML = `<svg viewBox="0 0 ${W} ${H + 13}" aria-hidden="true"><line x1="0" x2="${W}" y1="${yt}" y2="${yt}" stroke="${C.ink2}" stroke-dasharray="3 3"/><path d="${d}" fill="none" stroke="${C.ai}" stroke-width="1.5"/><line x1="${xk}" x2="${xk}" y1="0" y2="${H}" stroke="#fff"/>
      <text x="0" y="${H + 11}" font-size="10" fill="${C.ink3}">низ</text><text x="${W}" y="${H + 11}" font-size="10" fill="${C.ink3}" text-anchor="end">верх</text><text x="${Math.max(24, Math.min(W - 24, xk))}" y="${H + 11}" font-size="10" fill="${C.ink}" text-anchor="middle">срез ${pos.k + 1}</text></svg>`;
    $('svg', box).onclick = e => { const b = e.currentTarget.getBoundingClientRect(); pos = mri.clampPos({ ...pos, k: Math.round((e.clientX - b.left) / b.width * (n - 1)) }); draw(); };
  };

  render();
  const km = { ArrowUp: () => step('ax', 1), ArrowDown: () => step('ax', -1), KeyR: toggleMap, KeyJ: () => mark('lesion'), KeyY: () => mark('not'),
    Enter: () => { if (!left()) toReport(); else toast('Сначала отметьте кандидатов выше рабочей точки', true); } };
  for (let i = 1; i <= 9; i++) km['Digit' + i] = km['Numpad' + i] = () => pick(i);
  const off = keys(km), onResize = () => draw(); addEventListener('resize', onResize);
  state.leave = () => { off(); removeEventListener('resize', onResize); };
}
