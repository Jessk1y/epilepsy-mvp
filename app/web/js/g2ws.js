// Рабочее место МРТ (концепт 3 в макете Дарьи): три плоскости с перекрестием, профиль вероятности по срезам,
// список кандидатов (клик — переход к очагу), рабочая точка.
import { $, $$, go, fmt, esc, state, toast } from './ui.js';
import { api } from './api.js';
import { Mri, PLANES, loadCands, xyzText, opText } from './mri.js';

const S = { seq: 'FLAIR', op: .7 };

export async function wsG2(root, id) {
  const list = await api.get('/api/g2/studies'), done = list.filter(x => x.status === 'done');
  if (!id && done.length) id = done[0].id;
  const left = (s) => `<aside class="left"><h4>Исследования</h4><div class="it new" onclick="go('g2/upload')">+ Новый пациент</div>
    ${done.map(x => `<div class="it ${x.id === id ? 'on' : ''}" onclick="go('ws/g2/${x.id}')">${esc(x.name)}<small>${x.flair5 ? 'FLAIR 5 мм ⚠ · ' : ''}${x.n_cand ? x.n_cand + ' канд. · ' + fmt(x.top) : 'без кандидатов'}</small></div>`).join('') || '<div class="mute">Исследований пока нет</div>'}
    ${s ? `<h4 style="margin-top:16px">Серии</h4><div class="it">T1<small>${esc(s.check.summary.t1)}</small></div><div class="it">FLAIR<small>${esc(s.check.summary.flair)}</small></div>` : ''}</aside>`;
  if (!id) {
    root.innerHTML = left(null) + `<section class="center"><div class="empty"><p><b>Исследований пока нет.</b></p><p>Загрузите T1 и FLAIR${state.meta.demo ? ' или возьмите демо-снимки' : ''} в пошаговом режиме — после расчёта исследование появится здесь.</p>
      <p><button class="b" onclick="go('g2/upload')">Загрузить снимки</button></p></div></section>`;
    return;
  }
  const s = await api.get('/api/g2/studies/' + id), r = s.result, card = s.card;
  if (!r) return go('g2/check/' + id);
  root.innerHTML = left(s) + '<section class="center"><div class="empty">Загрузка снимков…</div></section>';
  const mri = await Mri.load(id, r.info);
  const table = card.table || [], thrs = table.length ? table.map(x => x.thr) : null, lo = thrs ? thrs[0] : 0.3;
  let wp = (s.decision && s.decision.thr) || r.default_thr;
  let { above, below } = await loadCands(id, wp, lo);
  let sel = above.length ? 1 : 0, pos = above.length ? { i: above[0].ijk[0], j: above[0].ijk[1], k: above[0].ijk[2] } : mri.center(), saveT = null;
  const all = () => above.concat(below);

  const candList = () => all().map(c => `<div class="cand ${c.n === sel ? 'on' : ''} ${c.below ? 'dim' : ''}" data-n="${c.n}"><span><b>№${c.n}</b> ${c.hemi}<br><span class="mute">${fmt(c.volume_ml, 1)} см³ · (${xyzText(c)})</span></span><b>${fmt(c.peak_prob)}</b></div>`).join('') || '<div class="mute">Кандидатов нет</div>';
  const opLine = () => { const t = opText(card, wp, fmt); return t ? (card.demo ? 'Для примера (' : '') + t.charAt(0).toUpperCase() + t.slice(1) + (card.demo ? '; к демо-снимкам не относится).' : '.') : 'Для этой рабочей точки цифр на открытых данных нет.'; };
  const render = () => {
    root.innerHTML = left(s) + `<section class="center"><div class="tool"><button class="b ${S.seq === 'T1' ? 'on' : ''}" data-seq="T1">T1</button><button class="b ${S.seq === 'FLAIR' ? 'on' : ''}" data-seq="FLAIR">FLAIR</button>
      <span class="mute">Прозрачность карты</span><input type="range" id="op" min="0" max="1" step=".05" value="${S.op}" style="width:120px">
      <button class="b" id="hide">${S.op ? 'Скрыть карту' : 'Показать карту'}</button><span class="mute">Окно/уровень: авто</span></div>
      <div class="view"><div class="grid"><div class="cell"><canvas data-plane="ax"></canvas></div><div class="cell"><canvas data-plane="cor"></canvas></div><div class="cell"><canvas data-plane="sag"></canvas></div>
        <div class="cell" id="cpr" style="background:#fff;flex-direction:column;align-items:stretch;padding:8px 10px 6px"></div></div></div></section>
      <aside class="right">${s.check.flair5 ? `<div class="alert"><b>FLAIR 5 мм.</b> Толстые срезы резко снижают находимость очагов. Результат ниже — с этой оговоркой.</div>` : ''}
       ${card.demo ? '<div class="alert"><b>ДЕМО.</b> Снимки и карта — схема, модель не запускалась.</div>' : ''}
       <div class="box"><h4>Кандидаты</h4><div id="cl">${candList()}</div></div>
       <div class="box"><h4>Рабочая точка</h4><input type="range" id="wp" min="${thrs ? 0 : 0.1}" max="${thrs ? thrs.length - 1 : 0.95}" step="${thrs ? 1 : 0.05}" value="${thrs ? Math.max(0, thrs.indexOf(wp)) : wp}">
        <div class="line"><span class="mute">порог</span><span id="wpv">${fmt(wp)}${Math.abs(wp - r.default_thr) < 1e-6 ? ' (рабочая точка модели)' : ''}</span></div>
        <div class="mute" id="opl">${esc(opLine())}</div>
        ${card.table_note ? `<div class="mute" style="margin-top:6px">${esc(card.table_note)}</div>` : ''}</div>
       <div class="info">Второе мнение. Решение — за рентгенологом. Ложные находки и пропуски возможны.</div>
       <div class="box" style="margin-top:10px"><h4>Решение врача</h4><div class="btnrow"><button class="b" onclick="go('g2/report/${s.id}')">В отчёт</button><button class="b" onclick="go('g2/result/${s.id}')">Пошаговый вид</button></div></div>
       <div class="mute" style="margin-top:10px"><kbd>↑</kbd> <kbd>↓</kbd> аксиальные срезы · колесо мыши — срезы · клик по кандидату — переход</div></aside>`;
    $$('[data-seq]', root).forEach(b => b.onclick = () => { S.seq = b.dataset.seq; render(); });
    $('#op', root).oninput = e => { S.op = +e.target.value; draw(); };
    $('#hide', root).onclick = () => { S.op = S.op ? 0 : .7; render(); };
    $$('canvas[data-plane]', root).forEach(cv => {
      cv.onclick = e => { pos = mri.pick(cv, e, pos); draw(); };
      cv.onwheel = e => { e.preventDefault(); const d = e.deltaY < 0 ? 1 : -1, p = cv.dataset.plane; pos = mri.clampPos({ i: pos.i + (p === 'sag' ? d : 0), j: pos.j + (p === 'cor' ? d : 0), k: pos.k + (p === 'ax' ? d : 0) }); draw(); };
    });
    $('#wp', root).oninput = async e => {
      wp = thrs ? thrs[+e.target.value] : +e.target.value;
      $('#wpv', root).textContent = fmt(wp) + (Math.abs(wp - r.default_thr) < 1e-6 ? ' (рабочая точка модели)' : ''); $('#opl', root).textContent = opLine();
      const want = wp, res = await loadCands(id, wp, lo); if (want !== wp) return;
      ({ above, below } = res); if (!all().some(c => c.n === sel)) sel = above.length ? 1 : 0;
      $('#cl', root).innerHTML = candList(); bindCands(); draw();
      clearTimeout(saveT); saveT = setTimeout(() => api.post(`/api/g2/studies/${id}/decision`, { thr: wp }).catch(() => {}), 600);   // рабочая точка запоминается для отчёта
    };
    bindCands(); requestAnimationFrame(draw);
  };
  const bindCands = () => $$('.cand', root).forEach(el => el.onclick = () => { const c = all().find(x => x.n === +el.dataset.n); sel = c.n; pos = { i: c.ijk[0], j: c.ijk[1], k: c.ijk[2] }; $('#cl', root).innerHTML = candList(); bindCands(); draw(); });
  const draw = () => {
    $$('canvas[data-plane]', root).forEach(cv => { const p = cv.dataset.plane; mri.draw(cv, p, pos, { seq: S.seq, op: S.op, overlay: S.op > 0, cands: all(), sel, cross: true, demo: card.demo, caption: `${PLANES[p]} · ${S.seq} · ${mri.slice(p, pos) + 1}/${mri.geom(p).n}` }); });
    // профиль: наибольшая вероятность по аксиальным срезам
    const pr = mri.profile(), n = pr.length; let d = '';
    for (let k = 0; k < n; k++) d += (k ? 'L' : 'M') + (k / (n - 1) * 300).toFixed(1) + ',' + (100 - pr[k] * 90).toFixed(1);
    const cp = $('#cpr', root); if (!cp) return;
    cp.innerHTML = `<div class="mute">Макс. вероятность по аксиальным срезам</div><svg viewBox="0 0 300 104" style="width:100%;flex:1;cursor:pointer" preserveAspectRatio="none"><line x1="0" x2="300" y1="${100 - wp * 90}" y2="${100 - wp * 90}" stroke="#8b98a5" stroke-dasharray="4 4" vector-effect="non-scaling-stroke"/><path d="${d}" fill="none" stroke="#ff8a3d" stroke-width="2" vector-effect="non-scaling-stroke"/><line x1="${pos.k / (n - 1) * 300}" x2="${pos.k / (n - 1) * 300}" y1="0" y2="104" stroke="#4fa3ff" vector-effect="non-scaling-stroke"/></svg><div class="mute" style="display:flex;justify-content:space-between"><span>низ</span><span>пунктир — рабочая точка</span><span>верх</span></div>`;
    $('svg', cp).onclick = e => { const b = e.currentTarget.getBoundingClientRect(); pos = mri.clampPos({ ...pos, k: Math.round((e.clientX - b.left) / b.width * (n - 1)) }); draw(); };
  };
  const onKey = e => { if (e.target.tagName === 'INPUT') return; if (e.key === 'ArrowUp') { e.preventDefault(); pos = mri.clampPos({ ...pos, k: pos.k + 1 }); draw(); } if (e.key === 'ArrowDown') { e.preventDefault(); pos = mri.clampPos({ ...pos, k: pos.k - 1 }); draw(); } };
  const onResize = () => draw();
  addEventListener('keydown', onKey); addEventListener('resize', onResize);
  state.leave = () => { removeEventListener('keydown', onKey); removeEventListener('resize', onResize); clearTimeout(saveT); };
  render();
}
