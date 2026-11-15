// Рабочее место ЭЭГ (концепт 3 в макете Дарьи): просмотр сигнала в центре, оценка модели — на панели справа.
// Сигнал настоящий (подготовленная запись с сервера), листается кусками.
import { $, $$, go, fmt, esc, state, drawEEG, mmss, progressHTML, progressTick, toast } from './ui.js';
import { api, waitJob } from './api.js';

const S = { mont: 'avg', uvmm: 0, sec: 10, filt: '0.5,70' };   // uvmm = 0 — усиление подбирается автоматически

export async function wsG1(root, id) {
  const list = await api.get('/api/g1/studies');
  const usable = list.filter(x => x.status !== 'rejected');
  if (!id && usable.length) id = (usable.find(x => x.status === 'done') || usable[0]).id;
  const left = () => `<aside class="left"><h4>Записи</h4>
    <div class="it new" onclick="go('g1/upload')">+ Новая запись</div>
    ${usable.map(x => `<div class="it ${x.id === id ? 'on' : ''}" onclick="go('ws/g1/${x.id}')">${esc(x.name)}<small>${esc(x.dur_text.replace(' мин ', ':').replace(' с', ''))} · ${x.score != null ? fmt(x.score) : 'нет оценки'}${x.low_quality ? ' ⚠' : ''}</small></div>`).join('') || '<div class="mute">Записей пока нет</div>'}
    <h4 style="margin-top:16px">Клиника</h4><div class="it" onclick="go('g1/clinic')">Порог: ${fmt(state.meta.g1.threshold.value)}<small>${state.meta.g1.threshold.clinic ? esc(state.meta.g1.threshold.source) : 'не настроен под клинику'}</small></div></aside>`;
  if (!id) {
    root.innerHTML = left() + `<section class="center"><div class="empty"><p><b>Записей пока нет.</b></p><p>Загрузите запись ЭЭГ в формате EDF${state.meta.demo ? ' или возьмите демо-запись' : ''}.</p>
      <p><button class="b" onclick="go('g1/upload')">Загрузить запись</button></p></div></section>`;
    return;
  }
  let s = await api.get('/api/g1/studies/' + id);
  const c = s.check, dur = c.summary.dur_s, arts = c.artifacts || [];
  let t0 = 0, sig = null, timer = null;

  const right = () => {
    const r = s.result, q = c.summary, cls = (st) => ({ ok: 'ok', warn: 'warn', bad: 'bad' }[st]);
    const item = (key) => c.items.find(i => i.key === key) || { status: 'ok' };
    const quality = `<div class="box"><h4>Качество записи</h4>
      <div class="line"><span>Длительность</span><span class="${cls(item('duration').status)}">${mmss(dur)}${item('duration').status === 'ok' ? ' ✓' : ' ⚠'}</span></div>
      <div class="line"><span>Каналы 10–20</span><span class="${cls(item('channels').status)}">${q.n_good}/19${c.flat.length ? ' ⚠' : ' ✓'}</span></div>
      <div class="line"><span>Артефакты</span><span class="${cls(item('artifacts').status)}">≈${q.art_pct} %${item('artifacts').status === 'ok' ? ' ✓' : ' ⚠'}</span></div>
      <div class="line"><span>Частота</span><span>${q.sfreq} → 200 Гц</span></div></div>`;
    if (!r) return `<aside class="right"><div class="box"><h4>Оценка модели · вся запись</h4><p class="mute">Оценка ещё не рассчитана.</p><div id="runBox"><button class="b on" id="runBtn">Запустить оценку</button></div></div>${quality}
      <div class="info">Модель оценивает запись целиком и <b>не указывает места разрядов</b>. Просмотр сигнала — для вашей собственной оценки.</div></aside>`;
    const t = r.threshold, d = s.decision || {};
    return `<aside class="right"><div class="box"><h4>Оценка модели · вся запись</h4><div style="display:flex;align-items:center;gap:10px"><span class="score">${fmt(r.score)}</span><span class="lbl ${r.above ? 'hi' : 'lo'}">${r.above ? 'ВЫШЕ ПОРОГА' : 'НИЖЕ ПОРОГА'}</span></div>
      <div class="thr"><u style="left:${t.value * 100}%"></u><i style="left:${r.score * 100}%"></i></div>
      <div class="line"><span class="mute">порог${t.clinic ? ' клиники' : ' (не настроен)'}</span><span>${fmt(t.value)}</span></div>
      ${r.spread ? `<div class="line"><span class="mute">разброс по частям записи</span><span>${fmt(r.spread[0])}–${fmt(r.spread[1])}</span></div>` : ''}
      <div class="line"><span class="mute">модель</span><span style="text-align:right">${esc(r.model_title)}</span></div>
      <div class="line"><span class="mute">время расчёта</span><span>${fmt(r.seconds, 0)} с</span></div></div>
      ${r.model_card.demo ? '<div class="alert"><b>ДЕМО-модель.</b> Оценка не имеет клинического смысла.</div>' : ''}
      ${!t.clinic ? '<div class="alert"><b>Порог не настроен под клинику.</b> <a href="#g1/clinic">Настроить</a></div>' : ''}
      ${quality}
      <div class="info">Модель оценивает запись целиком и <b>не указывает места разрядов</b>: оценка мало меняется по ходу записи (полоса под шкалой времени). Просмотр сигнала — для вашей собственной оценки.</div>
      <div class="box" style="margin-top:10px"><h4>Решение врача</h4><div class="btnrow"><button class="b ${d.agree === true ? 'on' : ''}" data-agree="1">Согласен</button><button class="b ${d.agree === false ? 'on' : ''}" data-agree="0">Не согласен</button><button class="b" onclick="go('g1/report/${s.id}')">В отчёт</button></div></div>
      <div class="mute"><kbd>←</kbd> <kbd>→</kbd> листать по ${S.sec} с</div></aside>`;
  };
  // полоса под шкалой времени: оценка модели по частям записи (ступеньками) и порог
  const stab = () => {
    const r = s.result; if (!r || !r.parts.length) return '<svg viewBox="0 0 1000 22" preserveAspectRatio="none"><rect width="1000" height="22" fill="#f7f5fd"/></svg>';
    const y = v => (22 - v * 22).toFixed(1); let d = '';
    r.parts.forEach((p, i) => { d += `${i ? 'L' : 'M'}${(p.t0 / dur * 1000).toFixed(1)},${y(p.score)} L${(p.t1 / dur * 1000).toFixed(1)},${y(p.score)} `; });
    return `<svg viewBox="0 0 1000 22" preserveAspectRatio="none"><rect width="1000" height="22" fill="#f7f5fd"/><line x1="0" x2="1000" y1="${y(r.threshold.value)}" y2="${y(r.threshold.value)}" stroke="#8b98a5" stroke-dasharray="6 6" vector-effect="non-scaling-stroke"/><path d="${d}" stroke="#4fa3ff" fill="none" stroke-width="1.5" vector-effect="non-scaling-stroke"/></svg>`;
  };
  const sel = (id2, opts, cur) => `<select id="${id2}">${opts.map(([v, t]) => `<option value="${v}" ${String(v) === String(cur) ? 'selected' : ''}>${t}</option>`).join('')}</select>`;
  const render = () => {
    root.innerHTML = left() + `<section class="center"><div class="tool">
      ${sel('mont', [['avg', 'Средний референт (как видит модель)'], ['bip', 'Продольный биполярный']], S.mont)}
      ${sel('uvmm', [[0, 'усиление: авто'], [5, '5 мкВ/мм'], [7, '7 мкВ/мм'], [10, '10 мкВ/мм'], [15, '15 мкВ/мм'], [20, '20 мкВ/мм']], S.uvmm)}
      ${sel('secs', [[10, '30 мм/с · 10 с'], [20, '15 мм/с · 20 с']], S.sec)}
      ${sel('filt', [['0.5,70', 'ФВЧ 0,5 · ФНЧ 70 · ' + state.meta.g1.notch_hz + ' Гц'], ['1,35', 'ФВЧ 1 · ФНЧ 35 · ' + state.meta.g1.notch_hz + ' Гц'], ['0,0', 'как у модели: 0,1–75 Гц']], S.filt)}
      <button class="b" id="prev">◀</button><span id="tm" class="mute"></span><button class="b" id="next">▶</button></div>
      <div class="view"><canvas class="eeg" id="eeg"></canvas></div>
      <div class="time"><div class="mute" style="margin-bottom:4px">${esc(s.name)} · вся запись ${mmss(dur)} · <span class="warn">■</span> артефакты (показатель качества) · ниже — оценка модели по частям записи${s.result ? ' и порог (пунктир)' : ''}</div>
       <div class="tl" id="tl">${arts.map(([a, b]) => `<i style="left:${a / dur * 100}%;width:${(b - a) / dur * 100}%"></i>`).join('')}<b id="pg"></b></div>
       <div class="stab" title="Оценка модели по частям записи">${stab()}</div></div></section>` + right();
    $('#mont', root).onchange = e => { S.mont = e.target.value; load(); };
    $('#uvmm', root).onchange = e => { S.uvmm = +e.target.value; draw(); };
    $('#secs', root).onchange = e => { S.sec = +e.target.value; render(); };
    $('#filt', root).onchange = e => { S.filt = e.target.value; load(); };
    $('#prev', root).onclick = () => page(-S.sec); $('#next', root).onclick = () => page(S.sec);
    $('#tl', root).onclick = e => { const r = e.currentTarget.getBoundingClientRect(); t0 = Math.round((e.clientX - r.left) / r.width * dur - S.sec / 2); load(); };
    $$('[data-agree]', root).forEach(b => b.onclick = async () => { s.decision = await api.post(`/api/g1/studies/${s.id}/decision`, { agree: b.dataset.agree === '1' }); render(); toast('Решение врача сохранено'); });
    const rb = $('#runBtn', root); if (rb) rb.onclick = run;
    load();
  };
  const draw = () => {
    const cv = $('#eeg', root); if (!cv || !sig) return;
    drawEEG(cv, sig, { uvmm: S.uvmm, auto: !S.uvmm, color: '#3a4670', labelColor: '#6b7387', grid: 'rgba(107,115,135,.16)', artColor: 'rgba(247,190,120,.25)', labelW: S.mont === 'bip' ? 56 : 44 });
    $('#tm', root).textContent = `${mmss(t0)}–${mmss(t0 + S.sec)}`;
    $('#pg', root).style.left = (t0 / dur * 100) + '%'; $('#pg', root).style.width = Math.max(.6, S.sec / dur * 100) + '%';
  };
  const load = () => {                                   // запрос куска сигнала; частые нажатия объединяются
    t0 = Math.max(0, Math.min(Math.max(0, dur - S.sec), t0));
    clearTimeout(timer);
    timer = setTimeout(async () => {
      const [hp, lp] = S.filt.split(',');
      const want = t0, x = await api.get(`/api/g1/studies/${s.id}/signal?t0=${t0}&sec=${S.sec}&montage=${S.mont}&hp=${hp}&lp=${lp}`);
      if (want !== t0 || !document.body.contains(root)) return;
      sig = x; draw();
    }, 30);
  };
  const page = (d) => { t0 += d; load(); };
  const run = async () => {
    const box = $('#runBox', root);
    try {
      const job = await api.post(`/api/g1/studies/${s.id}/run`);
      box.innerHTML = progressHTML(job.stages);
      const j = await waitJob(job.id, x => progressTick(box, x), () => document.body.contains(box));
      if (j) { s = await api.get('/api/g1/studies/' + s.id); render(); }
    } catch (e) { box.innerHTML = `<div class="alert">${esc(e.message)}</div>`; }
  };
  const onKey = e => { if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return; if (e.key === 'ArrowRight') page(S.sec); if (e.key === 'ArrowLeft') page(-S.sec); };
  const onResize = () => draw();
  addEventListener('keydown', onKey); addEventListener('resize', onResize);
  state.leave = () => { removeEventListener('keydown', onKey); removeEventListener('resize', onResize); clearTimeout(timer); };
  render();
}
