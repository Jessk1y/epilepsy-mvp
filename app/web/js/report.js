// Заключение врача и отчёт. Слева — лист отчёта, как он будет выдан; справа — заключение, без которого сервер отчёт не выдаёт.
// Содержимое выданного отчёта собирает сервер из сохранённого исследования; лист здесь — предпросмотр тех же строк.
import { $, $$, go, fmt, esc, state, keys, tabs, toast } from './ui.js';
import { api } from './api.js';
import { loadCands, xyzText, candKey } from './mri.js';

const MARK = { lesion: 'очаг', not: 'не очаг', unsure: 'не уверен' };

function eegRows(s) {
  const c = s.check, r = s.result, t = r.threshold, d = s.decision || {};
  const rows = [['Длительность / каналы', `${esc(c.summary.dur_text)} · ${c.summary.n_good} из 19${c.flat.length ? ' (без сигнала: ' + esc(c.flat.join(', ')) + ')' : ''}`],
    ['Артефакты', `≈ ${c.summary.art_pct} % записи`],
    ['Оценка модели', `<b>${fmt(r.score)}</b> (порог ${fmt(t.value)}) — ${r.above ? 'выше' : 'ниже'} порога`],
    ['Порог', esc(t.source) + (t.date ? ', ' + esc(t.date) : '')], ['Вывод модели', esc(r.verdict)]];
  if (r.spread) rows.push(['Устойчивость', `оценки по частям записи — от ${fmt(r.spread[0])} до ${fmt(r.spread[1])}`]);
  if (c.low_quality) rows.push(['Качество входа', 'ниже нормы: ' + esc((c.low_reasons || []).join(', ')) + ' — доверие к оценке снижено']);
  if (d.agree != null) rows.push(['Решение врача', d.agree ? 'согласен с оценкой программы' : 'не согласен с оценкой программы']);
  rows.push(['Модель', esc(r.model_title) + (r.model_card.demo ? ' — ДЕМО, оценка не имеет клинического смысла' : '')]);
  rows.push(['Ограничения', 'Модель не локализует события: оценка относится ко всей записи. ' + esc((r.model_card.limits || []).join(' '))]);
  return rows;
}
async function mriRows(s, wp) {
  const c = s.check, card = s.card, marks = (s.decision && s.decision.marks) || {}, { above } = await loadCands(s.id, wp, null);
  const row = (card.table || []).find(x => Math.abs(x.thr - wp) < 1e-6);
  const rows = [['Снимки', `T1 ${esc(c.summary.t1)} · FLAIR ${esc(c.summary.flair)}${c.flair5 ? ' (толстые срезы — низкое качество)' : ''}`],
    ['Кандидаты выше рабочей точки', above.length ? above.map(L => `№${L.n}: вероятность ${fmt(L.peak_prob)}, ${fmt(L.volume_ml, 1)} см³, (${xyzText(L)}) мм, ${L.hemi} полушарие${marks[candKey(L)] ? ` — <b>врач: ${MARK[marks[candKey(L)]]}</b>` : ''}`).join('<br>') : 'нет'],
    ['Рабочая точка', `порог ${fmt(wp)}` + (row && !card.demo ? ` — на открытых данных: очаг найден у ${Math.round(row.sens * 100)} % пациентов с ФКД, ${fmt(row.fp)} ложных находок на человека без ФКД` : '')]];
  if (c.flair5) rows.push(['Качество входа', 'FLAIR со срезами 5 мм: часть очагов могла быть пропущена. ' + esc(card.flair5_note || '')]);
  rows.push(['Модель', esc(card.title) + (card.demo ? ' — ДЕМО, снимки и карта — схема' : '')]);
  rows.push(['Ограничения', esc((card.limits || []).join(' '))]);
  return rows;
}

export async function reportScreen(root, kind, id) {
  if (!id || (kind !== 'eeg' && kind !== 'mri')) return go('list');
  const eeg = kind === 'eeg', s = await api.get(`/api/${eeg ? 'g1' : 'g2'}/studies/${id}`);
  if (!s.result) return go(`${kind}/${id}`);
  tabs.open(kind, id, s.name);
  const wp = eeg ? undefined : ((s.decision && s.decision.thr) || s.result.default_thr);
  const rows = eeg ? eegRows(s) : await mriRows(s, wp), demo = s.demo, m = state.meta;
  const quick = eeg ? ['Согласен с оценкой программы.', 'Не согласен с оценкой программы: ', 'Рекомендуется повторить запись.'] : ['Изменения соответствуют кандидату программы.', 'Признаков ФКД не выявлено.', 'Рекомендуется МРТ с 3D FLAIR.'];

  root.innerHTML = `<div class="desk"><div class="paper" id="draft">
      <div class="hd"><div class="small">ЭпиАссистент · ${eeg ? 'ЭЭГ' : 'МРТ'} · ${esc(m.clinic)} ${demo ? '<span class="demo">ДЕМО · ДАННЫЕ ВЫДУМАНЫ</span>' : ''}</div><h1>Отчёт для врача</h1>
      <div class="small">${demo ? 'Пациент: Демо-пациент (данные выдуманы)' : (s.code ? 'Код пациента: ' + esc(s.code) : 'Код пациента: не указан')} · ${esc(s.name)} · ${esc(s.date)}</div></div>
      <dl>${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('')}</dl>
      <h3>Заключение врача</h3><div class="concl ph" id="mirror">Появится здесь, когда вы его напишете.</div>
      <p class="small" style="margin-top:22px">Результат программы — второе мнение и не является диагнозом. Окончательное решение принимает врач.</p></div></div>
    <aside class="right" id="side"><div class="scroll">
      <div class="blk"><h3>Заключение врача</h3><p>Без него отчёт не выдаётся. Программа сюда ничего не вписывает.</p>
        <textarea id="concl" rows="8" aria-label="Заключение врача" placeholder="Ваше заключение"></textarea>
        <div class="quick" aria-label="Вставить готовую фразу">${quick.map(q => `<button class="btn" data-q="${esc(q)}">${esc(q.replace(/[.:]\s*$/, ''))}</button>`).join('')}</div>
        <label class="fld"><span>Врач — фамилия и инициалы</span><input type="text" id="doctor" maxlength="80" autocomplete="off"></label></div>
      ${s.report ? `<div class="blk"><div class="lab">Уже выдан</div><p style="margin:0">По этому исследованию есть отчёт № ${esc(s.report)}. <a href="/reports/${esc(s.report)}" target="_blank">Открыть</a>. Новый отчёт получит свой номер.</p></div>` : ''}
      </div><div class="decide"><p class="hint" id="why">Напишите заключение — кнопка станет доступна.</p><div id="repErr" role="alert"></div>
        <button class="btn pri lg wide" id="issue" disabled>Выдать отчёт</button>
        <button class="btn ghost wide" style="margin-top:6px" id="backBtn">Назад к ${eeg ? 'записи' : 'снимкам'} <kbd>Esc</kbd></button></div></aside>`;
  const ta = $('#concl', root), btn = $('#issue', root), doc = $('#doctor', root), mirror = $('#mirror', root);
  $('#backBtn', root).onclick = () => go(`${kind}/${id}`);
  try { doc.value = localStorage.getItem('epi.doctor') || ''; } catch (e) { /* хранилище недоступно */ }
  const sync = () => {
    const v = ta.value.trim(), ok = v.length >= 3;
    btn.disabled = !ok; $('#why', root).textContent = ok ? 'Отчёт сохранится на сервере клиники и получит номер.' : 'Напишите заключение — кнопка станет доступна.';
    mirror.textContent = v || 'Появится здесь, когда вы его напишете.'; mirror.classList.toggle('ph', !v);
  };
  ta.oninput = sync;
  $$('[data-q]', root).forEach(b => b.onclick = () => { ta.value = (ta.value.trim() ? ta.value.replace(/\s*$/, ' ') : '') + b.dataset.q; sync(); ta.focus(); });
  btn.onclick = async () => {
    btn.disabled = true; $('#repErr', root).innerHTML = '';
    try {
      try { localStorage.setItem('epi.doctor', doc.value.trim()); } catch (e) { /* не страшно */ }
      const r = await api.post('/api/reports', { study_id: id, doctor: doc.value, conclusion: ta.value, thr: wp });
      $('.desk', root).outerHTML = `<iframe class="repframe" id="repFrame" src="${r.url}" title="Выданный отчёт"></iframe>`;
      $('#side', root).innerHTML = `<div class="scroll"><div class="blk"><div class="lab o">Отчёт выдан</div><h3>№ ${esc(r.id)}</h3>
          <p>Сохранён на сервере клиники. Чтобы получить PDF, в окне печати выберите «Сохранить как PDF».</p></div></div>
        <div class="decide"><button class="btn pri lg wide" id="printBtn">Печать / PDF</button>
          <a class="btn wide" style="margin-top:6px" href="${r.url}" target="_blank">Открыть в отдельном окне</a>
          <button class="btn ghost wide" style="margin-top:6px" id="toList">К списку исследований</button></div>`;
      const fr = $('#repFrame', root);
      fr.onload = () => { try { const b = fr.contentDocument.querySelector('.bar'); if (b) b.style.display = 'none'; } catch (e) { /* кнопка печати останется и внутри листа */ } };
      $('#printBtn', root).onclick = () => fr.contentWindow.print();
      $('#toList', root).onclick = () => go('list');
      toast('Отчёт выдан');
    } catch (e) {
      $('#repErr', root).innerHTML = `<p class="hint x">${esc(e.message)}</p>`; sync();
    }
  };
  state.leave = keys({ Escape: () => go(`${kind}/${id}`) });
  ta.focus();
}
