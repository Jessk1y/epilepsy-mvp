// Отчёт для врача: поле заключения обязательно — без него сервер отчёт не выдаёт. Печать и PDF — средствами браузера.
import { $, esc } from './ui.js';
import { api } from './api.js';

export function reportScreen(root, { tab, study, kind, rowsHTML, thr }) {
  const demo = study.demo;
  root.innerHTML = `<div class="report" id="draft"><div class="small">ЭпиАссистент · ${kind}${demo ? ' · ДЕМО' : ''}</div><h1>Отчёт для врача</h1>
   <div class="small">${demo ? 'Пациент: Демо-пациент (данные выдуманы)' : (study.code ? 'Код пациента: ' + esc(study.code) : 'Код пациента: не указан')} · ${esc(study.name)} · ${esc(study.date)}</div>
   <dl>${rowsHTML}</dl><h3>Заключение врача</h3>
   <textarea id="concl" placeholder="Заполняет врач. Без заключения врача отчёт не выдаётся."></textarea>
   <label class="field"><span>Врач (фамилия и инициалы)</span><input type="text" id="doctor" maxlength="80"></label>
   <p class="small">Результат программы — второе мнение и не является диагнозом. Окончательное решение принимает врач.</p>
   <div id="repErr"></div>
   <div class="actions noprint"><button class="btn" onclick="go('${tab}/result/${study.id}')">← К результату</button><button class="btn pri" id="issue" disabled>Выдать отчёт · Печать / PDF</button></div></div>
   <div id="issued"></div>`;
  const ta = $('#concl', root), btn = $('#issue', root);
  ta.oninput = () => { btn.disabled = ta.value.trim().length < 3; };
  btn.onclick = async () => {
    btn.disabled = true; $('#repErr', root).innerHTML = '';
    try {
      const r = await api.post('/api/reports', { study_id: study.id, doctor: $('#doctor', root).value, conclusion: ta.value, thr });
      $('#draft', root).style.display = 'none';
      $('#issued', root).innerHTML = `<div class="card noprint"><h2>Отчёт выдан</h2><p class="sub">№ ${esc(r.id)} · сохранён на сервере клиники. Чтобы получить PDF, в окне печати выберите «Сохранить как PDF».</p>
        <div class="actions"><span><button class="btn" onclick="go('${tab}/result/${study.id}')">← К результату</button> <a class="btn" href="${r.url}" target="_blank" style="text-decoration:none;color:inherit">Открыть отдельно</a></span>
        <button class="btn pri" id="printBtn">Печать / PDF</button></div></div><iframe class="repframe" id="repFrame" src="${r.url}" title="Отчёт"></iframe>`;
      $('#printBtn', root).onclick = () => $('#repFrame', root).contentWindow.print();
    } catch (e) {
      $('#repErr', root).innerHTML = `<div class="note bad">${esc(e.message)}</div>`; btn.disabled = ta.value.trim().length < 3;
    }
  };
}
