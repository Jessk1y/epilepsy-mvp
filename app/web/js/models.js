// «О моделях»: паспорта моделей ЭЭГ и МРТ. Все цифры качества приходят с сервера из паспорта модели — со ссылкой на файл результатов.
import { fmt, esc, state } from './ui.js';

export async function modelsScreen(root) {
  const g1 = state.meta.g1, m = g1.model, c1 = m.card, g2 = state.meta.g2, c2 = g2.card, t = g1.threshold;
  const li = a => (a || []).map(x => `<li>${esc(x)}</li>`).join('');
  root.innerHTML = `<div class="pagecol"><div class="lab">Справка</div><h1>О моделях</h1>
    <p class="lead">Программа даёт второе мнение: решение принимает врач. Ниже — чему обучены модели, как они проверены и чего не умеют. Цифры качества получены на открытых данных и в другую клинику переносятся не полностью.</p>
    <div class="cards">
      <section><div class="lab">ЭЭГ · вероятность эпилепсии по записи</div><h2 style="margin-top:4px">${esc(m.title)} ${c1.demo ? '<span class="demo">ДЕМО</span>' : ''}</h2>
        <dl class="kv" style="grid-template-columns:150px 1fr"><dt>Задача при обучении</dt><dd>${esc(c1.task)}</dd><dt>Обучение</dt><dd>${esc(c1.trained_on)}</dd>
          <dt>Порог модели</dt><dd><span class="m">${fmt(c1.default_threshold)}</span> — ${esc(c1.default_threshold_source || '')}</dd>
          <dt>Порог сейчас</dt><dd><span class="m">${fmt(t.value)}</span> — ${esc(t.source)}${t.date ? ', ' + esc(t.date) : ''}. <a href="#clinic">Настройка под клинику</a></dd></dl>
        <h2>Качество на открытых данных</h2>
        ${(c1.metrics || []).length ? `<table class="t"><thead><tr><th>Данные</th><th>Значение</th><th>Источник</th></tr></thead><tbody>${c1.metrics.map(x => `<tr><td>${esc(x.data)}</td><td class="num">${esc(x.value)}</td><td class="src">${esc(x.source)}</td></tr>`).join('')}</tbody></table>` : '<p style="color:var(--ink2);margin:0">Для этой модели данных о качестве нет.</p>'}
        <h2>Чего модель не делает</h2><ul><li>Не показывает, где в записи разряды: оценка относится к записи целиком.</li><li>Не ставит диагноз.</li>${li(c1.limits)}</ul>
        <h2>Подключённые модели</h2>
        <table class="t"><tbody>${g1.models.map(x => `<tr><td>${esc(x.title)}${x.active ? ' <span class="o">работает сейчас</span>' : ''}</td><td class="src">${x.available ? 'готова к работе' : 'недоступна: ' + esc(x.reason)}</td></tr>`).join('')}</tbody></table>
      </section>
      <section><div class="lab">МРТ · поиск очагов ФКД</div><h2 style="margin-top:4px">${esc(c2.title)} ${c2.demo ? '<span class="demo">ДЕМО</span>' : ''}</h2>
        <dl class="kv" style="grid-template-columns:150px 1fr"><dt>Обучение</dt><dd>${esc(c2.trained_on)}</dd><dt>Рабочая точка модели</dt><dd><span class="m">${fmt(c2.default_thr)}</span> — меняется в исследовании</dd>
          ${g2.available ? '' : `<dt>Состояние</dt><dd class="w">недоступна: ${esc(g2.reason)}</dd>`}</dl>
        <h2>Рабочие точки</h2>
        ${(c2.table || []).length ? `<table class="t"><thead><tr><th>Порог</th><th>Очаг найден у пациентов с ФКД</th><th>Ложных находок на человека без ФКД</th></tr></thead><tbody>
          ${c2.table.map(x => `<tr class="${Math.abs(x.thr - c2.default_thr) < 1e-6 ? 'on' : ''}"><td class="num">${fmt(x.thr)}</td><td class="num">${Math.round(x.sens * 100)} %</td><td class="num">${fmt(x.fp)}</td></tr>`).join('')}</tbody></table>
          <p class="src" style="margin:8px 0 0">${esc(c2.table_note || '')}${c2.table_source ? ' Источник: ' + esc(c2.table_source) + '.' : ''}</p>` : '<p style="color:var(--ink2);margin:0">Таблицы рабочих точек для этой модели нет.</p>'}
        ${c2.flair5_note ? `<h2>Толстые срезы FLAIR</h2><p style="color:var(--ink2);margin:0">${esc(c2.flair5_note)}</p>` : ''}
        <h2>Чего модель не делает</h2><ul>${li(c2.limits)}<li>Не ставит диагноз: каждого кандидата оценивает рентгенолог.</li></ul>
      </section>
    </div></div>`;
}
