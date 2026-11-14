# TUEP v3.1.0 (TUH): скачать → разобрать → подготовить → прогнать

Код и инструкции для рабочей машины (Ubuntu, RTX A5000, **одна карта**). Данные TUEP и веса **в git не кладём**, на Kaggle не загружаем, пока NEDC письменно не подтвердил, что облако можно (условия — «не передавать третьим лицам»). Ноутбук Захара не нагружаем.

> **Протокол прогона — [PROTOCOL.md](PROTOCOL.md)** (записан 11.11.2026 до прогона). Сейчас TUEP скачивается с Яндекс Диска руководителя (`yadisk_pull.py`) в папку `TUEP/` рядом с репозиторием. По скачанной части 11.11 уже проверены: разбор папок `00_epilepsy`, формат разметки приступов `.csv_bi` (метки `bckg` / `seiz`).
>
> **Статус (14.11.2026):** шаги 1–5 выполнены на настоящих данных, результат — `../results/tuep_run_2026-11-14.md`; шаг 6 (дообучение) написан и проверен только на синтетике. Ниже — текст от 02.11, до получения данных. Всё, что про устройство TUEP, взято из описания корпуса (`Поиск_датасетов/Гипотеза_1_ЭЭГ.md`, раздел 3.1) и общих соглашений TUH; места, где формат не проверен, помечены **«проверить»**. Проверен только лёгкий тест на синтетических EDF (`test_synthetic.py`, секунды).

## 0. Подготовка машины (один раз)

```bash
python3 -m venv ~/venvs/tuep && source ~/venvs/tuep/bin/activate        # venv, без conda
pip install numpy scipy pandas scikit-learn tabulate openpyxl           # опись, подготовка, оценка (mne не нужен: EDF читаем сами)
pip install torch transformers huggingface_hub einops timm              # только для шага 5 (под CUDA-сборку машины); einops — коду REVE, timm — коду EEG-DINO с HF
git clone <репозиторий epilepsy-mvp> && cd epilepsy-mvp/h1_eeg/tuep
```

Место на диске: **36 ГБ** EDF + **~17 ГБ** подготовленных записей (627 ч × 200 Гц × 19 каналов × 2 байта; если брать один монтаж — меньше) + ~5 ГБ кэш HF и веса моделей → **с запасом 70 ГБ**. Рабочую папку держать вне репозитория, например `/data/tuep/` (EDF) и `/data/tuep_work/` (всё остальное).

## 1. Скачать (rsync по ключу SSH, источник — страница NEDC)

Источник: https://isip.piconepress.com/projects/nedc/html/tuh_eeg/ . Сервер `www.isip.piconepress.com`, пользователь `nedc-tuh-eeg`, доступ — по ключу SSH, который связан с одобренной заявкой (реквизиты NEDC присылает отдельным письмом; в репозиторий и в чаты их не писать). Путь TUEP: `data/tuh_eeg/tuh_eeg_epilepsy/v3.1.0`. Страница требует всегда ставить `-L` (ссылки на TUEG).

Сначала проверка связи на маленькой папке (команда с страницы NEDC):

```bash
rsync -auvxL -e "ssh -i ~/.ssh/<ВАШ_КЛЮЧ>" nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/TEST /data/tuep/
```

Если не работает: заменить `-auvxL` на `-auxvvvL`, сохранить лог и отдать ИТ (так написано на странице); `ssh -i ~/.ssh/<ВАШ_КЛЮЧ> nedc-tuh-eeg@www.isip.piconepress.com` покажет, не закрыт ли порт 22 (физическая отправка диска 8 ТБ почтой — запасной вариант NEDC, этот случай не про нас).

Скачивание TUEP (докачивается при обрыве: просто запустить ту же команду; в `tmux`/`screen`):

```bash
mkdir -p /data/tuep && cd /data/tuep
rsync -auvxL --partial --progress -e "ssh -i ~/.ssh/<ВАШ_КЛЮЧ>" nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/tuh_eeg_epilepsy/v3.1.0 /data/tuep/
```

Дальше `--root /data/tuep/v3.1.0`. Время зависит от канала (36 ГБ: при 100 Мбит/с около 1 часа, при 20 Мбит/с около 5 часов; **не измерено**). Сверить: `du -sh /data/tuep/v3.1.0` (~36 ГБ, в описании корпуса строка «362288M» похожа на опечатку), число `*.edf` должно быть 2 808, пациентов 200 (после шага 2 это видно в сводке).

## 2. Разобрать метаданные (задача роадмапа 1.3)

```bash
python tuep_inventory.py --root /data/tuep/v3.1.0 --out /data/tuep_work/inventory --threads 8
```

Читает только заголовки EDF (минуты). Выход: `inventory_files.csv` (пациент / сеанс / файл / группа / длительность / частота / каналы / монтаж `01_tcp_ar|02_tcp_le|03_tcp_ar_a` / полнота 19 каналов / сеанс общий с TUSZ / число отчётов `.txt` и разметки приступов рядом), `inventory_sessions.csv` (по сеансам), `inventory_summary.md` (сводка: пациенты, сеансы, часы, частоты, монтажи, наборы каналов, пересечение с TUSZ), `metadata_columns.txt` (столбцы и первые строки `metadata_v00r.xlsx`).

**Проверить после запуска** (формат до получения данных не известен): (а) группа определилась из пути (`epilepsy` / `no_epilepsy`) — иначе скрипт напишет предупреждение, и группу надо взять из таблицы метаданных (дать `--groups-csv patient,y` на шаге 3); (б) пересечение с TUSZ: скрипт ищет `*tusz*.list` и разбирает строки шаблоном «8 букв … sNNN» — открыть файл и убедиться, что число найденных сеансов совпадает со строками списка; (в) ожидаемые цифры из описания корпуса: 200 пациентов (100 + 100), 698 сеансов (530 + 168), 2 808 файлов (2 244 + 564), ~627 ч, файлы ≤ 30 мин; расхождение — повод написать NEDC.

## 3. Подготовить записи в формате общего набора Г1

```bash
python tuep_prepare.py --root /data/tuep/v3.1.0 --out /data/tuep_work/prep \
    --inventory /data/tuep_work/inventory/inventory_files.csv --threads 6
```

Как у `zakharsann/h1-prep-shared` (код — `src/shared_data.py`, блокнот `h1_2026-10-26_zs_prep-shared.ipynb`):

| Шаг | Что делается |
|---|---|
| каналы | 19 каналов 10–20 в порядке `EL` (Fp1 Fp2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 Fz Cz Pz). Имена TUH `EEG FP1-REF` (монтажи `01_tcp_ar`, `03_tcp_ar_a`) и `EEG FP1-LE` (`02_tcp_le`) приводятся к `FP1`; T7/T8/P7/P8 → T3/T4/T5/T6; A1, A2, EKG, PHOTIC и др. отбрасываются. Нет любого из 19 — файл в `h1_shared_failed.csv` |
| частота | к 200 Гц полифазным `resample_poly` (в TUEG бывают 250, 256, 400, 500, 512, 1000, 1024 Гц и др.) |
| референт | **средний по 19 каналам** для всех монтажей (как у турецкого набора). Модели обучались на среднем референте |
| фильтр | Баттерворт 4-го порядка 0,1–75 Гц (`sosfiltfilt`) + режекция **60 Гц** (`iirnotch`, Q = 30, `filtfilt`; в США 60 Гц, а не 50) по записи целиком. При обучении был вырез 50 Гц; `--also-notch50` добавляет и его — решить, нужен ли (по умолчанию нет) |
| единицы | мкВ (в TUH EDF уже в мкВ; масштаб берётся из заголовка, множитель `K = 1`, для NMT был 100). Обрезка ±60 000, float16 |
| выход | `h1_shared_meta.csv`, `h1_shared_info.json`, `h1_shared_failed.csv`, `fs200/tuep_NNN.npy` (по ~1 ГБ). Столбцы как у общего набора плюс `patient`, `session`, `montage`, `in_tusz`; `id` = имя файла без `.edf` (уникально) |

Читать теми же функциями: `from shared_data import load_meta, read_record, read_all` (`load_meta(папка)`; на локальном диске `read_record` через mmap быстро; `read_all` для 17 ГБ не использовать — не поместится в память).

Время: на Kaggle 2 417 записей NMT + Турция (~490 ч) подготовились за ~3 мин в 4 потока; для TUEP (627 ч, лишний пересчёт частоты) **ожидаю десятки минут при 6 потоках, не измерено** — сначала `--limit 20`. Предупреждение: файлы с монтажами `02_tcp_le` и `03_tcp_ar_a` — другая схема референта; можно подготовить отдельно (`--montages 01_tcp_ar`) и сравнить.

## 4. Протокол оценки (задача роадмапа 1.4)

```bash
python tuep_protocol.py folds   --meta /data/tuep_work/prep/h1_shared_meta.csv --out /data/tuep_work/protocol/tuep_folds.csv
python tuep_protocol.py leakage --out /data/tuep_work/protocol/model_leakage.csv
python tuep_protocol.py subgroup --meta /data/tuep_work/prep/h1_shared_meta.csv --root /data/tuep/v3.1.0 --out /data/tuep_work/protocol/subgroup_ied.csv
```

**Правила.**
1. **Разбиение строго по пациентам**: 5 фолдов, `StratifiedKFold` по группе на уровне пациента (seed 0), все сеансы и файлы пациента в одном фолде; скрипт проверяет, что пациент не лежит в двух группах и не разъезжается по фолдам. Для готовых моделей без дообучения фолды нужны для разброса и для будущего дообучения на TUEP; основная оценка — на всех 200.
2. **Один пациент — один сеанс** ([PROTOCOL.md](PROTOCOL.md), п. 2): оценка файла = среднее верхних 20 % вероятностей 60 равномерных окон 10 с → оценка сеанса = среднее по файлам (вес — длительность) → у пациента берётся **один случайный сеанс без приступов** (`--sessions random`, seed 0); `first` / `last` / `all` — проверка устойчивости. Плюс проверка утечки через год записи и монтаж (`tuep_confounds.csv`) и ROC AUC внутри монтажа. Бутстреп ДИ — по пациентам.
3. **Метрики** (CLAUDE.md): ROC AUC, accuracy, сбалансированная accuracy, precision, recall, специфичность, всё с 95 % ДИ. Порог — **не по TUEP**: Юден и чувствительность 80 % по отложенным 15 % NMT train (пересчитываются из `results/eegdino_v2_per_person.csv`: REVE v4 0,832, EEG-DINO v2 0,786, ансамбль 0,755 по Юдену), перенесены без изменений; плюс калибровка порога на k = 10 / 20 случайных пациентах TUEP, метрики на остальных (500 повторов, одни выборки для всех моделей). Предсказания по людям сохраняются (`tuep_patient_scores.csv`).
4. **Утечки** (`model_leakage.csv`): REVE, EEG-DINO, LaBraM (**видела сам TUEP**), CBraMod, LUNA предобучены на TUH → результат на TUEP для них не независим; **основной честный контроль — BIOT и EEGPT**. Это со слов задачи, **по статьям не перепроверено**: у BIOT есть варианты предобучения, в том числе на TUAB/TUEV (тоже TUH) — выяснить, какой вариант у нас, и нет ли пересечения пациентов с TUEP. Все наши дообученные на NMT модели видели TUH только через предобучение.
5. **Подгруппа «без разрядов».** Что в TUEP есть на самом деле: по описанию корпуса **эпилептиформные разряды не размечены** (размечены приступы, 16 пациентов), метки группы выставлены по заключениям врачей — поэтому разряды надо брать другим путём: (а) если в папках сеансов лежат тексты заключений (`.txt`; в `inventory_summary.md` столбец «сеансов с отчётом»; для v3.1.0 **не проверено**), `subgroup` размечает пациентов правилом по тексту («с разрядами» / «без разрядов» / «неясно») и пишет `*_reports.csv` — **проверить вручную 20–30 отчётов**, правило грубое; (б) если отчётов нет — детектор разрядов (в проекте есть опыт IED-детектора на турецком наборе, `h1_2026-10-20_zs_ied-senualab-turkish`): `subgroup --ied-csv файл(patient,n_ied)`. Для оценки: `eval --subgroup subgroup_ied.csv` даёт ROC AUC «эпилепсия без разрядов против всех контролей» и «с разрядами против контролей» (на EpiEEG-60 у REVE v4: 0,807 с разрядами, 0,603 без — 7 человек).

## 5. Прогнать готовые модели без дообучения (одна карта)

Веса — на машину (нужен токен Kaggle на ней; либо скачать на ноутбуке `kaggle kernels output …` — это лёгкая операция — и перекинуть scp):

```bash
kaggle kernels output zakharsann/h1-reve-v4    -p /data/weights/h1-reve-v4     # reve_v4_seed{0,1,2}.pt, reve_v4_model.py
kaggle kernels output zakharsann/h1-eegdino-v2 -p /data/weights/h1-eegdino-v2  # eegdino_v2_seed{0,1,2}.pt, eegdino_v2_model.py
```

Основы моделей с Hugging Face (`brain-bzh/reve-base`, `brain-bzh/reve-positions`, код авторов `eegdino/EEG-DINO`, снимок `5d8ae132…`) скачиваются при первом запуске; если с машины HF не открывается — положить в кэш заранее и поставить `HF_HUB_OFFLINE=1`, для EEG-DINO дать `--dino-src <папка с models/ авторов>`.

```bash
CUDA_VISIBLE_DEVICES=0 python tuep_infer.py --prep /data/tuep_work/prep --out /data/tuep_work/out \
    --reve-dir /data/weights/h1-reve-v4 --dino-dir /data/weights/h1-eegdino-v2 --limit 20      # пробный прогон
# затем без --limit; можно прерывать и запускать заново — уже посчитанные файлы пропускаются
python tuep_protocol.py eval --scores /data/tuep_work/out/tuep_record_scores.csv --meta /data/tuep_work/prep/h1_shared_meta.csv --folds /data/tuep_work/protocol/tuep_folds.csv \
    --results ../results --out /data/tuep_work/out [--subgroup /data/tuep_work/protocol/subgroup_ied.csv]
```

Как в блокнотах: ФВЧ 0,3 Гц по записи → окна 10 с подряд → 60 окон равномерно → верхние 20 % → среднее 3 запусков; ансамбль REVE + EEG-DINO = среднее двух вероятностей, fp16. Сохраняются оценки всех 60 окон каждого запуска (`tuep_window_scores.npz`), по файлам (`tuep_record_scores.csv`), по пациентам (`tuep_patient_scores.csv`), метрики (`tuep_metrics.csv`, `tuep_calibration_k*.csv`). Время: 2 808 файлов × 60 окон × 6 моделей; скорость на A4000 **не измерена** — смотреть по `--limit 20` и оценивать на 2 808 файлов (у REVE v4 замеры скорости на T4 лежат в `results/reve_v4_speed.csv`).

## 6. Дообучение на TUEP по фолдам (задача роадмапа 1.6) и эталон

Рабочая машина: одна RTX A5000 24 ГБ, venv `~/venvs/tuep`, данные `~/data/tuep_work/prep`, веса `~/data/weights/`. Всё с кодами пациентов остаётся в `~/data/tuep_work/ft`; в git — только `tuep_ft_summary.md` и `tuep_ft_metrics.csv`. Обоснование схемы — `Ревью/Ревью_TUEP_2026-11-14.md`.

```bash
source ~/venvs/tuep/bin/activate && cd epilepsy-mvp/h1_eeg/tuep
python -I test_finetune_synthetic.py                          # ~1 мин, процессор: схема целиком на синтетике, без весов
# 1) эталон и контроль утечки — процессор, минуты (сначала --limit 50: печатает оценку времени)
python tuep_baseline.py --prep ~/data/tuep_work/prep --out ~/data/tuep_work/ft --limit 50
python tuep_baseline.py --prep ~/data/tuep_work/prep --out ~/data/tuep_work/ft
# 2) пробный режим дообучения: 1 фолд, 1 запуск, 320 окон; печатает скорость и оценку времени полного прогона
CUDA_VISIBLE_DEVICES=0 python tuep_finetune.py train --model reve --prep ~/data/tuep_work/prep --out ~/data/tuep_work/ft --reve-dir ~/data/weights/h1-reve-v4 --smoke
# 3) полный прогон (в tmux; после обрыва — та же команда: готовые запуски пропускаются, незаконченный продолжается с последней эпохи)
CUDA_VISIBLE_DEVICES=0 python tuep_finetune.py train --model reve --prep ~/data/tuep_work/prep --out ~/data/tuep_work/ft --reve-dir ~/data/weights/h1-reve-v4
CUDA_VISIBLE_DEVICES=0 python tuep_finetune.py train --model dino --prep ~/data/tuep_work/prep --out ~/data/tuep_work/ft --dino-dir ~/data/weights/h1-eegdino-v2
# 4) сводка (можно запускать в любой момент — считает по готовым запускам)
python tuep_finetune.py report --prep ~/data/tuep_work/prep --out ~/data/tuep_work/ft --zero-shot ~/data/tuep_work/out/tuep_patient_scores.csv
```

**Схема.**
- Внешние 5 фолдов по пациентам — те же, что `tuep_protocol.py folds` (seed 0). Для фолда k: тест — фолд k, отложенная часть — фолд k+1 (выбор эпохи и порога), обучение — три остальных (≈120 / 40 / 40 пациентов). 3 запуска (seed 0/1/2, старт с весов после NMT того же номера), оценка файла — среднее по запускам.
- Один пациент — одна оценка: тот же случайный сеанс без приступов, что в прогоне 14.11 (seed 0); 60 окон по 10 с, верхние 20 %, файлы сеанса — с весом длительности.
- Обучение: все сеансы без приступов. Окно: класс 50/50 → слой «монтаж|частота» с одинаковым для классов распределением → пациент → сеанс → файл → случайное начало. Слои, где в одном классе меньше 3 пациентов (`03_tcp_ar_a`, 512 Гц), в обучение не идут (`--balance class` отключает слои). Аугментации — как в блокнотах NMT.
- Сигнал: ФВЧ 0,3 Гц + ФНЧ 45 Гц (`--lowpass 0` — без него), фильтр по куску «окно ± 5 с». Окна для оценки кэшируются в `ft/evalcache_*` (до ~12 ГБ; `--no-cache` — без кэша).
- До 8 эпох по 4000 окон, остановка через 3 эпохи без роста ROC AUC отложенной части. Скорости обучения — из блокнотов REVE v4 / EEG-DINO v2 (`--lr-scale` меняет). `--freeze-backbone` — учить только голову; `--init base` — старт с исходной основы; `--save-weights` — сохранить лучшие веса (не в git).
- Отчёт: ROC AUC (95 % ДИ, бутстреп по пациентам), среднее по фолдам, ROC AUC по парам внутри слоя и только в `01_tcp_ar`, пороговые метрики (порог — отложенная часть своего фолда), разность с прогоном без дообучения, устойчивость к выбору сеанса. Рядом — эталон `bp` / `bpamp` и контроль `tech` (без ЭЭГ).
- `--exclude-sessions ~/data/tuep_work/protocol/exclude_report_seizures.csv` (файл пишет `tuep_audit.py`) — проверка устойчивости: без сеансов, где приступ есть в заключении врача, но нет в разметке. Для неё нужна отдельная папка `--out`.

Время не измерено: пробный режим печатает оценку. Ожидание — 1–2 ч на модель.

## Файлы папки

| Файл | Что |
|---|---|
| `tuep_common.py` | имена каналов TUH, чтение EDF (без mne), 200 Гц, фильтры, разбор путей |
| `tuep_inventory.py` | шаг 2 |
| `tuep_prepare.py` | шаг 3; `--ref ears --filt hp` — подготовка для раздела 9 |
| `tuep_protocol.py` | шаг 4: `folds`, `leakage`, `subgroup`, `eval` |
| `PROTOCOL.md` | правила прогона, записанные до получения результатов |
| `yadisk_pull.py` | скачивание TUEP с Яндекс Диска (докачка, md5) |
| `tuep_infer.py` | шаг 5 |
| `tuep_finetune.py` | шаги 6 и 8: `train` (дообучение по фолдам: `reve`, `dino`, `de`; продолжение после обрыва, `--smoke`), `report` (сводка, `--ensemble` можно повторять) |
| `tuep_baseline.py` | шаг 6: эталон «мощности ритмов + логистическая регрессия» и контроль утечки без ЭЭГ — по тем же фолдам |
| `tuep_audit.py` | проверка подготовленных данных и оценок на рабочей машине (ревью 14.11): `python tuep_audit.py --work ~/data/tuep_work --root ~/data/TUEP` |
| `test_finetune_synthetic.py` | лёгкий тест шага 6 на синтетике: `python -I test_finetune_synthetic.py` |
| `tuep_deepepilepsy.py` | DeepEpilepsy авторов без дообучения (раздел 7) и проверка референта и фильтра (раздел 8) |
| `test_de_synthetic.py` | лёгкий тест ветки `--model de` и проверки референта на синтетике: `python -I test_de_synthetic.py` |
| `test_synthetic.py` | лёгкий тест на синтетических EDF: `python -I test_synthetic.py` (опись → подготовка → чтение `shared_data` → фолды → оценка) |

## Вопросы к NEDC (контакт — на странице NEDC; письмо отправляет Захар сам)

1. Коммерческое использование: допустимо ли развивать на TUEP/TUH продукт (стартап), в каких формах (модели, дообученные на TUEP; выдача результатов по данным другой клиники)? В форме v6.0 слов «коммерческое/некоммерческое» нет, есть «исследования и разработка технологий, но не иные цели».
2. Облако: можно ли временно хранить и считать на Kaggle / в другом облаке (данные туда загружаются как частный набор)? Пока нет ответа — только рабочая машина.
3. Веса моделей, дообученных на TUEP, и производные признаки: можно ли хранить после окончания проекта и передавать (внутри университета, заказчику)? Что значит «удалить данные по окончании работы» для обученных весов?
4. Доступ команды: каждому участнику (3 человека) нужна отдельная заявка/ключ, или можно общим доступом на одной машине?
5. Пересечения: есть ли списки пациентов TUEP, общих с TUAB, TUEV, TUSZ и TUEG v2.0.2 (кроме `sessions_common_with_tusz.list`)? Нужны для честного контроля моделей, предобученных на TUH (LaBraM «видела» TUEP).
6. Состав: входят ли в TUEP v3.1.0 тексты заключений врачей (`.txt`) и как формально поставлена группа «без эпилепсии» (диагнозы этих пациентов)? Формат и столбцы `metadata_v00r.xlsx` и списка `sessions_common_with_tusz.list`.
7. Как цитировать v3.1.0 и есть ли обязательные формулировки благодарностей.
8. Размер: подтвердить ~36 ГБ и 2 808 файлов (в описании строка «362288M»).

## 7. DeepEpilepsy авторов без дообучения (15.11)

Модель обучена ровно на нашей задаче (эпилепсия против других пациентов, рутинная ЭЭГ, CHUM). Код модели клонируется из gitlab авторов, веса — с HF при первом запуске. Те же 199 сеансов, что в основном прогоне:

```bash
CUDA_VISIBLE_DEVICES=0 python tuep_deepepilepsy.py --prep ~/data/tuep_work/prep --out ~/data/tuep_work/out_de --sessions-from ~/data/tuep_work/out/tuep_patient_scores.csv --limit 20
CUDA_VISIBLE_DEVICES=0 python tuep_deepepilepsy.py --prep ~/data/tuep_work/prep --out ~/data/tuep_work/out_de --sessions-from ~/data/tuep_work/out/tuep_patient_scores.csv
python tuep_protocol.py eval --scores ~/data/tuep_work/out_de/tuep_record_scores.csv --meta ~/data/tuep_work/prep/h1_shared_meta.csv --results ../results --out ~/data/tuep_work/out_de
```

Порога с NMT для неё нет: в выводе — ROC AUC, калибровка по 10 / 20 пациентам и порог без разметки.

## 8. Дообучение DeepEpilepsy на TUEP по 5 фолдам (15.11)

Та же схема, что в разделе 6 (те же фолды, тот же сеанс на пациента, те же веса сеансов и слои «монтаж|частота», те же правила порога, ФНЧ 45 Гц), в ту же папку `~/data/tuep_work/ft` — чтобы `report` показал всё в одной таблице. Отличия ветки `--model de`:

| Что | Как |
|---|---|
| окно | 30 с (19 × 6000), при оценке — 20 окон на файл равномерно (те же 10 минут записи, что 60 окон по 10 с) |
| вход | как у авторов: каналы в порядке `settings.ELECTRODES`, `standard_scale` — каждый канал окна к нулевому среднему и единичному СКО |
| аугментации | RandAugment авторов (n = 1, m = 7, до нормировки): режекция случайной полосы, ФНЧ, зануление прямоугольника «каналы × время», зануление каналов, шум. Перенесён в наш код, чтобы работало продолжение после обрыва |
| оценка файла | медиана вероятностей окон (как в статье), а не верхние 20 % |
| старт | веса авторов `DeepEpilepsy_large_50_30s_augment.pt` (кэш HF или `--de-weights`), код модели — `--de-repo` (по умолчанию `~/data/weights/dl_epilepsy_reeg`) |
| скорости обучения | основа 2·10⁻⁵ (как при дообучении на NMT 20.10), голова 5·10⁻⁴ (как у REVE), AdamW 0,05, обрезка градиента 1,0 |
| имена запусков | `de` — учится всё; `defz` — только голова (`--freeze-backbone`) |
| короткие файлы | файлы короче 30 с оценки не получают; сеансы без единого файла от 30 с в обучение не идут. Скрипт печатает, сколько таких |

**Заданная заранее пара: `de` + `bpampw`** (DeepEpilepsy, дообученная полностью, + эталон с wPLI; среднее вероятностей по файлам — то же правило, что у пары REVE). Вариант `defz` смотрим отдельной строкой, в пару не идёт. Других сочетаний по тестовым фолдам не подбираем. Засчитываем, если пара выше 0,661 и внутри слоя выше 0,633 (пара REVE + эталон с wPLI).

```bash
cd epilepsy-mvp && git pull && cd h1_eeg/tuep && source ~/venvs/tuep/bin/activate
python -I test_de_synthetic.py                 # ~1–2 мин, процессор, без кода и весов авторов; в конце «ТЕСТ ВЕТКИ DEEPEPILEPSY ПРОЙДЕН»
# пробный режим: 1 фолд, 1 запуск, 320 окон; печатает число коротких файлов, скорость и оценку времени
CUDA_VISIBLE_DEVICES=0 python tuep_finetune.py train --model de --prep ~/data/tuep_work/prep --out ~/data/tuep_work/ft --smoke
# полный прогон: сначала de, затем defz (после обрыва — та же команда)
CUDA_VISIBLE_DEVICES=0 nohup sh -c "python tuep_finetune.py train --model de --prep ~/data/tuep_work/prep --out ~/data/tuep_work/ft && python tuep_finetune.py train --model de --freeze-backbone --prep ~/data/tuep_work/prep --out ~/data/tuep_work/ft" > ~/data/tuep_work/ft/de.log 2>&1 &
tail -f ~/data/tuep_work/ft/de.log
# сводка: обе заданные пары в одной таблице
python tuep_finetune.py report --prep ~/data/tuep_work/prep --out ~/data/tuep_work/ft --zero-shot ~/data/tuep_work/out/tuep_patient_scores.csv --ensemble reve bpampw --ensemble de bpampw
```

Место на диске: кэш окон `ft/evalcache_lp45_w60_L6000` — до ~12 ГБ (`--no-cache` — без него, оценка медленнее). Если HF с машины не открывается: `HF_HUB_OFFLINE=1` (веса уже в кэше после раздела 7) или `--de-weights <файл>`. Время не измерено — пробный режим печатает оценку.

**Проверка референта без дообучения (по желанию, минуты).** У авторов запись сохранена относительно среднего A1 и A2 и идёт в сеть без программных фильтров; у нас — средний по 19 каналам и 0,1–75 Гц + режекция 60 Гц. `--ref ears` читает EDF тех же 199 сеансов заново: `01_tcp_ar` — вычитает (A1 + A2) / 2, `02_tcp_le` — как есть, `03_tcp_ar_a` — пропуск (ушных электродов нет). Опись берётся из `~/data/tuep_work/inventory/inventory_files.csv`, `--root` — та же папка, что при описи.

```bash
CUDA_VISIBLE_DEVICES=0 python tuep_deepepilepsy.py --prep ~/data/tuep_work/prep --out ~/data/tuep_work/out_de_ears --ref ears --root ~/data/TUEP --sessions-from ~/data/tuep_work/out/tuep_patient_scores.csv --compare ~/data/tuep_work/out_de/tuep_record_scores.csv
CUDA_VISIBLE_DEVICES=0 python tuep_deepepilepsy.py --prep ~/data/tuep_work/prep --out ~/data/tuep_work/out_de_ears_raw --ref ears --filt none --root ~/data/TUEP --sessions-from ~/data/tuep_work/out/tuep_patient_scores.csv --compare ~/data/tuep_work/out_de/tuep_record_scores.csv
```

Последняя строка вывода — ROC AUC обоих вариантов на общих пациентах и разность с 95 % ДИ. Знак класса у готовой модели не установлен, поэтому смотреть на удаление от 0,5 в любую сторону. Если ДИ разности не содержит 0 — готовить записи относительно ушей (только `01_tcp_ar` и `02_tcp_le`) и повторять дообучение на них; иначе референт ни при чём.

## 9. DeepEpilepsy в условиях авторов: референт к ушам, без ФНЧ и режекции (15.11, последняя попытка)

Раздел 8 дал `de` 0,510 и пару `de` + `bpampw` 0,594 — не лучше пары REVE (0,661). Здесь вход приближен к авторскому. Запуски называются `dea` / `deafz` и идут в отдельную папку `~/data/tuep_work/ft_de_auth`.

| Что | Раздел 8 | Здесь |
|---|---|---|
| референт | средний по 19 каналам | (A1 + A2) / 2: `01_tcp_ar` — по каналам A1, A2 файла; `02_tcp_le` — как есть |
| фильтры подготовки | 0,1–75 Гц + режекция 60 Гц | только ФВЧ 0,1 Гц (без него постоянная составляющая EDF съедает точность float16) |
| фильтры при чтении | ФВЧ 0,3 + ФНЧ 45 Гц | только ФВЧ 0,3 Гц (`--lowpass 0`) |
| `03_tcp_ar_a` | в тесте есть | **нет**: ушных электродов нет, файлы не готовятся, пациенты с таким выбранным сеансом остаются без оценки |

Остальное как в разделе 8. Сеансы, слои, фолды и выбранный сеанс считаются по основной подготовке (`--proto-prep`), поэтому совпадают с прежними; из `prep_ears` берётся только сигнал. Эталон `bpampw` — прежний (на среднем референте), подтягивается в отчёт из `ft/runs`.

**Задано заранее.** Варианты `dea` (учится всё) и `deafz` (только голова); пара — **`dea` + `bpampw`**. Сравнение — в сводке `_same_dea`, где все строки посчитаны на одних и тех же пациентах (у кого есть оценка `dea`). Засчитываем, если пара `dea+bpampw` выше пары `reve+bpampw` **и по ROC AUC, и внутри слоя** в этой сводке. Главные столбцы — «внутри слоя» и `01_tcp_ar`: без ФНЧ и режекции верх спектра и сеть 60 Гц сильно зависят от прибора, и рост только общего ROC AUC — это утечка, а не результат.

```bash
cd epilepsy-mvp && git pull && cd h1_eeg/tuep && source ~/venvs/tuep/bin/activate
python -I test_de_synthetic.py                 # ~3 мин, процессор
# подготовка: тот же --root и та же опись, что у основной; сначала проба на 20 файлах в отдельную папку
python tuep_prepare.py --root ~/data/TUEP --out ~/data/tuep_work/prep_ears_probe --inventory ~/data/tuep_work/inventory/inventory_files.csv --ref ears --filt hp --limit 20
python tuep_prepare.py --root ~/data/TUEP --out ~/data/tuep_work/prep_ears --inventory ~/data/tuep_work/inventory/inventory_files.csv --ref ears --filt hp --threads 6
# пробный режим
CUDA_VISIBLE_DEVICES=0 python tuep_finetune.py train --model de --name dea --prep ~/data/tuep_work/prep_ears --proto-prep ~/data/tuep_work/prep --out ~/data/tuep_work/ft_de_auth --lowpass 0 --smoke
# полный прогон: dea, затем deafz
mkdir -p ~/data/tuep_work/ft_de_auth && CUDA_VISIBLE_DEVICES=0 nohup sh -c "python tuep_finetune.py train --model de --name dea --prep ~/data/tuep_work/prep_ears --proto-prep ~/data/tuep_work/prep --out ~/data/tuep_work/ft_de_auth --lowpass 0 && python tuep_finetune.py train --model de --name dea --freeze-backbone --prep ~/data/tuep_work/prep_ears --proto-prep ~/data/tuep_work/prep --out ~/data/tuep_work/ft_de_auth --lowpass 0" > ~/data/tuep_work/ft_de_auth/dea.log 2>&1 &
# сводки: полная (видно n) и на одних и тех же пациентах (по ней решаем); --prep здесь — основная подготовка
python tuep_finetune.py report --prep ~/data/tuep_work/prep --out ~/data/tuep_work/ft_de_auth --extra-runs ~/data/tuep_work/ft/runs --zero-shot ~/data/tuep_work/out/tuep_patient_scores.csv --ensemble reve bpampw --ensemble dea bpampw --ensemble de bpampw
python tuep_finetune.py report --prep ~/data/tuep_work/prep --out ~/data/tuep_work/ft_de_auth --extra-runs ~/data/tuep_work/ft/runs --zero-shot ~/data/tuep_work/out/tuep_patient_scores.csv --ensemble reve bpampw --ensemble dea bpampw --ensemble de bpampw --same-patients dea
```

Подготовка в конце печатает таблицу «класс × монтаж» и причины, по которым файлы не подготовлены (ожидаемая — «нет A1/A2» у `03_tcp_ar_a`). Наличие A1/A2 в файлах `01_tcp_ar` заранее не проверялось: если их нет у заметной части файлов, пациентов с оценкой станет мало — тогда остановиться до прогона. Место на диске: `prep_ears` — до ~17 ГБ, кэш окон в `ft_de_auth` — до ~12 ГБ (`--no-cache` — без кэша). Пороговые метрики (accuracy, precision, recall, специфичность) — в разделе «Пороговые метрики» каждой сводки и в `tuep_ft_threshold_metrics*.csv`.
