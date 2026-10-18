# epilepsy-mvp

Командный репозиторий проекта по эпилепсии (Губкинский университет). Цель — работающий MVP (модель + интерфейс) по каждой из двух гипотез, отчёт и презентация.

- **Г1 — выявление эпилепсии по рутинной ЭЭГ.** По одной записи (20–30 мин) модель оценивает вероятность эпилепсии, в том числе когда явных разрядов нет.
- **Г2 — ассистент рентгенолога: поиск ФКД на МРТ.** По T1 (+ FLAIR) модель находит и подсвечивает очаги фокальной кортикальной дисплазии — «второе мнение».

Сроки: предзащита 25.11.2026, защита 11.12.2026.

> ## ⚠️ Данные и веса моделей в репозиторий не попадают никогда
> Данные TUH нельзя передавать третьим лицам; остальные наборы и веса тоже храним вне git. Это закреплено в `.gitignore`, в проверке перед коммитом (`.githooks/pre-commit`) и в GitHub Actions. Подробнее: [docs/data_access.md](docs/data_access.md).

## Структура

```
epilepsy-mvp/
├── h1_eeg/                 Г1 (ЭЭГ)
│   ├── notebooks/          эксперименты (.ipynb с выводами)
│   ├── src/                переиспользуемый код Г1
│   ├── results/            небольшие итоговые png/csv
│   ├── SUMMARY.md          краткие итоги
│   └── requirements.txt
├── h2_mri/                 Г2 (МРТ) — то же самое
├── common/                 общий код: пути к данным, деление по пациентам, метрики
├── app/                    будущий интерфейс (пока пусто)
├── templates/              шаблон блокнота-эксперимента
├── docs/                   данные и доступ, имена и журнал, git для новичков
├── scripts/check_repo.py   проверка: нет данных/весов, размер блокнотов
├── experiments_log.md      журнал экспериментов
└── .github/                шаблон pull request и автопроверка
```

## Первый запуск (на любой машине)

```bash
git clone https://github.com/Jessk1y/epilepsy-mvp.git
cd epilepsy-mvp
git config user.name "Имя Фамилия"
git config user.email "ваша@почта"
git config core.hooksPath .githooks      # включить проверку перед коммитом
```

> **Обязательно:** каждый участник на каждой машине после клонирования выполняет `git config core.hooksPath .githooks`. Без этого проверка перед коммитом не работает, и данные или веса могут попасть в git. Проверить: `git config core.hooksPath` должна вывести `.githooks`.

### Windows
Python запускаем через `py`; для кириллицы в выводе — `PYTHONIOENCODING=utf-8`.
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r h1_eeg\requirements.txt      # или h2_mri\requirements.txt
$env:EPI_DATA_DIR = "D:\epi_data"                  # куда сложены данные (вне репозитория)
py -m jupyter lab
```
Ноутбук с RTX 3050 (4 ГБ) и малым местом на диске подходит для небольших проверок; тяжёлое — на Kaggle или Ubuntu. PyTorch ставится отдельно по инструкции pytorch.org.

### Ubuntu (рабочая машина)
Основной вариант для тяжёлых расчётов. С этой машины physionet.org не открывается — данные оттуда берите через другую машину.
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r h1_eeg/requirements.txt        # или h2_mri/requirements.txt
export EPI_DATA_DIR=/mnt/data/epi
jupyter lab
```
MELD Graph: Docker/Linux, FreeSurfer, >20 ГБ ОЗУ, регистрация и некоммерческая лицензия.

### Kaggle (GPU T4, ~30 ч/нед)
1. Данные подключаются через «Add Input» (Kaggle Datasets), они лежат в `/kaggle/input`.
2. Репозиторий нужен ради папки `common/`. Варианты:
   - клонировать приватный репозиторий с токеном GitHub только на чтение, хранящимся в **Kaggle Secrets** (в блокнот токен не вписывать!):
     ```python
     from kaggle_secrets import UserSecretsClient
     t = UserSecretsClient().get_secret("GITHUB_TOKEN")
     !git clone https://x-access-token:{t}@github.com/Jessk1y/epilepsy-mvp.git /kaggle/working/epilepsy-mvp
     ```
     (вывод этой ячейки со ссылкой на токен перед коммитом очистить);
   - или загрузить папку `common/` как Kaggle Dataset.
3. Закончив, скачайте `.ipynb` **с сохранёнными выводами** и положите в `h?_*/notebooks/` под правильным именем, затем обычный путь: ветка → PR.
4. Веса скачивайте в `/kaggle/working/weights` — в git они не идут.

## Как работать

1. Новый эксперимент: скопируйте [templates/experiment_template.ipynb](templates/experiment_template.ipynb) в `h1_eeg/notebooks/` или `h2_mri/notebooks/`, назовите по правилу ([docs/naming_and_journal.md](docs/naming_and_journal.md)), заполните шапку.
2. Блокнот — это отчёт. Структура: **Данные → Предобработка → Модель → Метрики → Выводы → Что дальше.** Выводы ячеек всегда сохраняем (очищать их не нужно); перед коммитом — Restart & Run All.
3. **Деление на обучение/тест — только по пациентам** (`common/splits.py`). Метрики — на уровне пациента с доверительным интервалом (`common/metrics.py`). Не смешивать классы из разных источников.
4. Код, нужный в нескольких блокнотах, переносим из блокнота в `common/` или `h?_*/src/`.
5. Блокнот не должен раздуваться: картинки `dpi=100`, большие массивы не печатаем; проверка предупредит, если файл больше 10 МБ.
6. Завершили эксперимент → строка в [experiments_log.md](experiments_log.md) и (при необходимости) обновление `SUMMARY.md`.
7. Git: ветка → pull request → просмотр другим участником → слияние. Правила для новичков и как не ловить конфликты в `.ipynb`: [docs/git_for_beginners.md](docs/git_for_beginners.md).
8. Один блокнот — один автор. Чужие не редактируем.
9. Нечестные цифры хуже плохих: не подгонять порог под тест, не писать вывод сильнее, чем позволяют данные. Честный ориентир для Г1 — ROC AUC ~0,76–0,79.

## Проверка вручную

```bash
py scripts/check_repo.py --all       # Windows;  python3 scripts/check_repo.py --all — Ubuntu
```
Ошибка = в git попали данные/веса или большой файл; предупреждение = блокнот больше 10 МБ или без выводов.

## Лицензии и данные
Открытые наборы используем только для проверки моделей, не для коммерции. Некоммерческие лицензии (например, MELD Graph) на этом этапе не помеха. Закрытые данные команде выдадут позже.
