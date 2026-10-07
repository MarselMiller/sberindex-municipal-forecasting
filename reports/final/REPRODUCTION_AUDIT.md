# F5 — аудит воспроизведения из чистого clone

Дата: 2026-10-07. **Итог: PASS WITH LIMITATION.** Проверены отдельный GitHub clone, установка, часть pytest без новых fit, CLI и доступность финальных материалов. Полное воспроизведение исследовательских экспериментов не проверено: приватные данные, исходные outputs и веса не предоставляются через Git.

## 1. Проверяемая версия и изоляция

- Репозиторий: [MarselMiller/sberindex_python_mvp](https://github.com/MarselMiller/sberindex_python_mvp).
- Ветка: `main`.
- Cloned HEAD: `6ada52638299415936742669b24d12b6792f64d7` — `F4: add final presentation`.
- Перед clone локальный HEAD, `origin/main` и фактический GitHub `refs/heads/main` совпали.
- Новый clone: `${PROJECT_PARENT}/sberindex_python_mvp_repro`, соседний каталог к `sberindex_python_mvp`. `${PROJECT_PARENT}` — тот же родительский каталог, который указан в задании F5; переносимая запись сохраняет местоположение без машинного абсолютного пути.
- Clone выполнен из GitHub, без локального источника объектов, заимствованных окружений и ручного копирования данных. Папка до начала отсутствовала; существующие пользовательские файлы не удалялись.
- До clone в исходном репозитории была только локальная untracked `.vscode/`. Она не копировалась, не читалась и не участвовала в проверке.
- Исправления README и DATA_NOTICE сделаны в исходной рабочей папке после выявления проблем. Проверяемый clone сохранил исходный commit и чистое tracked-дерево; эти изменения ещё не входят в remote HEAD.

## 2. Среда и установка

| Параметр | Фактический результат |
| --- | --- |
| ОС | Windows, PowerShell |
| Python | 3.12.10 |
| Окружение | новая `sberindex_python_mvp_repro/.venv` |
| System site-packages | `include-system-site-packages=false` |
| User site-packages | выключены; проверено через `site.ENABLE_USER_SITE` |
| pip | 25.0.1 |
| Установленные runtime/test distributions | 26; [точный freeze](reproduction_audit/environment_freeze.txt) |
| Installation | exit 0, около 3,5 минуты |
| Проверка зависимостей | `pip check`: `No broken requirements found`, exit 0 |
| Optional компоненты | Prophet, LightGBM и Chronos не установлены; их отдельные requirements доступны |

Окружение создано командой README `py -3.12 -m venv .venv`. Установлены только `requirements.txt` и `requirements-ruptures.txt`. Для изоляции pip дополнительно использованы `--isolated`, `--no-cache-dir` и явный публичный PyPI index. Старые `.venv`, `.venv-chronos`, pip cache и глобальные Python-пакеты не использовались.

Во время установки ошибок не было. Pip сообщил о доступном обновлении; обновление не выполнялось. Транзитивные зависимости разрешились из текущего index, поэтому этот freeze не объявляется совпадающим с историческим Linux-снимком `requirements-lock.txt`.

Ограничения среды агента учитывались отдельно: первый GitHub-запрос в ограниченном режиме завершился ошибкой соединения, а Windows launcher в этом режиме не видел зарегистрированный Python. Разрешённые заданием сетевой запрос и создание окружения выполнены через штатный механизм доступа; оба завершились успешно. Эти отказы не классифицируются как ошибки репозитория.

## 3. Выполненные команды

Пути ниже указаны относительно соответствующей рабочей папки. Стандартная установка и запуск Python выполнялись из **нового clone**, а не из исходного репозитория.

```powershell
# Исходный repo: read-only проверка состояния и GitHub main.
git status --short
git log -5 --oneline
git rev-parse HEAD origin/main
git ls-remote https://github.com/MarselMiller/sberindex_python_mvp.git refs/heads/main

# Из исходного repo; переносимый эквивалент фактического полного пути назначения.
git clone --branch main --single-branch https://github.com/MarselMiller/sberindex_python_mvp.git ../sberindex_python_mvp_repro

# Из корня нового clone.
git status --short
git rev-parse HEAD
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip --version
.\.venv\Scripts\python.exe -m pip install --isolated --no-cache-dir --index-url https://pypi.org/simple -r requirements.txt -r requirements-ruptures.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pip freeze
.\.venv\Scripts\python.exe -I -c "import sys,site; print(sys.prefix != sys.base_prefix, site.ENABLE_USER_SITE)"
```

| Команда / проверка | Результат |
| --- | --- |
| GitHub main и новый clone | exit 0; HEAD совпал |
| Создание новой `.venv` | exit 0 |
| Установка и `pip check` | exit 0 |
| Ограниченный pytest без fit | exit 0; 1432 passed, 34 deselected, 1 warning |
| `run.py --help` | exit 0 |
| `scripts/run_national_local_lightgbm.py --help` | exit 0 |
| `scripts/run_chronos.py --help` | exit 0 |
| `scripts/run_early_warning_synthetic.py --help` | exit 0 |
| `scripts/run_chronos.py --config configs/chronos_zero_shot.yaml --preflight-only` | exit 1: отсутствует приватный CSV |
| Те же четыре `--help` из родительского каталога с явным путём к скрипту | все exit 0; выводы совпали с запуском из корня |
| Импорты 13 модулей и чтение 18 YAML | успешно |
| Ссылки README и SHA рисунков | успешно |
| Извлечение текста PDF через `pdftotext` | exit 0; текст извлечён без ошибок |
| `git diff --check` | exit 0 |

Выполнены также read-only проверки `git ls-files`, `git diff --numstat`, tracked status, SHA256, поиска путей и потенциальных секретов. Результаты зафиксированы в [verification.json](reproduction_audit/verification.json). Данные моделей, реальные прогнозы и метрики не пересчитывались.

## 4. Pytest: область проверки без новых fit

Полный `python -m pytest` в этом проекте включает небольшие обучения на синтетике. Чтобы соблюсти запрет F5 на новые model fits, он **не запускался без ограничений**. До выполнения проверен код тестов и их fixtures.

Из 1466 текущих execution nodes исключены 34: реальные CatBoost/LightGBM обучения, логистический optimizer и вызовы `Ruptures.fit`. Использованы 27 точных префиксов `--deselect`; параметризованные варианты покрываются префиксом. [Список исключений](reproduction_audit/nofit_deselect.txt) является частью протокола аудита, а не зависимостью проекта или исходными данными.

Фактический pytest был выполнен из clone с `-B -X utf8 -m pytest -p no:cacheprovider --basetemp=.venv/f5-pytest-temp` и каждым префиксом указанного списка в отдельном `--deselect=...`. Для повторения той же команды из нового clone:

```powershell
# Из корня clone; соседний исходный checkout содержит этот отчёт.
$f5AuditSource = (Resolve-Path '../sberindex_python_mvp').Path
$f5Deselect = Get-Content (Join-Path $f5AuditSource 'reports/final/reproduction_audit/nofit_deselect.txt') |
    ForEach-Object { "--deselect=$_" }
$env:PYTHONNOUSERSITE = '1'
$env:MPLCONFIGDIR = Join-Path (Get-Location).Path '.venv/f5-runtime/matplotlib'
.\.venv\Scripts\python.exe -B -X utf8 -m pytest -p no:cacheprovider --basetemp=.venv/f5-pytest-temp @f5Deselect
```

При исходном запуске список передан явными CLI arguments; файл из исходного checkout не загружался в reproduction environment. Пример выше лишь компактно воспроизводит те же аргументы. `basetemp` создан в новой `.venv`; не следует направлять его на каталог с пользовательскими файлами.

**Фактический результат: 1432 passed, 34 deselected, 1 warning, 276.25 с, exit 0.** Предупреждение `RuntimeWarning: invalid value encountered in subtract` относится к `tests/test_direct_training.py::test_nonfinite_anchor_and_target_are_excluded`, где специально проверяются неконечные значения. Падений тестов нет.

Оставшиеся tests используют synthetic fixtures, временные файлы и заглушки estimators/network/pipeline. Они читают tracked configs/source, не требуют приватного CSV, реальных outputs или скачанных весов. Файлы фикстур и новый Matplotlib cache остаются в новой `.venv`; это не новые исследовательские outputs. Число реально обученных моделей в F5 — ноль.

## 5. CLI, imports и working directory

```powershell
.\.venv\Scripts\python.exe run.py --help
.\.venv\Scripts\python.exe scripts/run_national_local_lightgbm.py --help
.\.venv\Scripts\python.exe scripts/run_chronos.py --help
.\.venv\Scripts\python.exe scripts/run_early_warning_synthetic.py --help
.\.venv\Scripts\python.exe scripts/run_chronos.py --config configs/chronos_zero_shot.yaml --preflight-only
```

Все help-команды работают с основными requirements, без optional learners. Их повтор из родительской папки с явным путём к executable/script дал те же результаты. README корректно требует запуск из корня; скрытой зависимости от исходного рабочего checkout не выявлено. Полные model runs из другой working directory не проверялись.

Проверены 13 импортов: пять модулей проекта через явно добавленный `clone/src` и восемь библиотек из `clone/.venv`. Ни один из них не загружен из старого окружения или глобальных site-packages. Все 18 tracked YAML разобраны. Это проверка документированного workflow со скриптами; самостоятельный `pip install .` не является проверенным способом установки runtime-зависимостей.

Chronos preflight остановился на отсутствующем `data/input/consumption_all_categories.csv`, до загрузки весов и создания outputs. Даже при наличии данных текущий preflight сверяет входы и сохранённые результаты, но не подтверждает готовность Chronos/torch: соответствующие проверки выполняются позднее. Это уточнено в README.

## 6. Данные и финальные материалы

В clone нет приватного расходного CSV, `.venv-chronos`, весов, исходных архивов, сохранённых E01/E02 outputs или лицензионного PDF. В `data/input/` и `outputs/` находятся служебные `.gitkeep`. Ничего из приватного комплекта не копировалось и не публиковалось.

Для чтения доступны README, [финальная сводка](RESULTS_SUMMARY.md), [методологический отчёт](METHODOLOGY_REPORT.md), финальные CSV/JSON, [figure manifest](figure_manifest.csv), 11 PNG и [презентация](presentation/presentation.pdf). Все 37 вхождений внешних относительно README локальных ссылок, 28 разных targets, существуют в clone и относятся к tracked-файлам либо каталогам с tracked-содержимым. Внутренние ссылки ведут к существующим разделам README.

SHA256 всех 11 PNG совпадают с figure manifest. PNG metadata проверены; PDF имеет корректные header/EOF и совпадает по SHA256 с сохранённой F4 verification. Локальный `pdftotext` извлёк 240 строк текста, exit 0, stderr пуст; визуальный рендеринг в F5 не проверялся. Экспорт нового PDF и построение новых figures не выполнялись.

В README теперь указаны:

- отсутствие опубликованного канала получения приватного комплекта и необходимость обращения к его владельцу;
- путь `data/input/consumption_all_categories.csv` и CSV contract;
- исходные `outputs/prophet_comparison_v1/` и `outputs/catboost_direct_v1/` с согласованными SHA для Chronos preflight;
- доступные без данных отчёты, презентация, CLI help и synthetic tests;
- зависимость реальных экспериментов и пересборки сводки от исходных данных и сохранённых артефактов.

Инструкция не заменяет сами данные и разрешение на их использование. Наличие кодовой и итоговой публичной копии не означает проверку прав на исходный комплект и построчные производные данные.

## 7. Paths, secrets и Git hygiene

Поиск охватил все 231 tracked-файл clone: 219 текстовых файлов, 11 PNG и один PDF. Потенциальные значения секретов не печатались. Рабочих API keys, tokens, passwords, private keys или private URLs не обнаружено. Синтетическая credential-URL в `tests/test_macro_cbr.py:173` относится к проверке отказа и не является действующим секретом. Это статическая проверка, без проверки credentials у внешних сервисов.

**Оперативные пути README, src, scripts и configs — PASS.** Однако буквальная проверка отсутствия абсолютных локальных путей во всём tracked repository — **FAIL**: 23 исторические строки в девяти report/JSON-файлах и отдельное упоминание корня Windows-диска в `docs/TASKS.md:69` проверяемого commit. Четыре первичных совпадения в binary bytes оказались ложными; PNG metadata и 21 PDF Flate stream находок не содержат.

| Исторический файл в проверяемом commit | Строки |
| --- | --- |
| `reports/final/results_summary.json` | 9337, 9390 |
| `reports/results/E03_chronos_zero_shot.md` | 33 |
| `reports/results/E04a_online_detection.md` | 5 |
| `reports/results/E05b_trend_calendar.md` | 280, 281, 288 |
| `reports/results/E05c_macro_forecast.md` | 527, 528 |
| `reports/results/E05d_national_local_lightgbm.md` | 758, 759 |
| `reports/results/E06a_offline_detection.md` | 414, 415, 416, 432 |
| `reports/results/E07b_early_warning_full_panel.md` | 180–184, 186 |
| `reports/results/E07c_synthetic_early_warning.md` | 317, 323 |

Эти строки описывают старые execution commands/provenance; они не используются документированными командами нового clone. Они сохранены, чтобы не переписывать исследовательскую provenance. Их нельзя копировать как переносимые инструкции запуска. В новых F5-файлах и изменённых оперативных инструкциях машинных абсолютных путей нет.

`git status --short` был пуст после clone, создания `.venv`, установки, CLI, импортов и pytest. `git diff --check` в clone прошёл. Cache и `.venv` не загрязняют tracked tree; `outputs/` сохранил только `.gitkeep`. Исходная `.vscode/` не затронута.

## 8. Обнаруженные проблемы и исправления

| Проблема | Действие | Повторная проверка / статус |
| --- | --- | --- |
| README приписывал preflight проверку зависимостей | Уточнена реальная область проверки | Сверено с кодом и фактическим CLI; исправлено локально |
| Приватные входы и места их размещения не перечислены | Добавлены CSV contract, пути E01/E02, требования SHA и граница действий без данных | Пути сопоставлены с configs и source clone; исправлено локально |
| DATA_NOTICE не пояснял отсутствие лицензионного PDF и содержал устаревшую фразу о публичной копии | Указано приватное хранение PDF, убрана устаревшая фраза | Сверено с tracked-составом clone; исправлено локально |
| Полный pytest включает настоящие fit | Для F5 применён документированный no-fit выбор; README предупреждает о малых обучениях | 1432 passed, 34 deselected; тесты не изменены |
| Абсолютные пути в исторических reports/provenance | Не переписывались | Остаётся ограничение глобальной path-проверки |
| Приватный комплект не доступен через Git | Ничего не копировалось | Data-dependent reproduction остаётся NOT TESTED |

Исправления затрагивают только README, DATA_NOTICE и статусы документации. Требования, модели, configs, scripts, tests, методология, числовые результаты и прежние final artifacts не изменены. Повторная проверка нового README выполнена против **структуры и кода чистого clone**; его относительные ссылки сохраняют те же targets. Новые формулировки ещё не опубликованы и не выдаются за содержимое клонированного commit.

## 9. Итог по этапам

| Этап | Статус | Граница вывода |
| --- | --- | --- |
| Remote synchronization и новый clone | PASS | Один зафиксированный GitHub commit |
| Структура и README links | PASS | Все объявленные tracked targets доступны |
| Fresh environment / dependencies | PASS | Python 3.12.10, основные requirements + Ruptures |
| No-fit pytest | PASS | Только 1432 выбранных checks; 34 исключены |
| Полный pytest без исключений | NOT TESTED | Запрещённые в F5 новые fit |
| CLI help, imports, YAML | PASS | Основное окружение; full runs не запускались |
| Chronos preflight с настоящими входами | NOT TESTED | Фактическая команда остановилась на отсутствии приватного CSV |
| Финальные отчёты, figures, PDF | PASS | Существование, чтение, SHA; без пересборки |
| Secrets / оперативные paths | PASS | Статическая проверка tracked-содержимого |
| Отсутствие абсолютных paths во всём repository | FAIL | Исторические команды/provenance сохранены |
| Git hygiene clone | PASS | Непреднамеренных tracked modifications нет |
| Data reproducibility | NOT TESTED | Нет приватного исходного комплекта |
| Full experiment reproducibility | NOT TESTED | Нет исходных inputs/outputs/weights; models не запускались |
| **Code reproducibility** | **PASS WITH LIMITATION** | Проверенный основной workflow и no-fit subset, без optional/full model ветвей |

**Финальный verdict: PASS WITH LIMITATION.** Внешний пользователь может установить основной код, выполнить проверенную no-fit часть tests и CLI help, прочитать финальные материалы. Full reproduction не подтверждено. Абсолютные исторические provenance paths, приватная data availability, optional learners и аудит прав на публикацию остаются отдельными ограничениями.

NO NEW MODEL FITS

NO NEW EXPERIMENTS

NO COMMIT/PUSH

F5 clean-clone reproduction audit complete. Ready for final audit.
