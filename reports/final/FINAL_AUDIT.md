# F6 — финальный audit публичного репозитория

Дата: 2026-10-07. **Verdict: PASS WITH LIMITATION.**

Base commit: `3371a44a90596551a897e56d585c89f00926157d` — F5. До изменений `main`, `origin/main` и GitHub `refs/heads/main` совпали; единственный untracked-каталог был `.vscode/`. F6 не создаёт commit и не публикует изменения.

## Проверки

| Область | Статус и основание |
| --- | --- |
| Финальные материалы | PASS: README, сводка, методологический отчёт, F5 audit, ограничения, терминология, PDF и figures существуют и tracked |
| Навигация README | PASS: вверху доступны методологический отчёт, PDF, сводка, ограничения; запрещённые выражения отсутствуют |
| Ссылки | PASS: 135 файловых ссылок и 4 внутренних якоря в README, отчёте, README презентации и F5 audit; все цели есть в tracked-дереве. Дополнительно проверена сводка |
| Figures / manifests | PASS: 11 PNG SHA, 35 source SHA figure manifest и 21 source SHA table manifest совпадают; файлы метрик и manifests сохранены |
| Презентация | PASS: PDF SHA совпадает с F4 verification; свежие `pdfinfo` / `pdftotext` завершились с exit 0: 12 страниц по 960 × 540 pt (16:9), 240 строк текста; без нового рендеринга/экспорта |
| Security | PASS: проверены 235 tracked-файлов, включая PNG metadata и PDF; рабочих credentials, private keys и private URLs не обнаружено |
| Git hygiene | PASS: `.vscode/`, venv, Python/pytest cache, build/dist/temp игнорируются; приватные данные и локальные reports остаются исключёнными |
| AI disclosure | PASS: README и отчёт согласованно указывают ChatGPT / Codex и контроль автора |
| Data notice | PASS: приватный CSV, его расположение, сохранённые outputs и границы действий без данных описаны; новая лицензия не заявляется |

## Claims consistency

Сопоставлены README, методологический отчёт и Markdown/HTML презентации с сохранёнными финальными CSV/JSON:

- SeasonalNaiveYoY лучше обоих Prophet на сопоставимом holdout h=1/3/6.
- National/Local + LightGBM имеет minimum holdout MAE среди девяти основных стратегий на h=1/3/6; преимущество не полностью устойчиво на validation.
- h=12 имеет одну origin; direct/National–Local используют SeasonalNaive fallback. Безусловный победитель не объявляется.
- Chronos-2 zero-shot не превзошёл сильные baselines; ограничение checkpoint / исторической интерпретации сохранено.
- Реальные detection alerts/breakpoints — диагностика без независимого ground truth; количественные detection metrics относятся к synthetic benchmark.
- News coverage недостаточно для надёжной оценки predictive uplift: 93 документа, 197 snapshots, 89 canonical groups, 26 features + 26 flags, 7 временных векторов.
- Реальный early-warning classifier не обучался: 73 weak events / 6 onset-дат, k=1 — 14 train / 4 test positives. Это результат проверки достаточности данных.
- Synthetic early warning проверяет механизм при наблюдаемых предвестниках; его метрики не оценивают предсказание реальных экономических шоков.

Таблицы, формулы и figure paths в README и отчёте сохранены. Числовой Source of Truth, модели, configs, tests, scripts, research outputs, старые experiment reports и содержимое слайдов не изменены.

## Absolute paths: решение

**Глобальное отсутствие абсолютных путей — FAIL с документированным исключением.** Остались 23 исторические строки в девяти файлах. A (runtime dependency) — 0; B (current instruction) — 0 после правки `docs/TASKS.md`. Все строки ниже — C (historical provenance), tracked; применимость ignore-правила к новым файлам не меняет их tracked-статус. Значения путей не являются credentials и не исполняются текущим pipeline.

| Файл | Строки base commit | Тип записи | Связь SHA |
| --- | --- | --- | --- |
| `reports/final/results_summary.json` | 9337, 9390 | executable в сохранённых command arrays | F4 source snapshot |
| `reports/results/E03_chronos_zero_shot.md` | 33 | выполненная CLI-команда | F1 frozen report |
| `reports/results/E04a_online_detection.md` | 5 | команды опыта и проверок | F1 frozen report |
| `reports/results/E05b_trend_calendar.md` | 280, 281; 288 | CLI-команды; metadata исполнителя | F1 frozen report |
| `reports/results/E05c_macro_forecast.md` | 527, 528 | CLI-команды | F1 frozen report |
| `reports/results/E05d_national_local_lightgbm.md` | 758, 759 | CLI-команды | F1 frozen report |
| `reports/results/E06a_offline_detection.md` | 414; 415, 416, 432 | test/CLI; check-command provenance | F1 frozen report |
| `reports/results/E07b_early_warning_full_panel.md` | 180–184; 186 | команды этапов; test-command provenance | F1 frozen report |
| `reports/results/E07c_synthetic_early_warning.md` | 317; 323 | CLI-команда; command field JSON проверки | F1 frozen report |

SHA всех восьми frozen reports совпадают с `results_summary.json.provenance`. Эти девять файлов сохранены побайтово: косметическая замена путей нарушила бы reproducibility trail. До cleanup в `docs/TASKS.md:93` было обозначение корня Windows-диска (B); оно заменено текстовым описанием. D: credential-URL в `tests/test_macro_cbr.py:173` — синтетическая проверка отказа, не рабочий секрет.

## Исправления и границы воспроизведения

- README получил две верхние ссылки. В отчёте пять hyperlinks к приватным/ignored-файлам заменены явными локальными provenance paths и публичными ссылками; исходные документы не опубликованы.
- В сводке ссылка на ignored AI draft заменена опубликованным disclosure README. README презентации корректно описывает tracked-статус файлов.
- `.gitignore` исключает IDE/build/temp artifacts и разрешает только новый `FINAL_AUDIT.md` среди ранее ignored reports. Обновлены фактические статусы в docs.

F4 `presentation/verification.json` остаётся историческим снимком, а не manifest актуальных текстов README/отчёта/сводки. Его пять source SHA проверены против Git-версии F4 (`6ada526`); последующие изменения затрагивают только навигацию и документационные формулировки. SHA PDF, figures и числовых источников актуальны и совпадают. 224 защищённых файла совпали с исходным F6 snapshot; исключены только три документа с исправленными ссылками. Исторические F1/F5 audit records не переписывались.

Воспроизводимость кода опирается на [F5 audit](REPRODUCTION_AUDIT.md): fresh GitHub clone / Python 3.12.10, installation и pip check PASS; 1432 passed / 34 deselected / 1 warning в no-fit pytest. В F6 pytest, модели, builder и экспорт PDF не запускались. Выполнены read-only Git/PowerShell, проверки ссылок, текста, структуры, SHA, `pdfinfo` / `pdftotext` и `git diff --check`. PDF-tools работали с workspace temp и отключёнными MiKTeX installer/maintenance/diagnose; после повторного запуска ошибок нет, временные файлы удалены.

Остаются ограничения: приватные raw data/исходные outputs/weights отсутствуют в Git; data/full experiment reproduction и optional learners NOT TESTED. Права на публикацию исходных и производных данных отдельно NOT VERIFIED; существующие curated CSV/PNG содержат выбранные реальные производные значения. F6 не добавляет данные. Исследовательские ограничения — 24 месяца, одна годовая origin, отсутствие real shock ground truth и ограниченное news coverage — сохраняются.

**Готово к финальному commit с указанными исключениями.** Новый audit пока untracked; commit/push должен выполняться отдельно. Свежий clone будущего commit в F6 не создавался.

NO NEW MODEL FITS

NO NEW EXPERIMENTS

NO COMMIT/PUSH

F6 final audit complete.

Ready for final commit.
