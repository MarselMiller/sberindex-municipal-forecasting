# Публикационный пакет и GitHub Pages

Планируемый адрес: [sberindex-municipal-forecasting](https://marselmiller.github.io/sberindex-municipal-forecasting/).
Сайт ещё не опубликован. Репозиторий планируется сделать публичным после review;
изменение видимости, commit/push и deployment не входят в подготовку пакета.

2026-10-09 автор утвердил публикацию **существующего** репозитория с сохранением
Git history, атрибуцию/CC BY-SA 4.0 для примера МО 21 и раскрытие исторических
путей/Git identities при отсутствии других blockers. Второй репозиторий,
новый публичный snapshot и переписывание истории исключены. Лицензия собственного
кода не назначена, MIT не добавлена. [Команды и короткая инструкция](PUBLISH_GITHUB_PAGES.md).

GitHub Pages публикует **`dist/submission-site/`**, собранный существующим
`scripts/build_publication_site.py`. Простая публикация `/docs` не включает
PDF и приложения из `reports/`, поэтому используется GitHub Actions.

## Локальная сборка из сохранённых результатов

Из корня репозитория, Python 3.12, в отдельном окружении:

```sh
python -m venv .venv-publication
# Windows: .venv-publication/Scripts/python; Linux/macOS: .venv-publication/bin/python
```

Следующие команды выполняйте Python из этого окружения:

```sh
python -m pip install -r requirements-publication.txt
python scripts/build_final_summary.py --editorial-only --check
python scripts/build_project_report.py --check
python scripts/build_publication_site.py --require-tracked-inputs
python scripts/build_publication_site.py --require-tracked-inputs --check
python -m pytest -q tests/test_publication_site.py tests/test_public_editorial.py tests/test_public_repository_audit.py tests/test_final_summary.py
python scripts/audit_public_repository.py --strict
git diff --check
```

Кандидат записывается в Git index для review; это делает входы tracked локально,
но до commit/push не подтверждает их доступность в GitHub. Проверка
`--require-tracked-inputs` обязательна и локально, и в Actions.

Проверка blockers 2026-10-09: безопасный `feature_coverage.csv` добавлен в index;
новый validation clone и экспорт candidate Git tree (361 tracked input) собраны
в новом окружении без private artifacts, 57 no-fit tests PASS. Это **CLEAN CLONE
REVIEW-CANDIDATE PASS**, а не утверждение, что изменения уже доступны в GitHub.
Неизменённый GitHub main clone всё ещё FAIL без coverage. После разрешённого
commit/push требуется GitHub-only проверка без candidate overlay.
[Полный review и решения автора](../reports/final/PUBLICATION_RIGHTS_REVIEW.md).

Результат: `dist/submission-site/` и ZIP с содержимым корня сайта.
Откройте `dist/submission-site/index.html` в браузере. Для локального HTTP-просмотра:

```sh
python -m http.server 8000 --bind 127.0.0.1 --directory dist/submission-site
```

Адрес локального сервера — `http://127.0.0.1:8000/`.
Нужно переносить всю папку сайта: CSS, JS, JSON, изображения и приложения
являются отдельными файлами. Runtime не использует CDN или внешние данные.
Новые прогнозы, метрики и model fits при сборке не выполняются.

## Ручной workflow

[`.github/workflows/pages.yml`](../.github/workflows/pages.yml) запускается только
через `workflow_dispatch`. Триггеров `push`, `pull_request` и расписания нет.
Параметр `deploy` по умолчанию **false**: ручной запуск проверяет и собирает пакет.

Workflow: checkout полной reachable-истории → Python 3.12 и минимальные
publication dependencies → проверка generated sources → сборка → тесты,
link checks и security audit → `upload-pages-artifact` → `deploy-pages`.
Deployment возможен только при явном `deploy=true`, запуске из `main`,
успешном build/security gate и допуске environment `github-pages`.
Actions закреплены полными commit SHA; build имеет только `contents: read`,
deployment — `pages: write` и `id-token: write`.

После review и устранения blockers владелец репозитория отдельно выполняет:

1. Финальное review diff; состав существующего Git и известные historical paths/identities приняты владельцем.
2. Commit/push и перенос проверенного workflow в `main`.
3. Settings → Pages → Build and deployment → Source → **GitHub Actions**.
4. Настройку protection/required reviewers для environment `github-pages`.
5. Ручной запуск из `main` с `deploy=true` и проверку фактического Pages URL.

Эти действия здесь не выполнялись. До успешного deployment ссылка в README
обозначена как планируемый адрес. `configure-pages` работает только в deploy job.

## Состав пакета и ссылки

Allowlist находится в [`configs/publication_site.json`](../configs/publication_site.json).
В пакет входят основной HTML, PDF, методология (12 таблиц, шесть display-формул,
девять рисунков), итоговая сводка, глоссарий, ограничения, отчёты Persistence,
Robustness и E08a/b/c/d, проверенные агрегированные CSV и национальные признаки.
Определения берутся из единственного canonical `reports/final/terminology.md`.

Включённые приложения открываются по локальным относительным URL. Их адреса
проверяются также под префиксом `/sberindex-municipal-forecasting/`.
Tracked-код, YAML/JSON, исторические аудиты и дополнительные отчёты вне пакета
ссылаются на существующие пути в GitHub. Анонимный доступ к ним до изменения
видимости не утверждается. Для local-only inputs/outputs показывается
локальное пояснение `references/materials.html#local-artifacts`, без broken link
и без обещания предоставить исходные данные.

`manifest.json` перечисляет размеры, SHA256 и источники всех файлов, кроме
самого manifest. SHA manifest и ZIP печатает builder. `link-audit.json`
содержит старые/новые URL, локальные проверки, решения по исключённым материалам,
статусы GitHub destinations и внешний inventory. При одинаковых исходных байтах
и окружении повторная сборка даёт одинаковые байты пакета и ZIP.

Внешние ссылки ЦБ/Мосбиржи сохранены как справочные. Предыдущая проверка получила
25 HTML/PDF-документов; три XLSX endpoint ответили spreadsheet content type,
их содержимое не проверялось. Для одного XLSX сохранён HTTP 403 и явная подпись
о неподтверждённом доступе. Это исторический review, не новый live HTTP-тест.
Работа сайта от этих внешних адресов не зависит.

Исторические E08 Markdown сохранены побайтово как provenance. В их публичном
HTML builder убирает рабочие заметки о review/commit и временных ошибках
подготовки; таблицы, независимые проверки, допущения и выводы сохраняются.
Встроенный PDF viewer Edge может отдельно запросить `/favicon.ico` в корень
домена. Browser QA фиксирует этот запрос интерфейса браузера отдельно;
все ссылки и runtime assets сайта остаются под project-prefix.

## Безопасность всего публичного репозитория

Безопасный allowlist сайта **не означает**, что безопасно раскрывать весь Git.
[`scripts/audit_public_repository.py`](../scripts/audit_public_repository.py)
проверяет tracked-файлы, каждый reachable blob под всеми историческими именами,
commit/tag metadata; matched секреты и значения локальных путей не выводятся.
Локальные доказательства сохраняются только в ignored `outputs/`.

**Data rights review 2026-10-09: PUBLICATION PERMITTED** для существующего
12-строчного примера `reports/final/figure_data/rolling_forecast.csv` при
соблюдении CC BY-SA 4.0. Он содержит исходные наблюдения, присутствует в истории
и не включается в сайт. Официальный API именно набора расходов 2023–2024
подтвердил grant; локальный PDF согласуется с ним. Проверка 12 строк против
сохранённых прогнозов и target прошла. [Доказательства](../data/metadata/publication_rights_review.json)
и [согласованная атрибуция](../reports/final/figure_data/README.md) сохранены;
автор принял CC BY-SA 4.0 для соответствующего адаптированного материала
и своего вклада. Лицензия кода не назначена. Старые F7-аудиты не изменяются.

Auditor допускает только точный SHA256 этого CSV под его исходным именем
при наличии metadata review и attribution notice. Другой путь или изменённое
содержание снова дают BLOCKER; secret/path checks не отключаются. Для этого
примера возвращается REVIEW, а не безусловное разрешение публикации.

Исторические локальные пути и Git identity metadata приняты владельцем к раскрытию.
Эвристический аудит не доказывает отсутствие всех секретов/PII и не проверяет
необъявленные remote refs, dangling objects, GitHub releases/attachments.
Полные журналы сохраняются как provenance, а не подменяются новой проверкой.

В сайт не копируются raw/parquet, построчные прогнозы, муниципальные CSV,
веса, архивы данных, `.env`, ключи, окружения, `.git` или целые `outputs/`.
Существующий рисунок МО 21 сохранён побайтово; его источник и CC BY-SA 4.0
указаны в приложении «Материалы». Права на справочник и другие private inputs
этим решением не подтверждаются. `PASS_WITH_REVIEW` security audit сохраняет
видимыми принятые замечания; решение автора записано отдельно в rights metadata.
Manual workflow и отсутствие автоматического deployment сохраняются.

## Проверка браузером

`python scripts/verify_publication_browser.py` использует установленный Edge
в Windows, копирует сайт в отдельную временную папку, блокирует внешние запросы
и проверяет desktop/mobile, тему, клавиатуру, no-JS, переключатели,
PDF, методологию и глоссарий. Доказательства и скриншоты — только в ignored
`outputs/publication_site_checks/browser/`. Для обычного просмотра Edge
конкретной версии и Python не требуются.

`python scripts/verify_publication_browser.py --project-prefix` повторяет
проверку на изолированном loopback HTTP-сервере под
`/sberindex-municipal-forecasting/`. Он обслуживает только временную копию пакета,
не публикует её во внешней сети; доказательства —
`outputs/pages_publication_checks/browser-prefix/`.
