# Самостоятельный публикационный пакет

Пакет готовится из существующих `docs/` и проверенных итоговых материалов.
Основной GitHub-репозиторий остаётся **PRIVATE**. Сборка не публикует сайт,
не меняет visibility и не запускает модели. Публичного адреса пока нет.

## Сборка и проверка

Из корня проекта в существующем Python-окружении с Matplotlib:

```powershell
python scripts/build_project_report.py --check
python scripts/build_publication_site.py
python scripts/build_publication_site.py --check
python -m pytest tests/test_publication_site.py -q -p no:cacheprovider
python scripts/verify_publication_browser.py
git diff --check
```

Последняя browser-команда использует установленный Microsoft Edge в Windows,
копирует пакет в отдельную временную папку и блокирует внешние запросы страницы.
Она сохраняет доказательства и скриншоты в ignored
`outputs/publication_site_checks/browser/`. Ничего устанавливать для неё не нужно.
Проверяющему Python, Edge конкретной версии или локальный сервер не требуются:
сайт открывается обычным браузером, в том числе напрямую через `index.html`.

Результат: `dist/submission-site/` и `dist/submission-site.zip`.
ZIP содержит содержимое корня сайта, включая `index.html`, без внешней папки.
`dist/` и ZIP игнорируются Git. Не добавляйте их автоматически в репозиторий.

`manifest.json` перечисляет все файлы, размеры и SHA256, кроме самого manifest:
его собственный SHA256 и SHA256 ZIP печатает builder. `link-audit.json` содержит
точные старые/новые URL, решения о доступе, результаты внутренних проверок,
объём Markdown-конверсии и отдельный статус каждого внешнего источника.
Время ZIP фиксировано, файлы отсортированы; повторная сборка из тех же байтов
источников и того же окружения даёт те же байты пакета и ZIP.

## Что открывается без GitHub

- Главный интерактивный отчёт со всеми существующими переключателями и поиском.
- Презентация `presentation/presentation.pdf`, побайтовая копия текущего PDF.
- Методология `references/methodology.html`: все 12 таблиц, шесть формул,
  девять рисунков и исходный Markdown для сверки содержания.
- Глоссарий, итоговая сводка, ограничения и README в локальном HTML.
- Отчёты National/Local Persistence, forecasting robustness, E08a/b/c/d.
- Проверенные агрегированные CSV и национальные origin features E08b.

Определения берутся из `reports/final/terminology.md`. Главный HTML/data bundle
копируется из canonical `docs/`; builder меняет только адреса ссылок и подписи
ограниченного доступа в выходном HTML. Формулы рендерятся в локальные SVG,
исходный LaTeX доступен рядом; MathJax, KaTeX и CDN не используются.

Код, подробный E06a, служебные аудиты, полные run/verification manifests,
`results_summary.json`, остальные конфигурации и research artifacts остаются
в приватном репозитории. Их ссылки ведут на доступное пояснение
`references/restricted.html` и явно подписаны «доступ по приглашению».
Это пояснение не предоставляет доступ к самим исключённым материалам.

## Решения по 18 исходным адресам главной страницы

Повторные появления одного URL объединены. Полные адреса и все ссылки приложений
также сохранены в `link-audit.json`.

| URL in original HTML | URL in publication package | Destination type | Anonymous accessibility | Publication status |
| --- | --- | --- | --- | --- |
| `https://github.com/MarselMiller/sberindex-municipal-forecasting` | тот же URL | private repository | требуется приглашение | явно подписан «Исходный код — доступ по приглашению» |
| `https://github.com/MarselMiller/sberindex-municipal-forecasting/blob/main/reports/final/METHODOLOGY_REPORT.md` | `references/methodology.html` | local HTML | да | включён |
| `https://github.com/MarselMiller/sberindex-municipal-forecasting/blob/main/reports/final/presentation/presentation.pdf` | `presentation/presentation.pdf` | local PDF | да | включён без изменения байтов |
| `https://github.com/MarselMiller/sberindex-municipal-forecasting/blob/main/reports/final/forecasting_metrics.csv` | `references/forecasting_metrics.csv` | aggregate CSV | да | включён без изменения байтов |
| `https://github.com/MarselMiller/sberindex-municipal-forecasting/blob/main/reports/results/E08a_leading_financial_indicators_audit.md` | `references/financial-sources.html` | local HTML | да | включён |
| `https://github.com/MarselMiller/sberindex-municipal-forecasting/blob/main/reports/results/E08b_leading_financial_features.md` | `references/financial-features.html` | local HTML | да | включён |
| `https://github.com/MarselMiller/sberindex-municipal-forecasting/blob/main/reports/results/E08c_leading_financial_forecasting.md` | `references/financial-forecasting.html` | local HTML | да | включён |
| `https://github.com/MarselMiller/sberindex-municipal-forecasting/blob/main/reports/results/E08d_real_financial_early_warning.md` | `references/financial-warning.html` | local HTML | да | включён |
| `https://github.com/MarselMiller/sberindex-municipal-forecasting/blob/main/reports/final/REPRODUCTION_AUDIT.md` | `references/restricted.html#audits` | access notice | доступно пояснение; оригинал по приглашению | исключён служебный журнал |
| `https://github.com/MarselMiller/sberindex-municipal-forecasting/blob/main/reports/final/DATA_PUBLICATION_AUDIT.md` | `references/restricted.html#audits` | access notice | доступно пояснение; оригинал по приглашению | исключён служебный журнал |
| `https://github.com/MarselMiller/sberindex-municipal-forecasting/blob/main/reports/final/terminology.md` | `references/glossary.html` | local HTML | да | canonical определения сохранены |
| `https://github.com/MarselMiller/sberindex-municipal-forecasting/blob/main/reports/final/RESULTS_SUMMARY.md` | `references/results.html` | local HTML | да | включён |
| `../reports/results/national_local_persistence.md` | `references/national-local-persistence.html` | local HTML | да | включён агрегированный отчёт |
| `../reports/results/national_local_persistence/gap_analysis.csv` | `references/persistence/gap_analysis.csv` | aggregate CSV | да | включён без изменения байтов |
| `../reports/results/forecast_robustness.md` | `references/forecast-robustness.html` | local HTML | да | включён агрегированный отчёт |
| `../reports/results/forecast_robustness/pairwise_metrics.csv` | `references/robustness/pairwise_metrics.csv` | aggregate CSV | да | включён без изменения байтов |
| `../reports/results/forecast_robustness/bootstrap_origin.csv` | `references/robustness/bootstrap_origin.csv` | aggregate CSV | да | включён без изменения байтов |
| `../reports/results/E06a_offline_detection.md` | `references/restricted.html#offline-detection` | access notice | доступно пояснение; оригинал по приглашению | исключены локальные пути и подробные муниципальные таблицы |

## Проверка внешних источников

На 09.10.2026 web-инструмент получил 25 внешних HTML/PDF-документов ЦБ/Мосбиржи.
Три XLSX endpoint ответили spreadsheet content type; содержимое этих файлов
не проверялось и не включалось в пакет. Один адрес
`https://www.cbr.ru/vfs/statistics/banksector/borrowings/02_27_Dep_ind_excluding_escrow.xlsx`
вернул HTTP 403: в HTML-приложении показано, что доступ не подтверждён.
Единственный GitHub URL требует приглашения по условиям задачи.

Эти 30 адресов сверены с полным inventory. Получение документа web-инструментом
может использовать его cache и не означает live HTTP 200 из каждой сети.
Источники приведены как внешние справочные ссылки; работа сайта от них
не зависит. Фиксированный review хранится в
`configs/publication_external_links.json`; builder не обращается к сети.

## Allowlist и ограничения публикации

Состав явно задан в `configs/publication_site.json`. Не копируются целые
`reports/`, `outputs/`, `data/` или репозиторий. В пакет не входят parquet,
построчные predictions, муниципальные CSV, веса, архивы данных, `.env`, ключи,
виртуальные окружения, `.git`, локальные пути или непроверенные вложения.
Включённые CSV содержат агрегаты, национальные финансовые значения или
сводные диагностические counts; строки отдельных муниципальных расходов
и прогнозов исключены. NA и числовые результаты не изменяются.

Право на перераспространение исходных parquet остаётся **UNCLEAR**.
Существующий рисунок МО 21 сохранён как ранее проверенная иллюстрация отчёта:
SHA256 `0ae922414791a15ad63f1aefc4f7458faa4bdee1e2da53baa2f04da0c143d6c2`.
Копии основной страницы и методологии идентичны. Исходные точки и другие
муниципальные таблицы не добавляются. Это не подтверждает права на raw data.

## Netlify Drop — действия автора после review

1. Соберите и проверьте пакет. Откройте `dist/submission-site/index.html`.
2. Откройте [Netlify Drop](https://app.netlify.com/drop).
3. Перетащите папку **submission-site**, внутри которой сразу находится
   `index.html`. ZIP для этого способа сначала распакуйте.
4. Сохраните фактически выданный адрес и проверьте его в инкогнито.

Для обновления загрузите заново собранную папку в раздел deploys того же сайта.
Не загружайте корень репозитория. Официальная инструкция:
[Create deploys](https://docs.netlify.com/deploy/create-deploys/).

## Cloudflare Pages Direct Upload — действия автора после review

1. В dashboard откройте **Workers & Pages → Create application → Get started →
   Drag and drop your files** для Pages.
2. Загрузите `dist/submission-site.zip` либо папку `submission-site`.
   В корне загружаемого содержимого должен лежать `index.html`.
3. После проверки состава автор выбирает **Deploy site**.
4. Проверьте выданный адрес в инкогнито. Следующие версии загружайте как новый
   deployment того же проекта.

Официальная инструкция: [Direct Upload](https://developers.cloudflare.com/pages/get-started/direct-upload/).
Публичный GitHub или подключение приватного репозитория для этого способа не нужны.

## Проверка опубликованного адреса и обновления

Откройте адрес в отдельном окне инкогнито без GitHub-сессии. Проверьте основной
HTML, PDF, методологию с формулами/рисунками, глоссарий, CSV, переключатели,
поиск, светлую/тёмную тему, мобильную ширину и отсутствие ошибок console/network.
Для этих документов не должен появляться login. Ссылки по приглашению должны
открывать пояснение, а переход в GitHub — требовать предоставленного доступа.
Доступ сайта на выбранном хостинге проверяется отдельно от GitHub visibility.

Меняйте canonical HTML/JSON/PDF в проекте. Для изменения словаря или агрегатов
сначала используйте существующий `build_project_report.py` и его `--check`.
Затем повторите publication build, targeted tests, browser QA и `--check`.
Не редактируйте `dist/` вручную. При новом внешнем URL обновите link review;
при новом приложении или рисунке сначала пересмотрите allowlist и безопасность.

В рамках подготовки пакета deployment, создание сайтов/репозиториев,
изменение visibility, commit, push и merge не выполняются.
