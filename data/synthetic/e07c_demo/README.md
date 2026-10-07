# SYNTHETIC: сохранённый технический demo

Это пять файлов из ранее выполненного технического smoke run. Они скопированы
побайтово; F7 не запускал генератор, обучение или оценку. Здесь нет расходов
реальных МО, настоящих macro/news наблюдений или разметки экономических шоков.

В demo 40 TRAIN, 16 VALIDATION и 20 TEST рядов по 24 месяца. Cohort seeds:
421101 / 421201 / 421301; это seeds smoke run, а не полного benchmark.
[Generator config](generator_config.json) задаёт параметры генерации;
[полная конфигурация](../../../configs/early_warning_synthetic.yaml) и
[генератор](../../../src/sberforecast/early_warning_synthetic.py) уже входят в Git.
Внутри конфигурации smoke sizes — 40/16/20, seed offset — 1000.

`observations.csv.gz` содержит наблюдения и синтетические внешние каналы.
`true_events.csv` и `series_metadata.csv` содержат известное генератору будущее:
это разметка и описание сценариев, **не входные features модели**.
`cohort_ids.csv` фиксирует раздельные cohorts. Восстановление полного benchmark
требует полного seed/config, а не расширения этого demo; команды с model fits
в этой папке не выполняются.

Проверка файлов, SHA256 и схем без сторонних пакетов, из корня проекта:

```powershell
py -3.12 scripts/check_data.py
```

Это проверка сохранённого data layer. Метрики полного synthetic benchmark
остаются в [финальной сводке](../../../reports/final/RESULTS_SUMMARY.md).
