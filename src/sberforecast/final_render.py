"""Russian F1 narrative and tables, rendered from the canonical evidence object."""
from __future__ import annotations

from .final_summary import forecasting_table, fmt, table


def render_summary(s: dict) -> str:
    forecasts = s['forecasting']['records']
    facts = s['forecasting']['facts']
    detection = s['detection']['records']
    news, macro = s['news'], s['external']
    real = s['real_early_warning']['E07b']
    pilot = s['real_early_warning']['E07a']
    synthetic = s['synthetic_early_warning']
    primary = sorted([r for r in synthetic['metrics'] if r['scenario'] == 'base_all'], key=lambda r: (r['k'], r['model']))
    synopsis = []
    for k in (1, 3):
        rows = {r['scope']: r for r in real['counts'] if r['k'] == k}
        all_, train, test = rows['all'], rows['train'], rows['test']
        synopsis.append([k, all_['fully_known'], all_['positives'], all_['negatives'], all_['censored_or_missing'],
            all_['right_censored'], train['positives'], test['positives'],
            f"{all_['positive_event_onset_dates']} / {train['positive_event_onset_dates']} / {test['positive_event_onset_dates']}"])
    paragraphs = ['# Итоговая сводка результатов СберИндекс',
        'F1 — единый источник чисел для будущих README, методологического отчёта и PDF-презентации. '
        'Research phase закрыта на `be6d993`. Сводка получена из сохранённых artifacts: новых model fits, настройки и исследовательских экспериментов нет.',
        'Цель — месячное значение категории **«Все категории» по МО**, оценка средних безналичных расходов жителей '
        'в исходных номинальных рублях. История: январь 2023 — декабрь 2024. '
        'Прогнозирование, обнаружение уже начавшегося изменения и предупреждение будущего события рассматриваются отдельно.',
        '**Статусы данных:** `real` — метрики непосредственно измерены на СберИндексе; coverage news/макро описывает реально сохранённые источники; `synthetic` — controlled benchmark; '
        '`diagnostic` — реальные данные без независимой истины для оценки качества; `not_evaluated` — качество не оценено. '
        'Пустое значение обозначается «—» в тексте, null в JSON и пустой ячейкой в CSV.',
        '## A. Forecasting: сопоставимый реальный пилот',
        f"Область `real`: {facts['comparison_area']['evaluable_municipalities']} оцениваемых МО из 64 первоначальных, "
        f"{facts['comparison_area']['evaluable_keys']} одинаковых прогнозных ключей на каждую стратегию. "
        'МО 1471 сохранено в исходной выборке, но его факты не позволяют оценку; 7 строк без конечного факта остаются в исходных predictions. '
        'Данные full panel baseline в таблицу не добавлены. Ключ сравнения: МО × forecast origin × target month × h. '
        'Все факты и множества ключей сверены, fallback оставлен в полной стратегии.',
        '**MAE macro, руб.** — сначала MAE каждого МО, затем среднее с равными весами МО. '
        'MAE micro, pooled R², число прогнозов, МО, origin, native и fallback отдельно сохранены в '
        '[forecasting_metrics.csv](forecasting_metrics.csv). Pooled R² характеризует общую группу и не заменяет качество динамики каждого МО.',
        '### Validation: цели до июня 2024 включительно',
        forecasting_table(forecasts, 'validation'),
        'На h=1/3/6 — соответственно 378/252/63 прогнозных случая и 6/4/1 forecast origin; '
        'h=12 не имеет validation-случаев (`not_evaluated`). Остальные ячейки — `real`.',
        '### Holdout: цели июля–декабря 2024',
        forecasting_table(forecasts, 'holdout'),
        '**Holdout был просмотрен в ходе дальнейших исследований; последующие эксперименты после E01 не являются новой полностью независимой проверкой.** '
        'На h=1/3/6 — по 378 случаев и 6 origin, на h=12 — 63 случая и 1 origin.',
        '\n\n'.join(facts['narrative']),
        '**Годовой горизонт требует отдельного прочтения.** Единственный выпуск — декабрь 2023 → декабрь 2024. '
        'У CatBoostDirect, LightGBMDirect и обоих National/Local learners на этой origin нет исторических годовых training pairs. '
        'Все их 63 оцениваемых годовых значения — SeasonalNaive fallback († в таблице); это MAE полной стратегии, '
        'а не обученной годовой direct-модели. Validation h=12 отсутствует. Минимальное значение ProphetAuto на единственном срезе '
        'не устанавливает устойчивый ranking и не служит выбором модели по holdout.',
        'National/Local прогнозирует национальную медиану причинным SeasonalNaiveYoY и локальное отношение расходов МО к этой медиане; '
        'будущий национальный факт не входит в признаки/восстановление прогноза. Контраст меняет одновременно шкалу признаков, метки и национальный прогноз. '
        'Chronos-2 использован zero-shot, без дообучения: современный checkpoint выпущен 20.10.2025, позже backtest 2023–2024. '
        'Историческая доступность весов и отсутствие pretraining overlap не доказаны. '
        'Промежуточные тренды и макро-ablation остаются в исходных отчётах и не расширяют эту основную таблицу.',
        'Provenance forecasting: E01/E02/E03/E05d, сохранённые `predictions.csv.gz`, metrics и `strategy_keys.csv`; '
        'в каждой итоговой CSV-строке записаны путь, SHA256, исходное имя модели, фильтр и ключи.',
        '## B. Structural change detection',
        'Основная synthetic truth — устойчивый положительный/отрицательный сдвиг уровня не менее трёх месяцев. '
        'Качество оценено на 360 основных test-событиях; параметры выбраны на synthetic validation при бюджете контрольных тревог. '
        'Проверяются изменения ошибки причинного SeasonalNaiveYoY. Сигнал такой ошибки сам по себе не доказывает экономический шок.',
        '### Online: только прошлое и текущий месяц',
        table(['Метод', 'data_status', 'Precision', 'Recall', 'F1', 'Miss rate', 'Медиана delay, мес.', 'FP / 12 мес.'],
            [[r['method'], r['data_status']] + [fmt(r[key]) for key in ('precision', 'recall', 'f1', 'miss_rate', 'median_detection_delay', 'false_positives_per_12_months')]
             for r in detection if r['family'] == 'online' and r['data_status'] == 'synthetic']),
        'Delay относится только к обнаруженным событиям: нулевая медиана EWMA не устраняет высокий miss rate. '
        'FP — несопоставленные сигналы, включая повторы; знаменатель — 2880 наблюдаемых месяцев мониторинга основных level-сценариев. '
        'Synthetic benchmark — quality metrics. Real data — diagnostic alerts only.',
        '### Offline: весь анализируемый отрезок',
        table(['Метод', 'data_status', 'Precision', 'Recall', 'F1', 'Miss rate', 'Медиана localisation error, мес.', 'FP / 12 мес.'],
            [[('Binary Segmentation' if r['method'] == 'BinSeg' else r['method']), r['data_status']] +
             [fmt(r[key]) for key in ('precision', 'recall', 'f1', 'miss_rate', 'median_absolute_localisation_error', 'false_positives_per_12_months')]
             for r in detection if r['family'] == 'offline' and r['data_status'] == 'synthetic']),
        'PELT/Binary Segmentation видят данные после оцениваемой точки. Localisation error — ошибка ретроспективной границы '
        'только у обнаруженных событий, не online delay и не lead time. Успешное окно T…T+3 одностороннее, поэтому медиана абсолютной ошибки '
        'совпадает с медианой offset по определению. Synthetic test повторяет ранее просмотренные реализации online benchmark; '
        'общий победитель online/offline не выбирается.',
        ('Для native ruptures PELT точность глобального минимума objective при min_size=2 не гарантирована: '
         f"сохранённая независимая проверка отметила {s['detection']['facts']['pelt_native_limitation']['n_objective_gap_cases']} "
         f"расхождений в {s['detection']['facts']['pelt_native_limitation']['n_native_vs_DP_comparisons']} контекстах. "
         'Финальные метрики относятся к фактическим неизменённым native partitions, а не альтернативному точному solver.'),
        '### Реальные МО: только диагностика',
        table(['Метод', 'data_status', 'МО', 'Сигналов / полных кандидатов', 'Monitoring / warmup кандидаты', 'Пересмотры дат / потери присутствия'],
            [[r['method'], r['data_status'], r['n_series'], r.get('n_alarms', r.get('n_candidates')), 
              '—' if r['family'] == 'online' else f"{r['n_monitoring_candidates']} / {r['n_warmup_candidates']}",
              '—' if r['family'] == 'online' else f"{r['n_date_revisions']} / {r['n_presence_losses']}"]
             for r in detection if r['data_status'] == 'diagnostic']),
        '63 МО, 504 месяца мониторинга на online-метод. Реальные alerts не являются true positives. '
        'Prefix stability означает повторное сопоставление сохранённых сегментаций удлиняющихся префиксов с итоговыми кандидатами; '
        'оно использует ретроспективную связь и не доказывает предупреждение. Наблюдаемые пересмотры/исчезновения границ сохранены '
        'в [detection_metrics.csv](detection_metrics.csv). Независимых real shock labels нет; precision/recall реальных методов не вычислены.',
        'Provenance detection: E04a/E06a, synthetic saved signals/breakpoints, events и их one-to-one matches; '
        'real alarm stream и prefix diagnostics. Метрики независимо восстановлены из этих таблиц без запуска детекторов.',
        '## C. News / external data: реализованная подготовка и ограниченное покрытие',
        table(['News v3: показатель', 'Значение', 'data_status'], [
            ['URL-документы / snapshots', f"{news['documents']} / {news['snapshots']}", 'real'],
            ['Canonical группы / ядра решений ЦБ', f"{news['canonical_events']} / {news['admitted_core_records']}", 'real'],
            ['Исторически допущенные URL к последней O', news['historical_documents_at_last_origin'], 'real'],
            ['Сетка coverage', f"{news['municipalities']} МО × {news['forecast_origins']} origin = {news['feature_rows']} строк", 'real'],
            ['News value features / missing flags', f"{news['value_features']} / {news['missing_flags']}", 'real'],
            ['Строки / origin с наблюдаемыми news_30d > 0', f"{news['nonzero_30d_rows']} / {news['nonzero_30d_origins']}", 'real'],
            ['Уникальные временные news vectors', news['unique_temporal_vectors'], 'real'],
            ['Исторические regional / municipal события', f"{news['historical_by_geography']['regional']} / {news['historical_by_geography']['municipality']}", 'real'],
        ]),
        'Canonical groups — группы дедупликации, а не независимая экспертная разметка экономических событий. '
        '17 ядер решений ЦБ содержат только проверенную первую фразу: 16 доступны до последней origin, '
        'решение 20.12.2024 после неё. Четыре PDF объединены с соответствующими релизами, поэтому групп 89, '
        'а не сумма документов и версий. В корпусе 21 исторически допущенный snapshot по политике, '
        'к последней origin доступны 20 URL-документов. Это разные знаменатели.',
        'Pipeline временного согласования news реализован, но имеющаяся историческая выборка недостаточна '
        'для надёжной оценки incremental contribution news к real-data early warning. '
        'Все исторически допустимые события national; тиражирование на 64 или 2190 МО не создаёт независимых пространственных новостей. '
        'Полнота архива не подтверждена: отсутствие наблюдаемой публикации не означает отсутствие события. '
        'Исторический допуск решений основан на явном доверии датированному официальному архиву, а не независимых снимках 2023–2024. '
        'E06b v2 (180 snapshots/93 groups/4 URL) и E07a v3 — сохранённые версии одной работы, их результаты не складываются.',
        table(['Макро-аудит', 'Результат', 'data_status'], [
            ['Реально использованы', f"Корпус A: {macro['rows_by_availability_class']['A']} проверенных forecast-строк; в E05c присоединены национальные ожидания инфляции Dec/Dec и прогноз реального потребления", 'real'],
            ['Scenario-only', f"B: {macro['rows_by_availability_class']['B']} строки современных сводных таблиц; исторические vintages не подтверждены, в E05c не использованы", 'diagnostic'],
            ['Недоступны', f"Региональные месячные ИПЦ/зарплата: {macro['regional_monthly_cpi_and_wage_observations']} наблюдений; deflation не выполнен", 'not_evaluated'],
        ]),
        'A/B обозначают историческую подтверждённость по протоколу источников, отдельно от data_status результата. '
        'Национальные годовые forecasts не выдаются за месячные региональные факты. '
        'Средина диапазона потребления — явно производное значение. Макро-ablation E05c показывает смешанный результат '
        'и не устанавливает устойчивый прирост. Provenance: E05a/c, E06b, E07a и news v3 saved documents/features/coverage.',
        '## D. Early warning: реальные ограничения и отдельный synthetic benchmark',
        '**На реальных данных classifier не обучался: данных недостаточно для честной temporal evaluation.** '
        'Перед обучением проверено число положительных меток и независимых onset-дат. '
        'Это методологическое ограничение имеющейся истории, а не техническая ошибка.',
        f"Полная real-панель: **{real['municipalities']} МО, {real['weak_events']} weak events, "
        f"{real['weak_event_onset_dates']} onset-дат, {real['municipalities_with_event']} МО с событиями**. "
        f"Предыдущий пилот: {pilot['municipalities']} МО, {pilot['weak_events']} weak events, {pilot['weak_event_onset_dates']} onset-дат. "
        'Weak labels описывают трёхмесячное устойчивое изменение ошибки SeasonalNaiveYoY относительно четырёх прошлых ошибок; '
        'они не являются независимой истиной об экономических шоках.',
        table(['k', 'Known', 'Positives', 'Negatives', 'Unknown: censored/missing', 'Right-censored', 'Train pos.', 'Test pos.', 'Onset-дат all/train/test'], synopsis),
        'Counts первой части таблицы относятся ко всем audit cases, а train/test — к eligible, известным и at-risk меткам после причинного исключения '
        'подтверждённого активного режима. Поэтому positives registry/all и train/test не следует механически складывать. '
        'Unknown объединяет отсутствующую информацию и цензурирование, right-censored — его отдельная часть. '
        'Cutoff знания меток — июль 2024, test origin — август–ноябрь. Реальная weak-метка требует информации до O+k+2: '
        'k=1 оставляет всего одну train и две test origin, k=3 не оставляет ни одной. '
        'Пороги достаточности train/test ≥30/10 positives и ≥3/2 onset-дат не выполнены; '
        'все 24 сохранённые календарные альтернативы также отказали. B0–B4 пропущены, их quality metrics — `not_evaluated`, не нули.',
        '### Synthetic early warning: controlled TEST, отдельно от реальных данных',
        '**Synthetic benchmark демонстрирует работоспособность методологии при наличии наблюдаемых предвестников, '
        'но не является оценкой способности прогнозировать реальные экономические шоки.** '
        'Независимые TRAIN/VALIDATION/TEST cohorts: 600/200/300 рядов по 24 месяца, 360/120/180 событий. '
        'S0 — train prior; S1 — история; S2 — история + online detector state; S3 — предыдущие группы + stochastic external macro/news channels. '
        'Параметры и генератор фиксированы; подготовка обучена на TRAIN, threshold выбран на VALIDATION, TEST не выбирает настройку.',
        table(['k', 'Модель', 'data_status', 'PR-AUC', 'Row F1', 'Event recall', 'Alert precision', 'Median lead, мес.', 'False alerts / 12 мес.'],
            [[r['k'], r['model'], r['data_status']] + [fmt(r[key]) for key in ('pr_auc', 'row_f1', 'event_recall', 'alert_precision',
                                                                                     'median_lead_time_months', 'false_alerts_per_12_monitored_months')]
             for r in primary]),
        'PR-AUC здесь — average precision. Event recall предупреждает одно событие один раз; повторные успешные alerts удалены из precision. '
        'False alerts — месячные ложные сигналы на 12 полностью известных at-risk месяцев, 2722/2482 месяцев для k=1/3. '
        'Lead time относится только к успешно предупреждённым событиям. S0 не выдаёт alerts, поэтому его alert precision и lead time не определены.',
        table(['S3: отдельный сценарий', 'k', 'Событий', 'Event recall', 'False alerts / 12 мес.', 'data_status'],
            [[{'anticipated_events': 'Наблюдаемые предвестники', 'unanticipated_events': 'Без предвестников',
               'controls_false_precursor': 'Ложные precursor controls'}[r['scenario']], r['k'], r['eligible_events'],
              fmt(r['event_recall']), fmt(r['false_alerts_per_12_monitored_months']), r['data_status']]
             for r in synthetic['metrics'] if r['model'] == 'S3' and r['scenario'] in ('anticipated_events', 'unanticipated_events', 'controls_false_precursor')]),
        'Сценарии с/без предвестников включают общую группу controls для оценки ложных alerts; сценарий ложных предвестников '
        'содержит 42 controls без истинных событий, поэтому event recall не определён. '
        'S3 способен использовать наблюдаемые precursor channels именно в этом controlled setting. '
        'Рост качества не является доказательством полезности реальных новостей. Частота ложных alerts в false-precursor controls превышает '
        'validation-бюджет: контрольный бюджет не гарантирует каждый TEST-поднабор. 25% TEST-событий не имеют предвестников; '
        'референс 75% не является строгим потолком recall из-за случайных/prior предупреждений.',
        '48 whole-series bootstrap 95% интервалов (500 повторов) сохранены в JSON и исходном `bootstrap_intervals.csv`; '
        'они условны на генераторе/моделях/порогах. Успех, ложная тревога и пропуск выбраны автоматически по первому ID, '
        'а не по красоте результата. Provenance: E07a/b label/event tables и E07c predictions/event_matches/alerts/scenario metrics.',
        '## E. Что проверено, ограничения и воспроизводимость',
        table(['Компонент', 'data_status', 'Метрика/проверка', 'Статус'],
            [[r['component'], r['data_status'], r['metric'], r['status']] for r in s['verified_components']]),
        'Соответствие семи конкурсным критериям и весам 10/20/20/15/15/10/10% — '
        '[competition_criteria.md](competition_criteria.md) и CSV/JSON. Это карта доказательств, баллы жюри не присваиваются.',
        '### Восемь основных ограничений',
        '\n'.join(f"{i}. {r['text']}" for i, r in enumerate(s['limitations'], 1)),
        '### Финальные графики и источник чисел',
        'Выбрано девять смысловых групп (11 PNG, поскольку success/false/miss — три отдельных примера). '
        'Каждая figure имеет исходные CSV/Parquet, SHA и правило выбора в [figure_manifest.csv](figure_manifest.csv). '
        'Новые графики строятся исключительно из сохранённых результатов; выбранные offline/E07c PNG скопированы без пересчёта моделей. '
        'Построенные агрегаты и реальный rolling-пример сохранены в `figure_data/`; это локальные производные данные, '
        'их публичная публикация требует отдельной проверки.',
        table(['Группа', 'Файл', 'data_status'], [[r['group'], f"[{r['title']}]({r['figure']})", r['data_status']] for r in s['figures']]),
        '### Проверка сводки и воспроизводимость',
        'Команда сборки: `.\\.venv\\Scripts\\python.exe -B -X utf8 scripts/build_final_summary.py`. '
        'Сборщик сверяет числовые таблицы отчётов с сохранёнными predictions/events и independently reconstructed counts/metrics; '
        'при существенном расхождении останавливается. JSON и Markdown формируются из той же структуры, что CSV. '
        'Сохранены research HEAD, текущий Git/status, команда, версии, SHA входных файлов и кода. '
        'F1 не запускает модели, детекторы, optimizer, загрузки или новый эксперимент.',
        'Один полный pytest запускается отдельно после сборки; фактический результат и визуальная проверка фиксируются '
        'в `verification.json`, а не приписываются сборщику заранее. `build_validation.json` подтверждает неизменность '
        'защищённых прежних файлов во время сборки; область проверки указана явно. '
        'Воспроизведение из чистой копии и финальный аудит публичной публикации остаются непроверенными. '
        'Исходные данные и веса не входят в итоговую сводку, `.gitignore` не менялся.',
        '### Обнаруженные несовпадения и подготовленная формулировка',
        ('\n'.join('- ' + r['description'] for r in s['discrepancies']) if s['discrepancies'] else 'Числовых расхождений между проверенными ключевыми таблицами прежних отчётов и сохранёнными данными не найдено.'),
        'Смена news v2 → v3 и нативной годовой модели → fallback являются различиями версии/области, а не исправлением старых результатов. '
        'Финальная формулировка AI disclosure подготовлена в [ai_disclosure_draft.md](ai_disclosure_draft.md); '
        'в README, отчёт и презентацию автоматически не вставлялась. Следующий шаг — использовать эту сводку для README/отчёта/презентации.',
    ]
    return '\n\n'.join(paragraphs) + '\n'
