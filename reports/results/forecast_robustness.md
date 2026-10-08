# Устойчивость прогнозов и неопределённость различий MAE

Исследовательский вопрос: насколько различия MAE устойчивы по муниципальным образованиям и датам выпуска при короткой истории?

## Протокол и сопоставимость

Gate **PASS**: максимальная разница воспроизведённых метрик 9.09e-13, допуск 1e-8. Для каждой пары 1890 одинаковых случаев с конечным фактом у 63 МО; исходные 1897 запросов и 7 отсутствующих фактов проверены без замены прогнозов.

Знак ΔMAE = MAE baseline − MAE candidate; положительное значение означает меньшую ошибку candidate. MAE macro — среднее MAE каждого МО с равными весами. Validation и holdout разделены по исходному календарному cutoff. F — дополнительная диагностика; A–E — основные сравнения.

Сверены точные ключи МО × origin × target × horizon, факты, split, статусы, history_cutoff и воспроизведённые MAE/R²/counts с сохранёнными CSV, JSON и округлёнными таблицами прежних отчётов. Будущий target используется только для оценки. Проверены сохранённые даты cutoff, но исторические vintages и публикация расходов при L=0 остаются допущением; обучение и его provenance заново не проверялись.

## Неопределённость и заранее заданные правила

Primary: 10 000 origin-cluster draws, seed=42. Каждый выбранный origin переносится со всеми его МО; в каждой выборке заново считается paired equal-MO macro ΔMAE. Secondary: resampling МО со всеми их origins. Доля положительных draws — эмпирическая чувствительность к этому resampling scheme, не p-value и не вероятность преимущества в будущем.

При одной origin primary CI NOT ESTIMABLE; при менее четырёх CI не оценивается. Интервалы при 4–6 origins — только descriptive sensitivity: отдельные origins могут оставаться зависимыми, их независимость или временная exchangeability не доказаны. Municipality bootstrap условен по наблюдаемым датам и не устраняет общие национальные шоки; он не заменяет primary.

Категории фиксированы до расчёта: D при Δ≤0; C при Δ>0 и win-rate МО≤50%; A при Δ>0, строгом большинстве МО и origins и нижней границе 95% origin-интервала>0; остальные положительные случаи с большинством МО — B. Концентрация описывается отдельно, без нового порога. Общая категория pair/horizon — D при неположительном Δ хотя бы в одном split; иначе единая категория split или B при их расхождении. Это описательные категории, не тест значимости.

## Парные результаты

### Validation

| Pair: candidate vs baseline | h | n | ΔMAE, руб. | МО wins | Origins wins | Origin 95% interval | Municipality 95% sensitivity | Category |
|---|---:|---:|---:|---:|---:|---|---|---|
| A: National/Local + LightGBM vs SeasonalNaiveYoY | 1 | 378 | -15.65 | 49.21% | 3/6 | [-198.79; 141.57] | [-100.72; 69.48] | D |
| A: National/Local + LightGBM vs SeasonalNaiveYoY | 3 | 252 | -351.52 | 41.27% | 1/4 | [-838.38; 388.59] | [-532.82; -173.22] | D |
| A: National/Local + LightGBM vs SeasonalNaiveYoY | 6 | 63 | -748.18 | 36.51% | 0/1 | NOT_ESTIMABLE_ONE_ORIGIN | [-1252.94; -272.51] | D |
| B: SeasonalNaiveYoY vs ProphetAuto | 1 | 378 | 375.44 | 80.95% | 4/6 | [-288.97; 1102.02] | [243.70; 523.22] | B |
| B: SeasonalNaiveYoY vs ProphetAuto | 3 | 252 | -1031.38 | 11.11% | 0/4 | [-1897.73; -299.94] | [-1273.05; -763.54] | D |
| B: SeasonalNaiveYoY vs ProphetAuto | 6 | 63 | -2795.90 | 7.94% | 0/1 | NOT_ESTIMABLE_ONE_ORIGIN | [-3281.03; -2304.85] | D |
| C: SeasonalNaiveYoY vs ProphetYearly | 1 | 378 | 1381.78 | 100.00% | 5/6 | [-330.84; 4267.68] | [1259.31; 1513.79] | B |
| C: SeasonalNaiveYoY vs ProphetYearly | 3 | 252 | 740.80 | 87.30% | 2/4 | [-1136.80; 3412.05] | [575.21; 909.24] | B |
| C: SeasonalNaiveYoY vs ProphetYearly | 6 | 63 | 3585.33 | 85.71% | 1/1 | NOT_ESTIMABLE_ONE_ORIGIN | [2832.77; 4344.63] | B |
| D: National/Local + LightGBM vs ProphetAuto | 1 | 378 | 359.80 | 79.37% | 4/6 | [-444.76; 1214.88] | [239.19; 500.01] | B |
| D: National/Local + LightGBM vs ProphetAuto | 3 | 252 | -1382.90 | 4.76% | 0/4 | [-1666.29; -1099.50] | [-1661.71; -1106.91] | D |
| D: National/Local + LightGBM vs ProphetAuto | 6 | 63 | -3544.07 | 4.76% | 0/1 | NOT_ESTIMABLE_ONE_ORIGIN | [-4185.52; -2904.09] | D |
| E: National/Local + LightGBM vs ProphetYearly | 1 | 378 | 1366.13 | 100.00% | 5/6 | [-527.29; 4400.75] | [1228.01; 1502.32] | B |
| E: National/Local + LightGBM vs ProphetYearly | 3 | 252 | 389.28 | 68.25% | 1/4 | [-1780.40; 3800.65] | [135.22; 640.39] | B |
| E: National/Local + LightGBM vs ProphetYearly | 6 | 63 | 2837.15 | 82.54% | 1/1 | NOT_ESTIMABLE_ONE_ORIGIN | [2013.67; 3656.89] | B |
| F: National/Local + LightGBM vs LightGBMDirect | 1 | 378 | 1362.75 | 100.00% | 4/6 | [-247.70; 3323.71] | [1199.63; 1529.48] | B |
| F: National/Local + LightGBM vs LightGBMDirect | 3 | 252 | 60.28 | 53.97% | 2/4 | [-1270.75; 1391.31] | [-47.25; 167.22] | B |
| F: National/Local + LightGBM vs LightGBMDirect | 6 | 63 | 852.73 | 65.08% | 1/1 | NOT_ESTIMABLE_ONE_ORIGIN | [-84.79; 1763.47] | B |

### Holdout

| Pair: candidate vs baseline | h | n | ΔMAE, руб. | МО wins | Origins wins | Origin 95% interval | Municipality 95% sensitivity | Category |
|---|---:|---:|---:|---:|---:|---|---|---|
| A: National/Local + LightGBM vs SeasonalNaiveYoY | 1 | 378 | 121.15 | 55.56% | 5/6 | [39.31; 188.97] | [24.87; 217.08] | A |
| A: National/Local + LightGBM vs SeasonalNaiveYoY | 3 | 378 | 27.68 | 52.38% | 4/6 | [-184.13; 218.36] | [-206.28; 224.47] | B |
| A: National/Local + LightGBM vs SeasonalNaiveYoY | 6 | 378 | 8.79 | 57.14% | 3/6 | [-480.45; 467.33] | [-254.03; 239.46] | B |
| A: National/Local + LightGBM vs SeasonalNaiveYoY | 12 | 63 | 0.00 | 0.00% | 0/1 | DESCRIPTIVE_ONLY_H12 | DESCRIPTIVE_ONLY_H12 | DESCRIPTIVE_ONLY |
| B: SeasonalNaiveYoY vs ProphetAuto | 1 | 378 | 947.86 | 98.41% | 6/6 | [263.94; 1894.12] | [782.86; 1140.71] | A |
| B: SeasonalNaiveYoY vs ProphetAuto | 3 | 378 | 800.91 | 85.71% | 6/6 | [322.07; 1560.15] | [612.65; 1012.29] | A |
| B: SeasonalNaiveYoY vs ProphetAuto | 6 | 378 | 259.77 | 61.90% | 3/6 | [-270.27; 960.69] | [102.79; 420.81] | B |
| B: SeasonalNaiveYoY vs ProphetAuto | 12 | 63 | -1267.97 | 36.51% | 0/1 | DESCRIPTIVE_ONLY_H12 | DESCRIPTIVE_ONLY_H12 | DESCRIPTIVE_ONLY |
| C: SeasonalNaiveYoY vs ProphetYearly | 1 | 378 | 761.61 | 96.83% | 5/6 | [54.28; 1659.12] | [651.47; 871.82] | A |
| C: SeasonalNaiveYoY vs ProphetYearly | 3 | 378 | 439.22 | 82.54% | 4/6 | [-113.76; 1099.38] | [316.95; 558.97] | B |
| C: SeasonalNaiveYoY vs ProphetYearly | 6 | 378 | 31.63 | 60.32% | 2/6 | [-466.47; 621.05] | [-88.76; 144.17] | B |
| C: SeasonalNaiveYoY vs ProphetYearly | 12 | 63 | 4173.86 | 85.71% | 1/1 | DESCRIPTIVE_ONLY_H12 | DESCRIPTIVE_ONLY_H12 | DESCRIPTIVE_ONLY |
| D: National/Local + LightGBM vs ProphetAuto | 1 | 378 | 1069.00 | 100.00% | 6/6 | [341.91; 2015.48] | [919.28; 1226.51] | A |
| D: National/Local + LightGBM vs ProphetAuto | 3 | 378 | 828.59 | 100.00% | 5/6 | [238.20; 1607.37] | [736.86; 924.03] | A |
| D: National/Local + LightGBM vs ProphetAuto | 6 | 378 | 268.56 | 68.25% | 3/6 | [-362.91; 960.46] | [45.64; 475.63] | B |
| D: National/Local + LightGBM vs ProphetAuto | 12 | 63 | -1267.97 | 36.51% | 0/1 | DESCRIPTIVE_ONLY_H12 | DESCRIPTIVE_ONLY_H12 | DESCRIPTIVE_ONLY |
| E: National/Local + LightGBM vs ProphetYearly | 1 | 378 | 882.76 | 96.83% | 6/6 | [147.63; 1810.82] | [767.51; 1000.07] | A |
| E: National/Local + LightGBM vs ProphetYearly | 3 | 378 | 466.90 | 77.78% | 5/6 | [-97.58; 1171.20] | [247.33; 652.73] | B |
| E: National/Local + LightGBM vs ProphetYearly | 6 | 378 | 40.43 | 53.97% | 4/6 | [-810.14; 827.28] | [-234.74; 274.35] | B |
| E: National/Local + LightGBM vs ProphetYearly | 12 | 63 | 4173.86 | 85.71% | 1/1 | DESCRIPTIVE_ONLY_H12 | DESCRIPTIVE_ONLY_H12 | DESCRIPTIVE_ONLY |
| F: National/Local + LightGBM vs LightGBMDirect | 1 | 378 | 203.73 | 87.30% | 2/6 | [-117.63; 610.84] | [153.61; 254.33] | B |
| F: National/Local + LightGBM vs LightGBMDirect | 3 | 378 | 237.09 | 65.08% | 6/6 | [110.77; 380.72] | [121.11; 358.28] | A |
| F: National/Local + LightGBM vs LightGBMDirect | 6 | 378 | 565.99 | 79.37% | 5/6 | [-86.75; 1327.54] | [375.24; 761.79] | B |
| F: National/Local + LightGBM vs LightGBMDirect | 12 | 63 | 0.00 | 0.00% | 0/1 | DESCRIPTIVE_ONLY_H12 | DESCRIPTIVE_ONLY_H12 | DESCRIPTIVE_ONLY |

## Концентрация и чувствительность

Positive improvement mass и negative deterioration mass считаются отдельно по municipality ΔMAE. Top 5/10/ceil(20% всех МО) выбираются по величине массы каждого знака. Доли не делятся на малый или отрицательный net gain. Primary не удаляет МО; дополнительно показаны median Δ и заранее заданное winsorized mean с границами 5/95%.

| Pair | Split | h | Positive mass: top5 / top10 / top20% | Negative mass: top5 / top10 / top20% | Median Δ | Winsorized mean 5% |
|---|---|---:|---|---|---:|---:|
| A | holdout | 1 | 34.19 / 55.98 / 66.33% | 51.63 / 72.10 / 82.03% | 72.19 | 123.19 |
| A | holdout | 3 | 33.42 / 52.44 / 62.32% | 62.68 / 80.50 / 87.00% | 111.48 | 92.04 |
| A | holdout | 6 | 37.67 / 60.81 / 69.34% | 54.19 / 75.17 / 83.54% | 77.36 | 82.68 |
| A | validation | 1 | 42.08 / 69.66 / 78.43% | 31.10 / 52.04 / 63.30% | -17.71 | -12.34 |
| A | validation | 3 | 43.56 / 68.59 / 80.51% | 28.05 / 50.04 / 59.36% | -423.69 | -354.50 |
| A | validation | 6 | 55.37 / 75.69 / 85.00% | 42.62 / 62.58 / 71.18% | -432.90 | -684.72 |
| B | holdout | 1 | 21.42 / 34.12 / 41.11% | 100.00 / 100.00 / 100.00% | 898.61 | 894.60 |
| B | holdout | 3 | 25.77 / 40.02 / 47.46% | 90.87 / 100.00 / 100.00% | 732.90 | 746.88 |
| B | holdout | 6 | 28.29 / 49.61 / 60.35% | 51.88 / 77.39 / 87.93% | 187.91 | 264.02 |
| B | validation | 1 | 35.95 / 52.36 / 60.41% | 73.83 / 98.20 / 100.00% | 279.47 | 355.32 |
| B | validation | 3 | 95.59 / 100.00 / 100.00% | 18.77 / 32.13 / 39.36% | -1208.16 | -1054.86 |
| B | validation | 6 | 100.00 / 100.00 / 100.00% | 17.61 / 30.06 / 36.92% | -3121.57 | -2773.92 |
| C | holdout | 1 | 15.98 / 30.44 / 38.07% | 100.00 / 100.00 / 100.00% | 778.56 | 762.90 |
| C | holdout | 3 | 20.52 / 37.17 / 44.54% | 74.96 / 98.17 / 100.00% | 514.52 | 448.26 |
| C | holdout | 6 | 27.50 / 49.75 / 60.43% | 49.56 / 78.14 / 88.41% | 119.05 | 41.53 |
| C | validation | 1 | 14.15 / 25.97 / 32.60% | — / — / —% | 1297.62 | 1377.84 |
| C | validation | 3 | 21.26 / 35.92 / 43.62% | 72.15 / 100.00 / 100.00% | 790.23 | 716.33 |
| C | validation | 6 | 19.95 / 35.22 / 43.38% | 78.59 / 100.00 / 100.00% | 3461.99 | 3488.89 |
| D | holdout | 1 | 17.69 / 30.29 / 37.18% | — / — / —% | 1008.26 | 1038.52 |
| D | holdout | 3 | 15.44 / 27.22 / 33.90% | — / — / —% | 787.21 | 815.72 |
| D | holdout | 6 | 31.75 / 52.11 / 61.22% | 66.10 / 85.71 / 94.40% | 256.24 | 316.56 |
| D | validation | 1 | 30.93 / 49.12 / 57.02% | 62.70 / 93.29 / 100.00% | 244.51 | 328.56 |
| D | validation | 3 | 100.00 / 100.00 / 100.00% | 21.37 / 34.89 / 42.07% | -1309.79 | -1388.55 |
| D | validation | 6 | 100.00 / 100.00 / 100.00% | 20.09 / 34.43 / 41.10% | -3222.61 | -3619.12 |
| E | holdout | 1 | 16.33 / 29.67 / 36.98% | 100.00 / 100.00 / 100.00% | 837.07 | 876.82 |
| E | holdout | 3 | 25.30 / 39.70 / 47.40% | 90.15 / 97.88 / 99.71% | 556.43 | 511.69 |
| E | holdout | 6 | 35.42 / 57.72 / 67.83% | 59.49 / 78.77 / 86.88% | 90.27 | 105.55 |
| E | validation | 1 | 13.50 / 25.42 / 32.19% | — / — / —% | 1322.24 | 1359.38 |
| E | validation | 3 | 29.07 / 48.93 / 59.06% | 51.34 / 76.23 / 86.57% | 336.09 | 395.70 |
| E | validation | 6 | 23.04 / 41.19 / 50.98% | 66.18 / 98.77 / 100.00% | 2438.85 | 2778.19 |
| F | holdout | 1 | 22.15 / 39.17 / 47.90% | 92.71 / 100.00 / 100.00% | 187.88 | 201.95 |
| F | holdout | 3 | 32.42 / 54.61 / 66.01% | 54.72 / 78.91 / 88.71% | 146.63 | 228.98 |
| F | holdout | 6 | 28.62 / 49.77 / 60.01% | 69.44 / 96.38 / 100.00% | 381.36 | 548.09 |
| F | validation | 1 | 15.07 / 28.16 / 35.14% | — / — / —% | 1239.49 | 1360.28 |
| F | validation | 3 | 37.70 / 58.33 / 66.85% | 45.44 / 69.26 / 77.76% | 79.00 | 67.70 |
| F | validation | 6 | 30.04 / 49.83 / 59.39% | 56.09 / 79.44 / 88.25% | 1168.98 | 809.27 |

## Общая устойчивость по validation и holdout

| Pair | h | Category |
|---|---:|---|
| A | 1 | D |
| A | 3 | D |
| A | 6 | D |
| B | 1 | B |
| B | 3 | D |
| B | 6 | D |
| C | 1 | B |
| C | 3 | B |
| C | 6 | B |
| D | 1 | B |
| D | 3 | D |
| D | 6 | D |
| E | 1 | B |
| E | 3 | B |
| E | 6 | B |
| F | 1 | B |
| F | 3 | B |
| F | 6 | B |

## Ответы на исследовательские вопросы

**A: National/Local + LightGBM vs SeasonalNaiveYoY.**

validation: h1: Δ=-15.65, МО win-rate=49.21%, origins=3/6, D; h3: Δ=-351.52, МО win-rate=41.27%, origins=1/4, D; h6: Δ=-748.18, МО win-rate=36.51%, origins=0/1, D.

holdout: h1: Δ=121.15, МО win-rate=55.56%, origins=5/6, A; h3: Δ=27.68, МО win-rate=52.38%, origins=4/6, B; h6: Δ=8.79, МО win-rate=57.14%, origins=3/6, B.

**B: SeasonalNaiveYoY vs ProphetAuto.**

validation: h1: Δ=375.44, МО win-rate=80.95%, origins=4/6, B; h3: Δ=-1031.38, МО win-rate=11.11%, origins=0/4, D; h6: Δ=-2795.90, МО win-rate=7.94%, origins=0/1, D.

holdout: h1: Δ=947.86, МО win-rate=98.41%, origins=6/6, A; h3: Δ=800.91, МО win-rate=85.71%, origins=6/6, A; h6: Δ=259.77, МО win-rate=61.90%, origins=3/6, B.

**C: SeasonalNaiveYoY vs ProphetYearly.**

validation: h1: Δ=1381.78, МО win-rate=100.00%, origins=5/6, B; h3: Δ=740.80, МО win-rate=87.30%, origins=2/4, B; h6: Δ=3585.33, МО win-rate=85.71%, origins=1/1, B.

holdout: h1: Δ=761.61, МО win-rate=96.83%, origins=5/6, A; h3: Δ=439.22, МО win-rate=82.54%, origins=4/6, B; h6: Δ=31.63, МО win-rate=60.32%, origins=2/6, B.

**D: National/Local + LightGBM vs ProphetAuto.**

validation: h1: Δ=359.80, МО win-rate=79.37%, origins=4/6, B; h3: Δ=-1382.90, МО win-rate=4.76%, origins=0/4, D; h6: Δ=-3544.07, МО win-rate=4.76%, origins=0/1, D.

holdout: h1: Δ=1069.00, МО win-rate=100.00%, origins=6/6, A; h3: Δ=828.59, МО win-rate=100.00%, origins=5/6, A; h6: Δ=268.56, МО win-rate=68.25%, origins=3/6, B.

**E: National/Local + LightGBM vs ProphetYearly.**

validation: h1: Δ=1366.13, МО win-rate=100.00%, origins=5/6, B; h3: Δ=389.28, МО win-rate=68.25%, origins=1/4, B; h6: Δ=2837.15, МО win-rate=82.54%, origins=1/1, B.

holdout: h1: Δ=882.76, МО win-rate=96.83%, origins=6/6, A; h3: Δ=466.90, МО win-rate=77.78%, origins=5/6, B; h6: Δ=40.43, МО win-rate=53.97%, origins=4/6, B.

Концентрация положительной массы top5 по primary split/h с ненулевой положительной массой находится между 13.50% и 100.00%; отрицательная масса учитывается отдельно.

**Outlier-влияние.** NL+LightGBM vs SNY, holdout: h1: top5=34.19% положительной массы, positive/negative mass=12997.23/5364.98, вклад top5 положительных МО/net gain=0.58; h3: top5=33.42% положительной массы, positive/negative mass=18575.40/16831.80, вклад top5 положительных МО/net gain=3.56; h6: top5=37.67% положительной массы, positive/negative mass=21201.45/20647.58, вклад top5 положительных МО/net gain=14.42. Выигрыш присутствует у большинства МО, но small net gain h3/h6 получается из почти компенсирующих друг друга улучшений и ухудшений. Положительная масса не ограничивается пятью МО; при этом их вклад может существенно превышать итоговый net gain. Отсутствие влияния отдельных территорий не установлено; primary результат сохранён без удаления МО.

Across-split категории A-сравнения: [(1, 'D'), (3, 'D'), (6, 'D')]. Для четырёх Prophet-сравнений положительное aggregate Δ наблюдается в 20/24 фиксированных split/h группах. Это сравнение robustness признаков, не доказательство универсального ranking или будущего превосходства.

**Какой вывод устойчивее?** Seasonal/panel представление относительно ProphetYearly сохраняет положительное aggregate Δ в 12/12 группах обоих split. Однако temporal интервалы остаются широкими, а сравнение с ProphetAuto на validation h3/h6 даёт положительный Δ только в 0/4 группах. Данные лучше поддерживают ограниченный вывод о сильном seasonal/panel benchmark относительно проверенного ProphetYearly, чем устойчивый ranking NL+LightGBM против SNY. Универсальное превосходство над обоими Prophet по всем периодам и горизонтам не подтверждено.

## Применимость формальных тестов и ограничения

Diebold–Mariano: **NOT RELIABLE / NOT ESTIMATED**. Для каждого split/h доступны лишь 1–6 target periods, multi-step horizons разделяют историю и могут иметь зависимые loss differentials. Число МО не увеличивает число независимых временных точек. Two-way bootstrap не выполнялся как необязательное усложнение.

История цели — 24 месяца. Holdout уже просмотрен; анализ post-hoc, не новый blind test. Общие national shocks создают зависимость между МО; origin-bootstrap сохраняет эту зависимость внутри даты, но не устраняет зависимость соседних dates. h12 имеет одну origin, только descriptive, без inference; learned references используют прежний SeasonalNaive fallback. Bootstrap описывает эмпирический sample, а не независимый будущий период; пороги, seed, единица bootstrap, варианты Prophet, horizons и sample не подбирались по результату.

Новых model fits: **0**; runtime полного анализа: **5.711 s**; seed=42.

## Сохранённые артефакты

Все публичные CSV содержат только сводные показатели по pair/split/h или origin, без y_true/y_pred и исходных IDs МО. Локальный `forecast_robustness/municipality_deltas.csv` содержит per-MO audit и исключён из Git; его полная копия находится также в ignored output_dir. Best/worst ссылки в summary — локальные порядковые обозначения, не исходные IDs.

[applicability_checks.csv](forecast_robustness/applicability_checks.csv) · [bootstrap_municipality.csv](forecast_robustness/bootstrap_municipality.csv) · [bootstrap_origin.csv](forecast_robustness/bootstrap_origin.csv) · [concentration.csv](forecast_robustness/concentration.csv) · [municipality_summary.csv](forecast_robustness/municipality_summary.csv) · [origin_deltas.csv](forecast_robustness/origin_deltas.csv) · [origin_summary.csv](forecast_robustness/origin_summary.csv) · [pairwise_metrics.csv](forecast_robustness/pairwise_metrics.csv)

[Run manifest](forecast_robustness/run_manifest.json) · [Конфигурация](../../configs/forecast_robustness.yaml)
