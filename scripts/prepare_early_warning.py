"""E07a preparation only: saved forecasts, causal features and weak future labels."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sberforecast.data import load_data, sha256_file
from sberforecast.news_audit import assert_feature_keys, document_audit, feature_coverage, origin_coverage
from sberforecast.news_features import FEATURE_COLUMNS, MISSING_COLUMNS, build_news_features, feature_dictionary as news_dictionary
from sberforecast.news_normalize import canonical_events, normalize_documents


def path(value: str) -> Path:
    result = (ROOT / value).resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError("All E07a paths must remain inside this project")
    return result


def save_json(filename: Path, value: object) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str)+"\n", encoding="utf-8")


def read_csv(filename: Path) -> pd.DataFrame:
    return pd.read_csv(filename, dtype={"municipality_id": str, "series_id": str, "region_id": str})


def write_table(output: Path, name: str, table: pd.DataFrame) -> None:
    table.to_csv(output / (name + ".csv.gz"), index=False, compression="gzip")


def manifest(output: Path, config: dict, inputs: list[Path], started: str, clock: float) -> None:
    code = [Path(__file__), ROOT / "configs/news_events_v3.yaml"]
    code += sorted((ROOT / "src/sberforecast").glob("early_warning*.py"))
    code += sorted((ROOT / "src/sberforecast").glob("news_*.py"))
    code += [ROOT / "src/sberforecast/online_detection.py", ROOT / "src/sberforecast/macro_forecast_features.py"]
    save_json(output / "run_manifest.json", dict(experiment=output.name, seed=config["seed"],
        started_at=started, completed_at=datetime.now(timezone.utc).isoformat(), runtime_seconds=time.perf_counter()-clock,
        command=subprocess.list2cmdline([sys.executable]+sys.argv), classifier_fit=False, forecast_models_refitted=False,
        synthetic_smoke=False, python=sys.version,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--short"],cwd=ROOT,text=True),
        versions={d.metadata["Name"]:d.version for d in importlib.metadata.distributions()},
        inputs={str(p.relative_to(ROOT)):sha256_file(p) for p in sorted(set(inputs))},
        code={str(p.relative_to(ROOT)):sha256_file(p) for p in code},
        artifacts={p.name:dict(sha256=sha256_file(p),bytes=p.stat().st_size) for p in sorted(output.iterdir()) if p.is_file()},
        assumptions=[config["data"]["availability_assumption"], config["news"]["archive_trust"],
            "E04 parameters selected previously, causal retrospective replay is not proof of historical parameter selection",
            "Already inspected history; weak labels are not independent ground truth; national rows are dependent"] ))


def combine_decision_snapshots(old: pd.DataFrame, added: pd.DataFrame, decision_map: dict) -> pd.DataFrame:
    """Retain every version, align mixed dates and summarize first availability."""
    old,added = old.copy(),added.copy()
    for table in (old, added):
        selected = table.document_id.isin(decision_map)
        table.loc[selected, "canonical_event_id"] = table.loc[selected, "document_id"].map(decision_map)
        for column in ("available_at","published_at","retrieved_at","updated_at","event_date"):
            table[column] = pd.to_datetime(table[column],utc=True,format="mixed")
    added["geography_level"] = "national"
    added["region"] = "Россия"
    added["region_id"] = ""
    added["municipality_id"] = ""
    added["topic"] = "macro_policy"
    added["event_type"] = "policy_decision"
    added["direction"] = "unknown"
    added["classification_rule"] = "E07a_official_key_rate_decision_core_macro_policy_policy_decision_direction_unknown_no_effect_inference"
    added["geography_rule"] = "E07a_explicit_official_Russian_national_rate_decision_core"
    normalized = pd.concat([old, added], ignore_index=True)
    normalized = normalized.sort_values(["available_at","published_at","document_id","snapshot_id"],kind="stable").reset_index(drop=True)
    normalized["is_canonical"] = ~normalized.canonical_event_id.duplicated()
    normalized["duplicate_count"] = normalized.canonical_event_id.map(normalized.groupby("canonical_event_id").document_id.nunique())
    return normalized


def build_v3(config: dict, samples: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Admission changes only dated official decision cores, preserving old snapshots."""
    started,clock = datetime.now(timezone.utc).isoformat(),time.perf_counter()
    audit = path(config["news"]["audit_output_dir"])
    core_path = audit / "core_trusted_records.json"
    cores = json.loads(core_path.read_text(encoding="utf-8"))
    old_dir = path(config["reference"]["e06b_dir"])
    old = read_csv(old_dir / "documents.csv.gz")
    old_features = read_csv(old_dir / "news_features.csv.gz")
    if not cores:
        return old_features, {"created_v3": False, "reason": "No newly admissible decision core"}
    output = path(config["news"]["new_output_dir"])
    if output.exists():
        raise FileExistsError(output)
    dictionary = read_csv(old_dir / "geography_dictionary.csv.gz")
    added = normalize_documents(cores, dictionary)
    mapping = read_csv(audit / "document_summary.csv")
    event_column = next(c for c in ("canonical_event_id_v3", "canonical_event_id", "decision_event_id") if c in mapping)
    decision_map = mapping.set_index("document_id")[event_column].dropna().to_dict()
    # Only explicit key-rate mappings replace E06b's title/day canonicalization.
    decision_map = {key: value for key, value in decision_map.items() if str(value).startswith("cbr_key_rate_decision_")}
    core_map = {record["document_id"]: record["canonical_event_id"] for record in cores}
    decision_map.update(core_map)
    normalized = combine_decision_snapshots(old,added,decision_map)
    features = build_news_features(normalized, samples, history_start="2023-01-01")
    assert_feature_keys(features, samples)
    features["source_archive_complete"] = False
    known_time = pd.to_datetime(normalized.available_at, utc=True,format="mixed")
    features["historical_version_evidence_available"] = [bool((normalized.availability_status.str.startswith("confirmed_") &
        known_time.le(pd.Timestamp(origin).tz_localize("Europe/Moscow").tz_convert("UTC"))).any()) for origin in features.forecast_origin]
    output.mkdir(parents=True)
    for name, table in (("documents", normalized), ("events", canonical_events(normalized)), ("forecast_cases", samples),
                        ("news_features", features), ("origin_coverage", origin_coverage(features))):
        write_table(output, name, table)
    news_dictionary().to_csv(output / "feature_dictionary.csv", index=False)
    feature_coverage(features, FEATURE_COLUMNS).to_csv(output / "feature_coverage.csv", index=False)
    dictionary.to_csv(output / "geography_dictionary.csv.gz", index=False, compression="gzip")
    stats = document_audit(normalized, "2023-01-01", "2024-12-31")
    stats.update(created_v3=True, feature_rows=len(features), admitted_core_records=len(cores),
        historical_documents_at_last_origin=int(normalized.loc[known_time.le(pd.Timestamp(samples.forecast_origin.max()).tz_localize("Europe/Moscow").tz_convert("UTC")),"document_id"].nunique()),
        historical_snapshots_at_last_origin=int(known_time.le(pd.Timestamp(samples.forecast_origin.max()).tz_localize("Europe/Moscow").tz_convert("UTC")).sum()),
        unique_temporal_vectors=len(features[list(FEATURE_COLUMNS)+list(MISSING_COLUMNS)].drop_duplicates()),
        previous_unique_temporal_vectors=len(old_features[list(FEATURE_COLUMNS)+list(MISSING_COLUMNS)].drop_duplicates()),
        nonzero_30d_rows=int(features.news_count_30d.gt(0).sum()),
        nonzero_30d_origins=features.loc[features.news_count_30d.gt(0),"forecast_origin"].nunique(),
        regional_nonzero_rows=int(features.regional_news_count_30d.gt(0).sum()))
    save_json(output / "coverage_audit.json", stats)
    (output / "config_resolved.yaml").write_text((ROOT / "configs/news_events_v3.yaml").read_text(encoding="utf-8"),encoding="utf-8")
    manifest(output, config, [core_path, audit / "document_summary.csv", old_dir / "documents.csv.gz",
             old_dir / "forecast_cases.csv.gz", old_dir / "news_features.csv.gz", old_dir / "geography_dictionary.csv.gz", ROOT / "configs/news_events_v3.yaml"],started,clock)
    return features, stats


def event_ids(group: pd.DataFrame) -> set[str]:
    values = group.loc[group.label.eq(1),"positive_event_ids"]
    return {identifier for value in values for identifier in
        (json.loads(value) if str(value).startswith("[") else str(value).split("|")) if identifier}


def training_availability(cases: pd.DataFrame, features: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    from sberforecast.early_warning_labels import admit_training_labels
    records = []
    for origin in sorted(cases.forecast_origin.unique()):
        for k in sorted(cases.k.unique()):
            current = cases.loc[(cases.forecast_origin == origin) & (cases.k == k)]
            train = admit_training_labels(cases.loc[cases.k.eq(k)], origin)
            positive = train.loc[train.label.eq(1)]
            ids = event_ids(train)
            f = features.loc[features.forecast_origin.eq(origin)]
            records.append(dict(forecast_origin=origin,k=int(k),cases=len(current),
                features_present=len(f),news_nonzero_30d_cases=int(f.news_count_30d.gt(0).sum()),
                news_unique_temporal_vectors=len(f[list(FEATURE_COLUMNS)+list(MISSING_COLUMNS)].drop_duplicates()),
                known_active_cases=int(current.known_active_at_origin.sum()),
                retrospective_inside_cases=int(current.retrospective_inside_regime_at_origin.sum()),
                training_examples=len(train),training_positive=len(positive),training_negative=int(train.label.eq(0).sum()),
                training_origin_dates=train.forecast_origin.nunique(),training_event_ids=len(ids),
                training_event_onset_dates=events.loc[events.event_id.isin(ids),"onset_period"].nunique(),
                fully_known_current=int(current.fully_known.sum()),current_positive=int(current.label.eq(1).sum()),
                current_negative=int(current.label.eq(0).sum()),at_risk_fully_known=int((current.fully_known & current.at_risk).sum()),
                required_confirmation_end_period=current.required_confirmation_end_period.iloc[0],
                current_label_known_dates=" | ".join(sorted(set(current.loc[current.fully_known,"label_known_at"]))),
                left_insufficient_cases=int(current.left_insufficient.sum()),right_censored_cases=int(current.right_censored.sum()),
                residual_current_available_cases=int(f.residual_current_value.notna().sum()),
                expense_current_available_cases=int(f.expense_current_value.notna().sum()),
                online_cusum_ready_cases=int(f.online_cusum_ready.eq(1).sum()),
                online_ewma_ready_cases=int(f.online_ewma_ready.eq(1).sum()),
                online_bocpd_ready_cases=int(f.online_bocpd_ready.eq(1).sum())))
    return pd.DataFrame(records)


def temporal_split(cases: pd.DataFrame, config: dict, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sberforecast.early_warning_labels import admit_training_labels
    spec = config["early_warning"]["split"]
    cutoff = str(pd.Period(spec["train_information_cutoff"],freq="M").end_time.normalize().date())
    train = admit_training_labels(cases, cutoff).copy()
    train["split"] = "train"
    origins = pd.to_datetime(cases.forecast_origin).dt.to_period("M")
    test = cases.loc[origins.between(pd.Period(spec["test_first_origin"],freq="M"),pd.Period(spec["test_last_origin"],freq="M"))].copy()
    test["split"] = np.where(test.fully_known & test.at_risk,"test_evaluable","test_unavailable_or_active")
    selected = pd.concat([train,test],ignore_index=True)
    rows = []
    for k in config["early_warning"]["horizons_months"]:
        for kind in ("train","test_evaluable","test_unavailable_or_active"):
            group = selected.loc[selected.k.eq(k) & selected.split.eq(kind)]
            ids = event_ids(group)
            rows.append(dict(k=k,split=kind,cases=len(group),positive=int(group.label.eq(1).sum()),negative=int(group.label.eq(0).sum()),
                             origin_dates=group.forecast_origin.nunique(),event_ids=len(ids),
                             event_onset_dates=events.loc[events.event_id.isin(ids),"onset_period"].nunique()))
    return selected,pd.DataFrame(rows)


def manual_queue(labels: dict[str,pd.DataFrame], residuals: pd.DataFrame) -> pd.DataFrame:
    candidates = labels["candidates"]
    cases = labels["cases"]
    rows = []
    # Explicit deterministic selection includes rejected/incomplete candidates.
    eligible = candidates.loc[candidates.status.eq("candidate")] if "candidate" in set(candidates.status) else candidates.loc[candidates.get("is_candidate", pd.Series(False,index=candidates.index)).eq(True)]
    if eligible.empty:
        eligible = candidates.loc[candidates.status.astype(str).str.contains("candidate|confirmed",regex=True)]
    for direction, group in eligible.groupby("candidate_direction",dropna=False):
        group = group.sort_values(["onset_period","series_id"])
        for name, row in (("earliest_candidate",group.iloc[0]),("latest_candidate",group.iloc[-1])):
            rows.append(dict(selection=name,series_id=str(row.series_id),onset_period=row.onset_period,
                confirmation_period=row.confirmation_period,forecast_origin="",k="",weak_label="candidate",event_id=row.get("accepted_event_id","")))
    controls = cases.loc[cases.fully_known & cases.label.eq(0)].sort_values(["forecast_origin","municipality_id","k"])
    for row in controls.drop_duplicates("municipality_id").head(3).itertuples():
        rows.append(dict(selection="finite_control",series_id=str(row.municipality_id),onset_period="",confirmation_period="",
            forecast_origin=row.forecast_origin,k=row.k,weak_label=0,event_id=""))
    missing = cases.loc[~cases.fully_known].sort_values(["forecast_origin","municipality_id","k"])
    chosen = missing.loc[missing.municipality_id.eq("1471")].head(1)
    if chosen.empty:
        chosen = missing.head(1)
    for row in chosen.itertuples():
        rows.append(dict(selection="missing_or_censored",series_id=str(row.municipality_id),onset_period="",confirmation_period="",
            forecast_origin=row.forecast_origin,k=row.k,weak_label="unknown",event_id=""))
    result = pd.DataFrame(rows).drop_duplicates()
    result["annotation_status"] = "pending_human_review"
    result["independent_annotation"] = False
    for column in ("human_event_supported","human_onset_period","human_reason","reviewer","reviewed_at"):
        result[column] = ""
    return result


def markdown_table(table: pd.DataFrame) -> str:
    def cell(value):
        return str(value).replace("|",";").replace("\n"," ") if pd.notna(value) else "—"
    return "\n".join(["| " + " | ".join(table.columns) + " |", "|" + "---|"*len(table.columns)] +
        ["| " + " | ".join(cell(value) for value in row) + " |" for row in table.itertuples(index=False,name=None)])


def build_report(config: dict) -> None:
    """All figures below come from saved E07a artifacts, never invented runs."""
    output = path(config["output_dir"])
    summary = json.loads((output / "feasibility_summary.json").read_text(encoding="utf-8"))
    audit_dir = path(config["news"]["audit_output_dir"])
    source = json.loads((audit_dir / "audit_summary.json").read_text(encoding="utf-8"))
    old = json.loads((path(config["reference"]["e06b_dir"]) / "coverage_audit.json").read_text(encoding="utf-8"))
    old_features = read_csv(path(config["reference"]["e06b_dir"]) / "news_features.csv.gz")
    news = summary["news"]
    features = read_csv(output / "features.csv.gz")
    candidates = read_csv(output / "weak_candidates.csv.gz")
    events = read_csv(output / "weak_events.csv.gz")
    availability = read_csv(output / "training_availability.csv.gz")
    split = read_csv(output / "temporal_split_summary.csv.gz")
    train_k1 = split.loc[split.k.eq(1) & split.split.eq("train")].iloc[0]
    test_k1 = split.loc[split.k.eq(1) & split.split.eq("test_evaluable")].iloc[0]
    feature_audit = read_csv(output / "news_feature_audit_full_history.csv")
    queue = read_csv(output / "manual_review_queue.csv.gz")
    definitions = read_csv(output / "history_feature_dictionary.csv")
    decision = read_csv(audit_dir / "extracted_decisions.csv")
    origin = features.groupby("forecast_origin",sort=True).agg(cases=("municipality_id","size"),
        news_30d=("news_count_30d","first"),news_90d=("news_count_90d","first"),
        regional_30d=("regional_news_count_30d","first"),national_30d=("national_news_count_30d","first"))
    compare = pd.DataFrame([
        {"Показатель":"URL-документы","v2":old["documents"],"v3":news["documents"]},
        {"Показатель":"Снимки/извлечённые версии","v2":old["snapshots"],"v3":news["snapshots"]},
        {"Показатель":"Canonical группы событий","v2":old["canonical_events"],"v3":news["canonical_events"]},
        {"Показатель":"Исторически доступные URL к последней O","v2":old["historical_documents_at_last_origin"],"v3":news["historical_documents_at_last_origin"]},
        {"Показатель":"Уникальные временные news-векторы","v2":news["previous_unique_temporal_vectors"],"v3":news["unique_temporal_vectors"]},
        {"Показатель":"Строки с news_30d>0","v2":int(old_features.news_count_30d.gt(0).sum()),"v3":news["nonzero_30d_rows"]},
        {"Показатель":"Origin с региональными news_30d>0","v2":old_features.loc[old_features.regional_news_count_30d.gt(0),"forecast_origin"].nunique(),
         "v3":features.loc[features.regional_news_count_30d.gt(0),"forecast_origin"].nunique()},
    ])
    useful = feature_audit.loc[~feature_audit.all_missing & ~feature_audit.exact_constant_including_missing & feature_audit.exact_duplicate_of.isna(),"feature"].tolist()
    known = availability[["forecast_origin","k","training_examples","training_positive","training_negative","training_origin_dates",
                          "training_event_ids","training_event_onset_dates","fully_known_current","at_risk_fully_known",
                          "known_active_cases","retrospective_inside_cases","required_confirmation_end_period"]]
    test_path = ROOT / "outputs/e07a_checks/final_tests.json"
    tests = json.loads(test_path.read_text(encoding="utf-8")) if test_path.exists() else {"status":"финальный полный pytest ещё не выполнен"}
    verification_path = ROOT / "outputs/e07a_checks/verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8")) if verification_path.exists() else {"status":"ожидается проверка новых артефактов"}
    manifest_data = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    network = pd.DataFrame([{key:source[key] for key in ("new_requests_total_ledger","new_requests_this_command","new_bytes_total_ledger","stopped_after_blocks")}])
    confirmed = candidates.loc[candidates.is_candidate.eq(True)]
    text = f"""# E07a — данные, слабые метки и возможность временной проверки

Код подготовки выполнен; реальные таблицы сохранены в `{config['output_dir']}/`,
обоснованная отдельная политика — в `{config['news']['new_output_dir']}/`.
Классификатор не обучался; метрики раннего предупреждения не получены.
Старые прогнозы, MAE, модели, новости v1/v2 и отчёты не менялись.
Дата: {config['experiment_date']}; HEAD при подготовке: `{manifest_data['git_commit']}`;
seed={config['seed']}. Незакоммиченные изменения записаны в manifest.

## 1. Почему E06b допускал четыре PDF

E06b принял четыре оригинальных официальных PDF ЦБ по явному доверию датированному
архиву E05a. Независимых SHA-снимков 2024 года нет и для этих PDF.
176 современных HTML/index-снимков оставались retrieval-only: дата публикации
не подтверждала исходную версию всего скачанного текста. Отсутствие старого SHA
было основанием строгой политики, а не найденным доказательством правки каждого HTML.
Сам формат не служит критерием допуска.

В E07a для каждого из {source['old_snapshots']} старых снимков и
{source['old_document_ids']} URL-документов сохранены publication, content update,
catalogue update, retrieval, availability, причины и evidence:
`{config['news']['audit_output_dir']}/snapshot_audit.csv` и `document_summary.csv`.
Дата обновления каталога не является датой обновления статьи; событие,
публикация и скачивание разделены. Lenta и непроверенный полный текст HTML
по-прежнему не получают исторический допуск. Поздние article:modified_time
отклоняют допуск неотделимого ядра; поздний каталог сам по себе не переписывает ядро.

## 2. Ограниченное дополнение и новая версия

Заново проверен только заранее заданный класс: официальные решения ЦБ о ставке
2023–2024, включая unchanged. Нет отбора по расходам, остаткам или детекторам.
Из {source['original_release_candidates']} архивных ссылок допущено
{source['admitted_decision_cores']} ядер решений. Используется первая фраза решения,
совпадающая с заголовком, ставкой/действием, датой архивной строки, URL и точным
footer публикации. Дальнейший современный комментарий не приписывается старой дате.
Это явное **допущение доверия оригинальному официальному архиву**, сопоставимое
с PDF, а не независимое подтверждение исторической неизменности текста.
16 ядер переиспользованы из прежнего cache; заново получено только решение
20.12.2024, которое позже последней O=2024-11 и не увеличивает её признаки.

Календарь [ЦБ](https://www.cbr.ru/dkp/cal_mp/) и
[архив решений](https://www.cbr.ru/dkp/mp_dec/decision_key_rate/) — источники ссылок.
`extracted_decisions.csv/json` сохраняет publication_date, decision_date,
effective_date, действие, ставку, подтверждающую фразу и каждую причину допуска/отказа.
Если effective_date не указан в ядре, поле пусто; правило следующего понедельника
не применяется. Последующие резюме заседаний не подставляются в день решения.

Новый cache и ledger изолированы в `{config['news']['raw_dir']}/`.
Лимиты:100 новых запросов, фактически последовательный сбор, timeout20s,
не более одного повтора, остановка после двух блокировок, 2MiB/file и20MiB/сбор.
Фактический ledger (включая неудачные попытки):

{markdown_table(network)}

Отказы: {json.dumps(source['refused_decisions'],ensure_ascii=False)}.
`cbr_key_rate_decision_YYYY_MM_DD` объединяет релиз, PDF и версии одного решения.
Четыре группы пересекаются с E05c; их PDF и макропрогноз не являются независимыми
новыми источниками. `{source['e05c_overlapping_decisions']}` допущенных core имеют
это пересечение. Поздние версии не меняют первую известную версию.

## 3. Фактическое news-покрытие

{markdown_table(compare)}

{markdown_table(origin.reset_index())}

Исходные 64 МО ×12 O=768 ключей сохранены, включая1471. Во всех доступных news
векторах информация национальная и одинакова для64МО на одной O.
Исторических региональных и муниципальных событий по-прежнему0.
`source_archive_complete=false`; ноль означает отсутствие допущенного документа
в этом частичном корпусе, а не отсутствие события или полную интенсивность региона.
Отдельные snapshots не считаются независимыми событиями.

## 4. Точное слабое событие

Протокол записан в `configs/early_warning_feasibility.yaml` **до реального подсчёта**;
SHA протокола и время фиксации сохранены в `outputs/e07a_checks/preservation_before.json`.
Порог/окно после подсчёта не подбирались.

Сигнал: `e[t]=y_true[t]−y_pred[t]` из уже сохранённого причинного h=1
SeasonalNaiveYoY E04/E01. Сезонность учитывает тот же календарный месяц прошлого
года с сохранённой причинной поправкой YoY по предыдущим трём месяцам
(границы0.5…2); старые прогнозы не пересчитываются и не обучаются.
Цель — **устойчивый сдвиг ошибки этого прогноза**, не непосредственно уровня
самих расходов и не подтверждённый экономический шок.

Для кандидата с началом T берутся четыре календарных остатка T−4…T−1.
`c=median(e_past)`;
`s=max(1.4826×MAD(e_past),0.03×median(abs(y_pred_past)),1 рубль)`.
Все три `z=(e[T:T+2]−c)/s` должны быть одновременно≥3 либо одновременно≤−3.
Все семь входных месяцев должны быть конечными; пропуски не заполняются.
Начало=T, подтверждение=T+2 или более поздняя фактическая доступность,
выпуск предупреждения=O, когда T ещё находится в `(O,O+k]`.

Одно направление в соседних кандидатах с разрывом≤2месяцев или внутри ещё
не восстановившегося режима — продолжение прежнего события. Восстановление:
два последовательных signed z<1.5 по замороженным c/s события. Пропуск прерывает
подтверждение восстановления и помечает неопределённость. Подтверждённый
противоположный кандидат создаёт новый onset; возврат ошибки к прежнему уровню
может стать противоположным кандидатом относительно нового baseline — нужна
ручная интерпретация. Активное изменение известно причинно только после
подтверждения, а полносэмпловое нахождение внутри режима — отдельная диагностика.

Сохранено {len(candidates)} строк календарных кандидатов, {len(confirmed)}
прошли трёхмесячный порог, после объединения {len(events)} слабых новых событий
на {events.onset_period.nunique()} уникальных onset-датах:

{markdown_table(events.groupby(['onset_period','confirmation_period','direction'],dropna=False).size().rename('events').reset_index())}

## 5. Доступность обучения и оценки

k=1 — основной горизонт предупреждения, k=3 — дополнительный; это отдельные
горизонты от прогноза расходов1/3/6/12. Для любого O нужна полная область будущего
`O+1…O+k+2`, а также baseline-прошлое. Даже найденная положительная метка
допускается только после полного окна O+k+2, так же как отрицательная.
При задержке старых остатков известность также ждёт доступности всего конечного
прошлого префикса, участвующего в объединении событий; при сохранённом L0
это не изменяет дату O+k+2.
Правый край — unknown/censored, а не0; неполное прошлое также unknown.
Расходы L=0 и их historical vintages остаются непроверенным допущением E01.

Полная таблица ниже показывает накопленное обучение, **известное к каждой O**,
число МО-событий и число общих onset-дат отдельно; current fully_known —
ретроспективно проверяемые выпуски, не доступные для обучения на той же O.

{markdown_table(known)}

Фиксированный временной раздел: train information cutoff=2024-07,
проверка O=2024-08…2024-11; random split отсутствует.
Обучение и оценка исключают только уже подтверждённый активный режим на собственной O.

{markdown_table(split)}

64 строки на O не дают64независимых news/macro наблюдений. Общая дата одного
национального решения и одновременно меняющиеся МО могут быть зависимы.
Просмотренная история E01–E06 не объявляется новым слепым тестом.

## 6. Признаки и ручная проверка

`features.csv.gz` содержит прошлые расходы/current/lag/MoM/YoY/strict mean3,
ранее выпущенные ошибки, префиксные CUSUM/EWMA/BOCPD score/alarm/readiness/age,
допустимые национальные годовые макропрогнозы и news candidates.
Состояния заново проиграны на собственном доступном префиксе с сохранёнными E04
параметрами; дата исторического выбора этих параметров не доказана.
Warmup использует только уже доступные остатки; будущая калибровка не допускается.
Для каждого history/macro поля сохранён max_available_at и происхождение,
для каждого O — `feature_availability_by_origin.csv.gz`.
Макро относятся к календарному году O и не превращаются в факты месячных расходов.
PELT/BinSeg сохранены только в `offline_breakpoint_diagnostics.csv.gz`, не входят
в features или истинные метки. Критерий меток не совпадает с тревогами CUSUM/BOCPD.

Аудит всех52news столбцов: полностью отсутствуют
{int(feature_audit.all_missing.sum())}, строго постоянны включая missing
{int(feature_audit.exact_constant_including_missing.sum())}, имеют точный предыдущий
дубликат {int(feature_audit.exact_duplicate_of.notna().sum())}.
Непостоянные недублирующиеся поля диагностически: {', '.join(useful)}.
Это описание всей истории, **никакие столбцы автоматически не выбраны/удалены**.
Будущий отбор выполняется только по train; отдельные52строки train-аудита k1/k3
сохранены даже при пустой обучающей части.

Очередь `{len(queue)}` случаев: earliest/latest кандидаты каждого направления,
finite controls и missing/censored пример; соответствующие остатки сохранены.
`annotation_status=pending_human_review`, независимых человеческих меток0.
Проверка другим ИИ не считается независимой ручной разметкой.
Определения полей: `field_definitions.json`, history/news dictionaries
({len(definitions)} history/macro определений).

## 7. Решение о E07b, проверки и ограничения

Минимальная будущая последовательность заранее сохранена: constant training risk,
простая модель на истории, та же модель плюс train-selected admitted news.
Настройки/порог предупреждения выбираются только по прошлому; сравнение — на одинаковых
at-risk cases, с явными FP/пропусками и временным упреждением.
В k=1 train есть {train_k1.cases} случая на {train_k1.origin_dates} O:
{train_k1.positive} positive и {train_k1.negative} negative,
{train_k1.event_ids} событие на {train_k1.event_onset_dates} onset-дате.
Constant training risk={train_k1.positive}/{train_k1.cases} можно рассчитать описательно;
бинарную модель технически можно подогнать, но одной положительной метки и
одной даты недостаточно для устойчивого выбора/валидации модели.
В at-risk test остаётся {test_k1.cases} случаев на {test_k1.origin_dates} O,
positive={test_k1.positive}, negative={test_k1.negative}: возможны только
описательные FP rate, specificity и вероятностная ошибка на отрицательных случаях;
recall, пропуски событий, упреждение и прирост предупреждения с news не проверяются.
Положительная test-метка в общем наборе относится к уже активному на O режиму
и исключена заранее заданным causal risk filter, а не скрыта.
k=3 в этом временном разделе вообще не имеет train/test пары.
Полноценная проверка вклада news и доказательство раннего предупреждения
на этой истории невозможны. Возможны аудит слабых меток и ручное рассмотрение
сохранённых событий/контролей; затем требуется более длинная история и проверенная
доступность источников, после отдельного решения о расширении.

Проверка новых артефактов: `{json.dumps(verification,ensure_ascii=False)}`.
Финальный полный pytest (один прогон): `{json.dumps(tests,ensure_ascii=False)}`.
Целевые тесты и команды — `outputs/e07a_checks/`; manifests содержат command,
seed, package versions, HEAD, dirty status, SHA входов/кода/артефактов.
Нет нового обучения, установки пакетов, commit/push или публикации.
Следующий конкретный шаг — независимый человеческий просмотр очереди и решение
о дополнительной временной истории; E07b classifier автоматически не начинается.
"""
    report = path(config["report_path"])
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(text,encoding="utf-8")
    print(str(report.relative_to(ROOT)))


def main() -> None:
    from sberforecast.early_warning_labels import build_early_warning_labels
    from sberforecast.early_warning_features import (build_prefix_history_features,build_origin_macro_features,
                                                    audit_feature_columns,feature_dictionary)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/early_warning_feasibility.yaml")
    parser.add_argument("--stage",choices=["build","report"],default="build")
    args = parser.parse_args()
    config_path = path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.stage == "report":
        build_report(config)
        return
    output = path(config["output_dir"])
    if output.exists():
        raise FileExistsError(output)
    started,clock = datetime.now(timezone.utc).isoformat(),time.perf_counter()
    old = path(config["reference"]["e06b_dir"])
    samples = read_csv(old / "forecast_cases.csv.gz")
    if len(samples) != 768 or samples.municipality_id.nunique() != 64 or samples.forecast_origin.nunique() != 12:
        raise ValueError("Pilot keys changed")
    residual_path = path(config["reference"]["e04_dir"]) / "real_residuals.csv.gz"
    residuals = read_csv(residual_path)
    observations = load_data(path(config["data"]["path"]),config["data"]["category"])
    observations = observations.loc[observations.municipality_id.isin(samples.municipality_id.unique())]
    labels = build_early_warning_labels(residuals,samples,config)
    e04 = path(config["reference"]["e04_dir"])
    selected_parameters = json.loads((e04 / "selected_parameters.json").read_text(encoding="utf-8"))
    feature_config = dict(config,detector_preparation=yaml.safe_load((e04 / "config_resolved.yaml").read_text(encoding="utf-8"))["preparation"])
    history,provenance = build_prefix_history_features(observations,residuals,samples,feature_config,selected_parameters)
    macro_dir = path(config["reference"]["macro_audit_dir"])
    macro_path = macro_dir / "macro_table.csv.gz"
    source_ids_path = path(config["reference"]["e05c_dir"]) / "source_ids.json"
    macro,macro_provenance = build_origin_macro_features(samples,read_csv(macro_path),json.loads(source_ids_path.read_text(encoding="utf-8")))
    news,news_stats = build_v3(config,samples)
    keys = ["municipality_id","forecast_origin"]
    features = history.merge(macro.drop(columns=[c for c in ("region_id","region") if c in macro]),on=keys,validate="one_to_one")
    features = features.merge(news[keys+list(FEATURE_COLUMNS)+list(MISSING_COLUMNS)+["source_archive_complete","historical_version_evidence_available"]],on=keys,validate="one_to_one")
    assert_feature_keys(features,samples)
    availability = training_availability(labels["cases"],features,labels["events"])
    split_cases,split_summary = temporal_split(labels["cases"],config,labels["events"])
    queue = manual_queue(labels,residuals)
    output.mkdir(parents=True)
    for name,table in labels.items():
        write_table(output,"weak_"+name,table)
    for name,table in (("features",features),("history_feature_provenance",provenance),("macro_feature_provenance",macro_provenance),
        ("training_availability",availability),("temporal_split_cases",split_cases),("temporal_split_summary",split_summary),("manual_review_queue",queue)):
        write_table(output,name,table)
    available_columns = [c for c in features if c not in ("municipality_id","forecast_origin","region_id","region")]
    feature_availability = []
    for origin,group in features.groupby("forecast_origin",sort=True):
        for column in available_columns:
            feature_availability.append(dict(forecast_origin=origin,feature=column,rows=len(group),
                nonmissing=int(group[column].notna().sum()),missing=int(group[column].isna().sum()),
                distinct_values_including_missing=int(group[column].nunique(dropna=False)),
                usage="prepared_candidate_not_automatically_selected_for_future_model"))
    write_table(output,"feature_availability_by_origin",pd.DataFrame(feature_availability))
    news_columns = list(FEATURE_COLUMNS)+list(MISSING_COLUMNS)
    audit_feature_columns(features,news_columns).to_csv(output / "news_feature_audit_full_history.csv",index=False)
    for k in config["early_warning"]["horizons_months"]:
        train_keys = split_cases.loc[split_cases.k.eq(k) & split_cases.split.eq("train"),keys].drop_duplicates()
        train_features = features.merge(train_keys,on=keys,validate="one_to_one")
        audit_feature_columns(train_features,news_columns).to_csv(output / f"news_feature_audit_train_k{k}.csv",index=False)
    field_defs = feature_dictionary()
    if isinstance(field_defs,pd.DataFrame):
        field_defs.to_csv(output / "history_feature_dictionary.csv",index=False)
    else:
        save_json(output / "history_feature_dictionary.json",field_defs)
    news_dictionary().to_csv(output / "news_feature_dictionary.csv",index=False)
    save_json(output / "field_definitions.json",dict(keys="municipality_id + forecast_origin (+ k for label cases)",
        onset_period="First calendar residual month in the three-month weak shift",confirmation_period="onset+2 (or later actual availability)",
        forecast_origin="Warning issuance month end midnight Moscow, with expenses L0 assumption",
        label_known_at="Uniform full window O+k+2; positive and negative admission wait for complete future evidence",
        label="1 onset in (O,O+k]; 0 complete evidence without onset; missing unknown/censored",
        at_risk="Exclude only confirmed active regime already known at O",
        retrospective_inside_regime_at_origin="Full-history diagnostic only; never a causal feature or risk exclusion",
        feature_selection="No automatic selection; full-history audits descriptive; future selection training only",
        macro="National annual forecast for calendar year of O, not actual monthly expenditure",
        news="Bounded collected corpus counts, not full regional event intensity"))
    offline = path(config["reference"]["e06a_dir"]) / "real_breakpoints.csv"
    breakpoints = read_csv(offline)
    breakpoints["usage"] = "diagnostic_manual_review_only_not_truth_or_feature"
    write_table(output,"offline_breakpoint_diagnostics",breakpoints)
    # Saved evidence for human review; no AI annotation is called independent.
    review_ids = queue.series_id.unique()
    write_table(output,"manual_review_residuals",residuals.loc[residuals.series_id.isin(review_ids)])
    (output / "config_resolved.yaml").write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding="utf-8")
    summary = dict(feature_rows=len(features),case_rows=len(labels["cases"]),weak_event_count=len(labels["events"]),
        weak_event_unique_onset_dates=labels["events"].onset_period.nunique(),manual_review_cases=len(queue),
        independent_human_annotations=0,news=news_stats,split=split_summary.to_dict("records"),
        classifier_fit=False,classification_metrics_obtained=False)
    save_json(output / "feasibility_summary.json",summary)
    inputs = [config_path,residual_path,macro_path,source_ids_path,offline,path(config["data"]["path"]),
        old / "forecast_cases.csv.gz",e04 / "selected_parameters.json",e04 / "config_resolved.yaml",
        path(config["news"]["audit_output_dir"]) / "run_manifest.json",ROOT / "outputs/e07a_checks/preservation_before.json"]
    manifest(output,config,inputs,started,clock)
    print(json.dumps(summary,ensure_ascii=False,default=str),flush=True)


if __name__ == "__main__":
    main()
