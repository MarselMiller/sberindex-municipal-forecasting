"""Publish saved aggregate results for the project report using only the standard library.

This builder reads an explicit list of public reports. It does not import research
modules, read municipal observations, fit models or recalculate research metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = [
    "SeasonalNaive", "SeasonalNaiveYoY", "ProphetAuto", "ProphetYearly",
    "CatBoostDirect", "LightGBMDirect", "National/Local + CatBoost",
    "National/Local + LightGBM", "Chronos-2",
]
FINANCIAL_VARIANT_LABELS = {
    "F0": "Базовая модель", "F1": "+ ключевая ставка",
    "F2": "+ USD/RUB", "F3": "+ ставка и USD/RUB",
}
FINANCIAL_MODEL_LABELS = {"LightGBMDirectNationalLocal": "National/Local + LightGBM", "LightGBMDirect": "LightGBMDirect"}
SYNTHETIC_WARNING_LABELS = {
    "S0": "Constant risk", "S1": "History only",
    "S2": "History + detector state",
    "S3": "History + detector state + external precursors",
}
FEATURES = [
    ("key_rate_level", "Ключевая ставка", "%"),
    ("key_rate_delta_last", "Последнее изменение ставки", "процентные пункты"),
    ("key_rate_change_3m", "Изменение ставки за 3 месяца", "процентные пункты"),
    ("key_rate_change_6m", "Изменение ставки за 6 месяцев", "процентные пункты"),
    ("months_since_rate_change", "Месяцы после изменения ставки", "месяцы"),
    ("usd_rub_last", "Официальный USD/RUB", "рублей за 1 USD"),
    ("usd_rub_change_1m", "Изменение USD/RUB за месяц", "log ratio"),
    ("usd_rub_change_3m", "Изменение USD/RUB за 3 месяца", "log ratio"),
    ("usd_rub_vol_1m", "Волатильность USD/RUB за месяц", "std log returns"),
    ("usd_rub_vol_3m", "Волатильность USD/RUB за 3 месяца", "std log returns"),
]


def read_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8-sig"))


def sources(*names: str) -> list[dict[str, str]]:
    return [
        {"path": name,
         "sha256": hashlib.sha256((ROOT / name).read_text(encoding="utf-8-sig").encode("utf-8")).hexdigest(),
         "hash_basis": "UTF-8 text normalized to LF, without BOM"}
        for name in names
    ]


def number(value: str | int | float | None, *, integer: bool = False):
    if value is None or str(value).strip() in {"", "NA"}:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite value in a published aggregate: {value!r}")
    if integer:
        if not result.is_integer():
            raise ValueError(f"Expected an integer aggregate count: {value!r}")
        return int(result)
    return result


def normalized(value: str) -> str:
    return " ".join(value.split())


def forecasting_data() -> dict:
    path = "reports/final/forecasting_metrics.csv"
    source = read_csv(path)
    expected = {(m, split, h) for m in MODEL_ORDER for split in ["holdout", "validation"] for h in [1, 3, 6, 12]}
    actual = {(r["model"], r["split"], int(r["horizon"])) for r in source}
    if len(source) != 72 or actual != expected:
        raise ValueError("Final forecasting CSV must contain nine strategies × two splits × four horizons")
    rows = []
    for raw in sorted(source, key=lambda r: (r["split"], MODEL_ORDER.index(r["model"]), int(r["horizon"]))):
        row = {"model": raw["model"], "label": raw["model"], "split": raw["split"], "data_status": raw["data_status"]}
        for key in ["mae_macro", "mae_micro", "r2_pooled"]:
            row[key] = number(raw[key])
        for key in ["horizon", "n_predictions", "n_municipalities", "n_origins", "native_count", "fallback_count", "failed_count"]:
            row[key] = number(raw[key], integer=True)
        row["metric_status"] = raw["metric_status"]
        row["note"] = ""
        if row["horizon"] == 12:
            row["note"] = "Нет пригодных validation cases." if row["split"] == "validation" else "Одна дата выпуска; устойчивый ranking годового горизонта не установлен."
            if row["split"] == "holdout" and row["fallback_count"]:
                row["note"] += " Все значения — SeasonalNaive fallback, а не обученная годовая direct-модель."
        rows.append(row)
    return {
        "schema_version": 1, "sources": sources(path),
        "models": [{"id": model, "label": model} for model in MODEL_ORDER],
        "horizons": [1, 3, 6, 12], "splits": ["holdout", "validation"], "default_split": "holdout", "metric": "mae_macro", "unit": "nominal_RUB",
        "note": "Общая benchmark-выборка муниципальных расходов СберИндекса: категория «Все категории», оценка средних безналичных расходов жителей. Просмотренный holdout июля–декабря 2024, 63 оцениваемых МО.",
        "rows": rows,
    }


def detection_data() -> dict:
    path = "reports/final/detection_metrics.csv"
    synthetic, diagnostic = [], []
    fields = ["precision", "recall", "f1", "miss_rate", "median_detection_delay", "median_absolute_localisation_error", "false_positives_per_12_months"]
    counts = ["n_series", "n_events", "n_monitoring_months", "n_alarms", "n_candidates", "n_monitoring_candidates", "n_warmup_candidates", "n_date_revisions", "n_presence_losses"]
    for raw in read_csv(path):
        row = {key: raw[key] for key in ["family", "method", "data_status", "access_mode"]}
        row["label"] = "Binary Segmentation" if raw["method"] == "BinSeg" else raw["method"]
        row.update({key: number(raw[key]) for key in fields})
        row.update({key: number(raw[key], integer=True) for key in counts})
        if raw["data_status"] == "synthetic" and raw["split"] == "test" and raw["scope"] == "primary_level":
            synthetic.append(row)
        elif raw["data_status"] == "diagnostic":
            diagnostic.append(row)
    if len(synthetic) != 5 or len(diagnostic) != 5:
        raise ValueError("Expected five saved benchmark methods and five aggregate diagnostic rows")
    return {
        "schema_version": 1, "sources": sources(path), "rows": synthetic, "diagnostic_rows": diagnostic,
        "note": "Quality metrics относятся к controlled synthetic benchmark с 360 test-событиями. На СберИндексе сохранены только диагностические сигналы без независимой разметки. Offline localisation error не является online delay или lead time.",
    }


def financial_data() -> dict:
    feature_path = "reports/results/e08b/financial_features_by_origin.csv"
    ablation_path = "reports/results/e08c/forecasting_metrics.csv"
    origins = []
    for raw in read_csv(feature_path):
        row = {"forecast_origin": raw["forecast_origin"][:10], "origin_cutoff": raw["forecast_origin"]}
        row.update({name: number(raw[name]) for name, _, _ in FEATURES})
        origins.append(row)
    if len(origins) != 12 or len({r["forecast_origin"] for r in origins}) != 12:
        raise ValueError("Expected twelve distinct saved financial forecast origins")
    ablation = []
    for raw in read_csv(ablation_path):
        if int(raw["horizon"]) not in [1, 3, 6]:
            continue
        row = {key: raw[key] for key in ["model", "split", "variant", "evaluation_role"]}
        row["model_label"] = FINANCIAL_MODEL_LABELS[raw["model"]]
        row["variant_label"] = FINANCIAL_VARIANT_LABELS[raw["variant"]]
        row.update({key: number(raw[key]) for key in ["mae_macro", "delta_mae_macro_vs_F0", "relative_delta_mae_macro_pct_vs_F0"]})
        row.update({key: number(raw[key], integer=True) for key in ["horizon", "n_predictions", "n_origins", "n_native", "n_fallback", "n_failed"]})
        ablation.append(row)
    if len(ablation) != 48:
        raise ValueError("Expected forty-eight fixed saved h1/h3/h6 financial ablation rows")
    return {
        "schema_version": 1, "sources": sources(feature_path, ablation_path),
        "features": [{"id": name, "label": label, "unit": unit} for name, label, unit in FEATURES],
        "origins": origins, "ablation": ablation, "conclusion": "NO STABLE FORECASTING UPLIFT",
        "note": "Ключевая ставка и официальный USD/RUB Банка России. Использованы сведения, доступные на собственной origin при cutoff 00:00 Москвы последнего дня месяца и принятом доверии официальному архиву. Национальные признаки не увеличивают число независимых календарных дат. Только USD/RUB: h6 −1.67% / −2.11% MAE; ставка и USD/RUB: −1.67% / −1.78% на validation / просмотренном holdout. Validation h6 имеет одну origin.",
    }


def early_warning_data() -> dict:
    summary_path = "reports/final/results_summary.json"
    synthetic_path = "reports/final/early_warning_metrics.csv"
    cohort_path = "reports/results/e08d/eligibility_summary.csv"
    manifest_path = "reports/results/e08d/run_manifest.json"
    origin_path = "reports/results/e08d/origin_event_study.csv"
    summary = read_json(summary_path)
    panel = summary["real_early_warning"]["E07b"]
    cohort_fields = {"k": "k", "origins": "n_origins", "eligible": "n_eligible", "positive": "n_positive", "negative": "n_negative", "unknown": "n_unknown"}
    cohorts = [{key: number(raw[source], integer=True) for key, source in cohort_fields.items()} for raw in read_csv(cohort_path)]
    manifest = read_json(manifest_path)
    origins = [
        {"forecast_origin": raw["forecast_origin"][:10],
         "k1_label": {None: "unknown", 0: "negative", 1: "positive"}[number(raw["event_next_1m"], integer=True)],
         "k3_label": {None: "unknown", 0: "negative", 1: "positive"}[number(raw["event_next_3m"], integer=True)]}
        for raw in read_csv(origin_path)
    ]
    if len(origins) != 12 or len({r["forecast_origin"] for r in origins}) != 12:
        raise ValueError("Expected twelve aggregate event-study dates")
    synthetic = []
    for raw in read_csv(synthetic_path):
        if raw["experiment"] != "E07c" or raw["cohort"] != "test" or raw["scenario"] != "base_all":
            continue
        row = {key: raw[key] for key in ["model", "data_status", "evaluation_status"]}
        row["label"] = SYNTHETIC_WARNING_LABELS[raw["model"]]
        row.update({key: number(raw[key], integer=True) for key in ["k", "cases", "positives", "negatives", "eligible_events", "warned_events"]})
        row.update({key: number(raw[key]) for key in ["pr_auc", "row_f1", "event_recall", "alert_precision", "median_lead_time_months", "false_alerts_per_12_monitored_months"]})
        synthetic.append(row)
    if len(synthetic) != 8 or len(cohorts) != 2:
        raise ValueError("Expected eight synthetic strategy TEST rows and two saved origin-level cohorts")
    return {
        "schema_version": 1, "sources": sources(summary_path, synthetic_path, cohort_path, manifest_path, origin_path),
        "real": {
            "weak_events": panel["weak_events"], "onset_months": panel["weak_event_onset_dates"],
            "event_municipalities": panel["municipalities_with_event"], "panel_municipalities": panel["municipalities"],
            "origin_count": 12, "onset_dates": manifest["onset_dates"], "cohorts": cohorts, "origins": origins,
            "statistical_unit": "unique_forecast_origin", "classifier_fits": panel["classifiers_fitted"],
            "quality_metrics": panel["metrics"], "quality_status": panel["metric_status"],
            "financial_conclusion": manifest["primary_conclusion"],
            "note": "Weak-event labels описывают сдвиги ошибки SeasonalNaiveYoY, а не независимо размеченные экономические шоки. В анализе financial indicators нет известных отрицательных origin dates: сравнительный precursor effect и permutation inference не оцениваются. Это не доказательство отсутствия предвестников.",
        },
        "synthetic": {
            "data_status": "synthetic", "cohorts": summary["synthetic_early_warning"]["cohorts"],
            "rows": sorted(synthetic, key=lambda r: (r["k"], r["model"])),
            "note": "Controlled synthetic TEST: сгенерированные предвестники, события без них и ложные precursor controls. Результат условен на генераторе и не подтверждает способность предупреждать экономические шоки по муниципальным расходам.",
        },
    }


def first_sentence(paragraph: str) -> str:
    match = re.search(r"[.!?](?:\s|$)", paragraph)
    return paragraph[:match.start() + 1] if match else paragraph


def glossary_data() -> dict:
    name = "reports/final/terminology.md"
    content = (ROOT / name).read_text(encoding="utf-8-sig")
    groups, terms = [], []
    current_group = None
    pending_anchor = None
    current_term = None
    lines = []

    def finish_term():
        nonlocal current_term, lines
        if current_term is None:
            return
        blocks = re.split(r"\n\s*\n", "\n".join(lines).strip())
        paragraphs = [normalized(block) for block in blocks if block.strip()]
        if not paragraphs:
            raise ValueError(f"Glossary definition is empty: {current_term['id']}")
        current_term["paragraphs"] = paragraphs
        current_term["definition"] = "\n\n".join(paragraphs)
        current_term["first_sentence"] = first_sentence(paragraphs[0])
        terms.append(current_term)
        current_group["term_ids"].append(current_term["id"])
        current_term, lines = None, []

    for line in content.splitlines():
        group = re.fullmatch(r"##\s+(\d+)\.\s+(.+)", line.strip())
        anchor = re.fullmatch(r'<a\s+id=["\']([a-z0-9-]+)["\']\s*></a>', line.strip())
        term = re.fullmatch(r"###\s+(.+?)\s+/\s+(.+)", line.strip())
        if group:
            finish_term()
            current_group = {"id": "group-" + group[1], "title": group[2], "term_ids": []}
            groups.append(current_group)
        elif anchor:
            finish_term()
            pending_anchor = anchor[1]
        elif term:
            if current_group is None or pending_anchor is None:
                raise ValueError("Every glossary term needs a numbered group and canonical anchor")
            current_term = {"id": pending_anchor, "en": term[1], "ru": term[2], "group_id": current_group["id"]}
            pending_anchor, lines = None, []
        elif current_term is not None:
            lines.append(line)
    finish_term()
    if not terms or len({term["id"] for term in terms}) != len(terms):
        raise ValueError("Canonical glossary requires nonempty entries with unique anchors")
    return {"schema_version": 1, "sources": sources(name), "groups": groups, "terms": terms}


def build_data() -> dict:
    return {
        "forecasting": forecasting_data(), "detection": detection_data(),
        "financial_indicators": financial_data(), "early_warning": early_warning_data(),
        "glossary": glossary_data(),
    }


def json_text(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def data_outputs(data: dict) -> dict[Path, str]:
    result = {ROOT / "docs/data" / (name + ".json"): json_text(value) for name, value in data.items()}
    serialized = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    result[ROOT / "docs/assets/project-report-data.js"] = (
        "// Generated from published aggregate reports. Rebuild with scripts/build_project_report.py.\n"
        "window.PROJECT_REPORT_DATA = " + serialized + ";\n"
    )
    return result


def formatted(value, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value:,.{digits}f}".replace(",", "\u00a0").replace(".", ",")


def table(caption: str, headings: list[str], rows: list[list[str]]) -> str:
    head = "".join(f'<th scope="col">{html.escape(value)}</th>' for value in headings)
    body = []
    for values in rows:
        cells = f'<th scope="row">{html.escape(values[0])}</th>'
        cells += "".join(f"<td>{html.escape(value)}</td>" for value in values[1:])
        body.append("<tr>" + cells + "</tr>")
    return f'<table class="metrics-table"><caption>{html.escape(caption)}</caption><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def render_fragments(data: dict) -> dict[str, str]:
    fragments = {}
    lookup = {(r["model"], r["split"], r["horizon"]): r for r in data["forecasting"]["rows"]}
    forecast_tables = []
    for split, label in [("holdout", "Holdout"), ("validation", "Validation")]:
        rows = []
        for model in MODEL_ORDER:
            values = [model]
            for horizon in [1, 3, 6, 12]:
                row = lookup[(model, split, horizon)]
                value = formatted(row["mae_macro"])
                if row["fallback_count"]:
                    value += "†"
                values.append(value)
            rows.append(values)
        forecast_tables.append(table(label + ": MAE macro, руб.", ["Стратегия", "h = 1", "h = 3", "h = 6", "h = 12"], rows))
    fragments["forecasting-table"] = "\n".join(forecast_tables)
    fragments["detection-table"] = table(
        "Synthetic TEST: качество обнаружения уже начавшегося сдвига",
        ["Метод", "Режим", "Precision", "Recall", "F1", "Miss rate", "Delay, мес.", "Localisation error, мес.", "FP / 12 мес."],
        [[r["label"], r["family"], *[formatted(r[key], 3) for key in ["precision", "recall", "f1", "miss_rate"]], formatted(r["median_detection_delay"]), formatted(r["median_absolute_localisation_error"]), formatted(r["false_positives_per_12_months"], 3)] for r in data["detection"]["rows"]],
    )
    fragments["financial-table"] = table(
        "Показатели Банка России, доступные на дату выпуска",
        ["Forecast origin", "Ключевая ставка, %", "USD/RUB, руб. за 1 USD"],
        [[r["forecast_origin"], formatted(r["key_rate_level"]), formatted(r["usd_rub_last"], 4)] for r in data["financial_indicators"]["origins"]],
    )
    fragments["early-warning-table"] = table(
        "Уникальные даты с известной weak-event меткой",
        ["k, мес.", "Origins", "Eligible", "Positive", "Negative", "Unknown"],
        [[str(r[key]) for key in ["k", "origins", "eligible", "positive", "negative", "unknown"]] for r in data["early_warning"]["real"]["cohorts"]],
    )
    fragments["synthetic-warning-table"] = table(
        "Controlled synthetic TEST: k = 3, 180 событий",
        ["Стратегия", "PR-AUC", "Row F1", "Event recall", "Alert precision", "Median lead, мес.", "False alerts / 12 мес."],
        [[r["label"], *[formatted(r[key], 3) for key in ["pr_auc", "row_f1", "event_recall", "alert_precision"]], formatted(r["median_lead_time_months"]), formatted(r["false_alerts_per_12_monitored_months"], 3)] for r in data["early_warning"]["synthetic"]["rows"] if r["k"] == 3],
    )
    glossary = data["glossary"]
    terms = {term["id"]: term for term in glossary["terms"]}
    sections = []
    for group in glossary["groups"]:
        entries = []
        for term_id in group["term_ids"]:
            term = terms[term_id]
            paragraphs = "".join(f"<p>{html.escape(p)}</p>" for p in term["paragraphs"])
            entries.append(
                f'<details class="glossary-term" id="{term_id}" data-group="{group["id"]}">'
                f'<summary><span class="term-en">{html.escape(term["en"])}</span><span class="term-ru">{html.escape(term["ru"])}</span></summary>'
                f'<div class="glossary-definition">{paragraphs}</div></details>'
            )
        sections.append(f'<section class="glossary-group" data-group="{group["id"]}"><h3>{html.escape(group["title"])}</h3>{"".join(entries)}</section>')
    fragments["glossary"] = "\n".join(sections)
    return fragments


def render_index(content: str, data: dict) -> str:
    for marker, fragment in render_fragments(data).items():
        begin = "<!-- " + marker + ":begin -->"
        end = "<!-- " + marker + ":end -->"
        if content.count(begin) != 1 or content.count(end) != 1:
            raise ValueError(f"HTML requires exactly one begin/end marker pair: {marker}")
        pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
        content = pattern.sub(lambda _: begin + "\n" + fragment + "\n" + end, content, count=1)
    terms = {term["id"]: term for term in data["glossary"]["terms"]}
    pattern = re.compile(
        r'(?P<opening><(?P<tag>[a-z][a-z0-9]*)\b[^>]*\bdata-first-definition=(?P<quote>["\'])(?P<id>[a-z0-9-]+)(?P=quote)[^>]*>)(?P<body>.*?)(?P<closing></(?P=tag)>)',
        re.DOTALL,
    )
    count = 0

    def definition(match):
        nonlocal count
        term_id = match["id"]
        if term_id not in terms:
            raise ValueError(f"HTML first-definition ID is absent from canonical glossary: {term_id}")
        count += 1
        sentence = terms[term_id]["first_sentence"]
        if match["tag"] == "span" and " — " in sentence:
            prefix, body = sentence.split(" — ", 1)
            if len(prefix) <= 70:
                sentence = body
        return match["opening"] + html.escape(sentence) + match["closing"]

    content = pattern.sub(definition, content)
    if count != len(re.findall(r"\bdata-first-definition=", content)):
        raise ValueError("Unexpected or unclosed HTML first-definition element")
    return content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify saved data files without writing")
    parser.add_argument("--data-only", action="store_true", help="Build the five JSON files and offline bundle")
    args = parser.parse_args()
    data = build_data()
    outputs = data_outputs(data)
    if not args.data_only:
        index = ROOT / "docs/index.html"
        outputs[index] = render_index(index.read_text(encoding="utf-8"), data)
    failures = []
    for path, text in outputs.items():
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                failures.append(path.relative_to(ROOT).as_posix())
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")
    if args.check and not args.data_only:
        original = ROOT / "reports/final/figures/rolling_forecast.png"
        copy = ROOT / "docs/assets/figures/rolling_forecast.png"
        if not copy.is_file() or hashlib.sha256(copy.read_bytes()).digest() != hashlib.sha256(original.read_bytes()).digest():
            failures.append("docs/assets/figures/rolling_forecast.png (must be an unchanged copy of the published figure)")
    if failures:
        raise SystemExit("Generated report data are stale: " + ", ".join(failures))
    summary = f"72 forecasting rows, 5 benchmark detection rows, 12 financial origins, 48 ablation rows, 8 synthetic warning rows, {len(data['glossary']['terms'])} canonical terms"
    print("PASS: five public JSON datasets, offline bundle" + (" and HTML" if not args.data_only else "") + " " + ("checked" if args.check else "generated") + "; " + summary)


if __name__ == "__main__":
    main()
