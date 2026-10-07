"""Render the completed REAL E08d diagnostic and two real-data figures.

No source reconstruction, label rebuilding, fitting, threshold selection,
network calls or generated observations are performed by this script.
The root runner owns run_manifest.json; this renderer never changes it.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
FEATURES = (
    "key_rate_level", "key_rate_delta_last", "key_rate_change_3m",
    "key_rate_change_6m", "months_since_rate_change", "usd_rub_last",
    "usd_rub_change_1m", "usd_rub_change_3m", "usd_rub_vol_1m", "usd_rub_vol_3m",
)
PRIMARY = {
    "P1": "key_rate_change_3m", "P2": "usd_rub_change_1m",
    "P3": "usd_rub_change_3m", "P4": "usd_rub_vol_3m",
}
TITLES = {
    "P1": "P1: ставка, изменение за 3 месяца",
    "P2": "P2: USD/RUB, изменение за 1 месяц",
    "P3": "P3: USD/RUB, изменение за 3 месяца",
    "P4": "P4: USD/RUB, волатильность за 3 месяца",
}
UNITS = {"P1": "п.п.", "P2": "log ratio", "P3": "log ratio", "P4": "std log returns"}
TABLE_NAMES = (
    "origin_event_study", "event_window_table", "diagnostic_comparisons",
    "permutation_results", "leave_one_event_out", "event_intensity", "eligibility_summary",
)


def project_path(value: str) -> Path:
    resolved = (ROOT / value).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError("E08d report paths must stay within the project.")
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def relative_link(destination: Path, report: Path) -> str:
    return os.path.relpath(destination, report.parent).replace("\\", "/")


def require_columns(frame: pd.DataFrame, names: list[str], table: str) -> None:
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f"{table} lacks required columns: {sorted(missing)}")


def explicit_bool(series: pd.Series, name: str) -> pd.Series:
    def convert(value):
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        raise ValueError(f"{name} requires explicit nonmissing true/false values.")
    return series.map(convert)


def moscow(value) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError("Missing E08d financial source or origin date.")
    return result.tz_localize("Europe/Moscow") if result.tzinfo is None else result.tz_convert("Europe/Moscow")


def table(frame: pd.DataFrame, columns: list[str], labels: dict | None = None) -> str:
    labels = labels or {}

    def cell(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, (bool, np.bool_)):
            return "да" if value else "нет"
        if isinstance(value, (float, np.floating)):
            return f"{value:.6g}"
        return str(value).replace("|", "/").replace("\n", " ")

    lines = ["| " + " | ".join(labels.get(name, name) for name in columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    lines += ["| " + " | ".join(cell(row[name]) for name in columns) + " |"
              for _, row in frame.iterrows()]
    return "\n".join(lines)


def load_checked(cfg: dict, validation: Path) -> tuple[Path, dict, dict, dict[str, pd.DataFrame]]:
    output = project_path(cfg["output_dir"])
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Completion is checked before any financial outcome table is opened.
    diagnostic_complete = (manifest.get("stage") == "REAL_FULL_DIAGNOSTIC" and
                           (manifest.get("complete") is True or
                            manifest.get("status") == "REAL_CODE_RUN_COMPLETE_REPORT_PENDING"))
    if not diagnostic_complete:
        raise ValueError("E08d report requires the completed real-data run.")
    for flag in ("real_data_only", "no_synthetic_data", "no_classifier_fits", "no_threshold_tuning"):
        if cfg.get(flag) is not True or manifest.get(flag) is not True:
            raise ValueError(f"E08d real-data-only requirement is not recorded: {flag}")
    if cfg.get("primary_diagnostics") != PRIMARY or cfg.get("maximum_figures") != 2:
        raise ValueError("The predeclared primary diagnostics or two-figure limit changed.")
    for name, expected in manifest["source_sha256"].items():
        if sha256(project_path(name)) != expected:
            raise ValueError(f"Frozen E08d input changed: {name}")
    required = {name + ".csv" for name in TABLE_NAMES}
    if not required.issubset(manifest["artifact_sha256"]):
        raise ValueError("Completed E08d manifest does not cover all report input tables.")
    # Verify every saved output covered by the completed manifest, including
    # tables not printed in full here, before reading numerical outcomes.
    for name, expected in manifest["artifact_sha256"].items():
        artifact = (output / name).resolve()
        if not artifact.is_relative_to(output) or sha256(artifact) != expected:
            raise ValueError(f"Completed E08d artifact changed: {name}")
    proof = json.loads(validation.read_text(encoding="utf-8"))
    if proof.get("status") != "PASS":
        raise ValueError("The independent real-data validation must pass before rendering.")
    proof_manifest = next((proof[name] for name in (
        "observed_run_manifest_sha256", "complete_manifest_sha256", "run_manifest_sha256", "manifest_sha256") if name in proof), None)
    if proof_manifest is None:
        raise ValueError("Independent validation must record its observed diagnostic manifest.")
    if not required.issubset(proof.get("artifact_sha256", {})):
        raise ValueError("Independent validation must cover all seven numerical report tables.")
    for name, expected in proof.get("artifact_sha256", {}).items():
        if manifest["artifact_sha256"].get(name) != expected:
            raise ValueError(f"Independent validation covers a different output artifact: {name}")
    for name, expected in proof.get("source_sha256", {}).items():
        if manifest["source_sha256"].get(name) != expected:
            raise ValueError(f"Independent validation covers a different frozen input: {name}")
    # Root finalization adds report/figure provenance to this manifest after
    # rendering. The mandatory matching seven-table hashes bind the proof to
    # the numerical diagnostic; its observed manifest hash remains context.
    frames = {name: pd.read_csv(output / (name + ".csv"), float_precision="round_trip")
              for name in TABLE_NAMES}
    validate_tables(manifest, frames)
    return output, manifest, proof, frames


def validate_tables(manifest: dict, frames: dict[str, pd.DataFrame]) -> None:
    study, windows, eligibility = (frames[name] for name in (
        "origin_event_study", "event_window_table", "eligibility_summary"))
    require_columns(study, ["forecast_origin", "max_source_date_used", "max_source_available_at_used",
                            "distance_to_next_onset_evaluation_only", *FEATURES], "origin_event_study")
    require_columns(eligibility, ["k", "n_origins", "n_eligible", "n_positive", "n_negative", "n_unknown"], "eligibility_summary")
    if study.forecast_origin.isna().any() or study.forecast_origin.duplicated().any():
        raise ValueError("E08d must have one nonmissing row per unique forecast origin.")
    if len(study) != manifest["source_counts"]["n_origins"]:
        raise ValueError("E08d origin count does not match its completed manifest.")
    if not eligibility.n_negative.eq(0).all() or manifest["primary_conclusion"] != "NO REAL PRECURSOR EVIDENCE":
        raise ValueError("This fixed E08d report requires its documented no-negative-date diagnostic.")
    if not np.isfinite(study.loc[:, list(FEATURES)].to_numpy(dtype=float)).all():
        raise ValueError("The real E08b financial values must remain observed, without replacement.")
    origins = study.forecast_origin.map(moscow)
    for name in ("max_source_date_used", "max_source_available_at_used"):
        if not study[name].map(moscow).le(origins).all():
            raise ValueError(f"Financial source exceeds its own origin: {name}")
    if not explicit_bool(study.distance_to_next_onset_evaluation_only, "distance evaluation-only").all():
        raise ValueError("Retrospective onset distance cannot be a model feature.")
    for k, suffix in ((1, "1m"), (3, "3m")):
        names = [f"known_label_k{k}", f"event_next_{suffix}", f"event_count_next_{suffix}",
                 f"affected_municipality_count_next_{suffix}", f"affected_share_next_{suffix}",
                 f"event_count_complete_k{k}", f"monitoring_eligible_at_risk_count_k{k}",
                 f"monitoring_known_municipality_count_k{k}", f"monitoring_unknown_municipality_count_k{k}",
                 f"monitoring_positive_municipality_count_k{k}", f"monitoring_positive_event_count_k{k}",
                 f"label_known_at_k{k}", f"label_status_k{k}"]
        require_columns(study, names, "origin_event_study")
        known = explicit_bool(study[f"known_label_k{k}"], f"known_label_k{k}")
        study[f"known_label_k{k}"] = known
        outcome = study[f"event_next_{suffix}"]
        if not outcome.loc[known].isin([0, 1]).all() or outcome.loc[~known].notna().any():
            raise ValueError("Unknown date labels must remain missing, never negative.")
        summary = eligibility.loc[eligibility.k.eq(k)]
        if len(summary) != 1:
            raise ValueError("Exactly one eligibility summary is required for each k.")
        expected = dict(n_origins=len(study), n_eligible=int(known.sum()),
                        n_positive=int((known & outcome.eq(1)).sum()),
                        n_negative=int((known & outcome.eq(0)).sum()), n_unknown=int((~known).sum()))
        if any(int(summary.iloc[0][name]) != value for name, value in expected.items()):
            raise ValueError("Date-level eligibility summary differs from the saved study.")
    require_columns(windows, ["onset_date", "onset_period", "event_count", "affected_municipalities",
                             "affected_share_population", "lead_months", "forecast_origin", "financial_available",
                             "known_label_k1", "known_label_k3", "previous_eligible_origin_k1",
                             "previous_eligible_origin_k3", *FEATURES], "event_window_table")
    windows["financial_available"] = explicit_bool(windows.financial_available, "financial_available")
    for k in (1, 3):
        windows[f"known_label_k{k}"] = explicit_bool(windows[f"known_label_k{k}"], f"window known_label_k{k}")
    if windows.duplicated(["onset_period", "lead_months"]).any() or not windows.lead_months.isin([1, 2, 3]).all():
        raise ValueError("Event windows must retain the three predeclared calendar leads.")
    if len(windows) != manifest["source_counts"]["unique_onset_dates"] * 3:
        raise ValueError("The six onset months must retain all three real financial windows.")
    for row in windows.itertuples():
        onset = pd.Period(row.onset_period, freq="M")
        if pd.Timestamp(row.onset_date).normalize() != onset.end_time.normalize():
            raise ValueError("onset_date must be the documented month-end marker.")
        if pd.Period(row.forecast_origin, freq="M") != onset - int(row.lead_months):
            raise ValueError("Financial lead dates do not match their onset calendar month.")
    available = windows.loc[windows.financial_available]
    if not np.isfinite(available.loc[:, list(FEATURES)].to_numpy(dtype=float)).all():
        raise ValueError("An available event window lacks a real financial value.")
    finance = study.set_index("forecast_origin")
    expected = finance.reindex(available.forecast_origin).loc[:, list(FEATURES)].to_numpy(dtype=float)
    if not np.allclose(expected, available.loc[:, list(FEATURES)].to_numpy(dtype=float), atol=1e-12, rtol=0):
        raise ValueError("Event-window values differ from the saved real origin matrix.")
    permutations = frames["permutation_results"]
    require_columns(permutations, ["diagnostic_id", "feature", "transform", "k", "n_eligible",
                                  "n_positive", "n_negative", "raw_exact_p", "holm_adjusted_p",
                                  "permutations_possible", "permutations_evaluated", "test_status", "holm_status"], "permutation_results")
    if len(permutations) != 8 or permutations.duplicated(["diagnostic_id", "k"]).any():
        raise ValueError("The predeclared primary permutation family must have eight rows.")
    missing_class = permutations.n_positive.eq(0) | permutations.n_negative.eq(0)
    if (permutations.loc[missing_class, ["raw_exact_p", "holm_adjusted_p"]].notna().any().any()
            or permutations.loc[missing_class, "permutations_evaluated"].ne(0).any()):
        raise ValueError("A nonestimable contrast cannot receive a p-value or evaluated assignments.")
    for table_name in ("diagnostic_comparisons", "leave_one_event_out"):
        frame = frames[table_name]
        fields = ["difference_in_means", "difference_in_medians", "rank_biserial"]
        if table_name == "leave_one_event_out":
            fields += ["effect_sign", "effect_ratio_to_full"]
        require_columns(frame, ["n_positive", "n_negative", *fields], table_name)
        missing = frame.n_positive.eq(0) | frame.n_negative.eq(0)
        if frame.loc[missing, fields].notna().any().any():
            raise ValueError(f"{table_name} replaces a nonestimable effect with a value.")
        for label, size in (("positive", frame.n_positive), ("negative", frame.n_negative)):
            if frame.loc[size.eq(0), [f"{label}_mean", f"{label}_median"]].notna().any().any():
                raise ValueError(f"{table_name} invents a summary for an empty class.")


def make_figures(output: Path, frames: dict[str, pd.DataFrame]) -> list[dict]:
    # Existing project dependency only; lazy import never starts a model.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    destination = output / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    study = frames["origin_event_study"].sort_values("forecast_origin", kind="stable")
    windows = frames["event_window_table"].sort_values(["onset_period", "lead_months"], kind="stable")
    markers = windows[["onset_period", "onset_date"]].drop_duplicates().sort_values("onset_period")
    x = pd.to_datetime(study.forecast_origin)
    files = []
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for ax, (diagnostic, feature) in zip(axes.flat, PRIMARY.items()):
            ax.plot(x, study[feature], marker="o", markersize=4, linewidth=1, color="#17608b")
            for marker in pd.to_datetime(markers.onset_date):
                ax.axvline(marker, color="#b46637", alpha=.6, linestyle=":", linewidth=1)
            ax.set(title=TITLES[diagnostic], ylabel=UNITS[diagnostic], xlabel="Forecast origin / календарная дата")
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.tick_params(axis="x", rotation=30)
            ax.grid(alpha=.2)
        fig.suptitle("E08d: 12 реальных финансовых origins; пунктир — шесть месячных onset markers")
        path = destination / "e08d_financial_by_origin.png"
        fig.savefig(path, dpi=150, metadata={"Software": "E08d real-data report renderer"})
        plt.close(fig)
        files.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=sha256(path),
                          source="origin_event_study.csv and event_window_table.csv",
                          panels=4, financial_origin_rows=len(study), plotted_financial_values=4 * len(study),
                          onset_month_markers=len(markers), line_policy="straight_segments_between_saved_real_points",
                          onset_marker_rule="month_end_display_marker_intra_month_day_unknown"))
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        colors = plt.get_cmap("tab10")
        for number, (onset, group) in enumerate(windows.groupby("onset_period", sort=True)):
            group = group.loc[group.financial_available].sort_values("lead_months", ascending=False)
            for ax, (diagnostic, feature) in zip(axes.flat, PRIMARY.items()):
                ax.plot(-group.lead_months, group[feature], marker="o", markersize=4,
                        linewidth=1, label=onset, color=colors(number))
                ax.set(title=TITLES[diagnostic], ylabel=UNITS[diagnostic],
                       xlabel="Календарных месяцев до onset month", xticks=[-3, -2, -1])
                ax.grid(alpha=.2)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, title="Onset month", loc="outside lower center", ncols=6)
        fig.suptitle("E08d: реальные значения на t−3, t−2, t−1 для шести onset months")
        path = destination / "e08d_event_centered_financial.png"
        fig.savefig(path, dpi=150, metadata={"Software": "E08d real-data report renderer"})
        plt.close(fig)
        files.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=sha256(path),
                          source="event_window_table.csv", panels=4, onset_months=len(markers),
                          saved_window_rows=len(windows), plotted_window_rows=int(windows.financial_available.sum()),
                          plotted_financial_values=int(windows.financial_available.sum()) * 4,
                          unique_financial_origins=int(windows.loc[windows.financial_available, "forecast_origin"].nunique()),
                          line_policy="three_saved_points_no_interpolation_or_smoothing",
                          overlapping_windows_are_not_independent=True))
    return files


def render(cfg: dict, output: Path, manifest: dict, frames: dict[str, pd.DataFrame],
           figures: list[dict], validation: Path, commands: list[str]) -> str:
    report = project_path(cfg["report_path"])
    study, windows, comparisons, permutations, loo, intensity, eligibility = (
        frames[name] for name in TABLE_NAMES)
    counts = manifest["source_counts"]
    conclusion = manifest["primary_conclusion"]
    core = comparisons.loc[comparisons.role.eq("PRIMARY") & comparisons["transform"].isin(["signed", "intrinsic"])]
    magnitudes = comparisons.loc[comparisons["transform"].eq("absolute_descriptive")]
    t1 = windows.loc[windows.lead_months.eq(1)].sort_values("onset_period")
    removals = loo[["omitted_onset_date", "k", "n_origins_removed", "removed_origins", "removal_status"]].drop_duplicates().sort_values(["omitted_onset_date", "k"])
    selected_intensity = intensity.loc[intensity.feature.isin(PRIMARY.values()) & intensity["transform"].isin(["signed", "intrinsic"])]
    count_table = windows[["onset_period", "onset_date", "event_count", "affected_municipalities", "affected_share_population"]].drop_duplicates().sort_values("onset_period")
    text = [
        "# E08d — real financial early-warning diagnostic", "",
        f"**{conclusion}. E08d real financial early-warning diagnostic complete.**", "",
        "На имеющейся истории основной positive-vs-negative контраст не идентифицирован: "
        "среди дат с известной меткой нет отрицательного класса. Это ограничение данных, "
        "а не доказательство отсутствия финансовых предвестников. Реальные leading "
        "financial indicators **не добавили доказательства раннего предупреждения** на этой истории.", "",
        "## Источники, событие и статистическая единица", "",
        f"Использованы только сохранённые реальные E08b key-rate/official-USD-RUB признаки "
        f"и E07b weak labels: {counts['panel_municipalities']} МО, {counts['weak_events']} weak events, "
        f"{counts['unique_onset_dates']} onset months, {counts['municipalities_with_events']} МО с событиями "
        f"и {counts['n_origins']} forecast origins. Состав источников и SHA проверены до чтения "
        "результатов. Метки не пересобирались; synthetic data, E07c и synthetic fixtures не использовались.", "",
        "Weak event — устойчивый сдвиг ошибки сохранённого причинного SeasonalNaiveYoY h=1, "
        "подтверждённый исходным трёхмесячным правилом E07. Это не независимая разметка "
        "реальных экономических shocks и не прямое определение роста/падения расходов. "
        "73 муниципальных события не являются 73 независимыми наблюдениями национального "
        "financial signal. Единица основной оценки — уникальная calendar origin.", "",
        "`onset_date` ниже — **month-end marker для onset month**, а не установленный день "
        "начала события внутри месяца. Внутримесячные даты onset неизвестны.", "",
        table(count_table, list(count_table.columns)), "",
        "## Известность меток и censoring", "",
        "Для каждой O сохраняется исходная causal monitoring population: eligible_at_origin "
        "и at_risk. Date-level OR положителен только при наличии хотя бы одной сохранённой "
        "fully-known positive witness. Отрицателен он только для непустой monitoring population, "
        "в которой **все** municipality labels fully-known negative. Иначе метка unknown. "
        "Unknown не заменяется нулём; retrospective regime membership не используется вместо "
        "causal active exclusion.", "",
        "Исходное правило требует прошлые residual baselines и полное будущее окно до O+k+2. "
        "Dec2023…Mar2024 не становятся negative controls: прошлых residuals недостаточно. "
        "Поздние окна right-censored. Существующий event registry сам по себе не позволяет "
        "сократить этот uniform confirmation window или объявить unknown positive/negative.", "",
        table(eligibility, ["k", "n_origins", "n_eligible", "n_positive", "n_negative", "n_unknown"]), "",
        "Здесь n_eligible означает даты с известной агрегированной меткой. Для k=1 это "
        "Apr…Sep2024, для k=3 — Apr…Jul2024; все эти даты positive. label_known_at — "
        "ретроспективная доступность outcome после confirmation window, не финансовый feature на O.", "",
    ]
    for k, suffix in ((1, "1m"), (3, "3m")):
        text += [f"### Сохранённые origin labels, k={k}", "",
                 table(study, ["forecast_origin", f"known_label_k{k}", f"event_next_{suffix}",
                               f"event_count_next_{suffix}", f"monitoring_positive_event_count_k{k}",
                               f"monitoring_known_municipality_count_k{k}", f"monitoring_unknown_municipality_count_k{k}",
                               f"label_known_at_k{k}"]), ""]
    text += [
        "event_count и affected counts относятся к **наблюдаемым подтверждённым событиям "
        "полного frozen registry**, а monitoring_positive_event_count — к known-positive "
        "events внутри causal at-risk cohort. Это разные denominators. На Aug2024 origin "
        "k=1 registry count=4, monitoring count=2; на Jul2024 origin k=3 — 13 и 11: "
        "два МО уже были known-active. affected_share — описательная доля от всех 2190 МО.", "",
        "Известный positive OR не означает известность точного числа событий всей monitoring "
        "population: неизвестные municipality labels остаются. event_count_complete_k1/k3 "
        "в этом запуске false и на known-positive dates. Наблюдаемые counts не заполняют "
        "unknown labels. distance_to_next_onset_months — исключительно retrospective "
        "evaluation-only поле; ни одной модели оно не передавалось.", "",
        "## Заранее заданные primary diagnostics", "",
        "P1 — key_rate_change_3m; P2 — usd_rub_change_1m; P3 — usd_rub_change_3m; "
        "P4 — usd_rub_vol_3m. Первые три используются со знаком; volatility уже intrinsic "
        "magnitude. Rate change измеряется в процентных пунктах; FX changes — natural-log "
        "ratios, volatility — sample std опубликованных setting log returns, ddof=1, "
        "без annualisation. Absolute magnitudes отдельно описательные, без дополнительных "
        "primary tests. Остальные шесть E08b features — secondary/descriptive only.", "",
        table(core, ["diagnostic_id", "feature", "k", "n_positive", "n_negative", "positive_mean",
                     "positive_median", "negative_mean", "negative_median", "difference_in_means",
                     "difference_in_medians", "rank_biserial", "comparison_status"]), "",
        "Positive-group summaries доступны, но при пустом negative class differences "
        "и rank-based effect size не определены. **NA не означает effect=0.** "
        "Формулы заранее заданы: Δmean = mean_positive − mean_negative; "
        "rank_biserial = (wins − losses)/(n_positive × n_negative), где wins — "
        "positive value выше negative value, losses — ниже, ties дают ноль.", "",
        "### Absolute magnitude — descriptive only", "",
        table(magnitudes, ["feature", "role", "k", "n_positive", "n_negative", "positive_mean",
                           "positive_median", "comparison_status"]), "",
        "## Exact permutation и Holm", "",
        "Планировался полный перебор fixed-positive-count assignments на уникальных датах "
        "для signed difference in means и intrinsic volatility: 4 diagnostics × 2 horizons, "
        "фиксированная Holm family из восьми гипотез. Ни municipality rows, ни отдельные "
        "муниципальные events не являются единицей перестановки. В этом запуске нет "
        "negative dates: statistic и p-values NONESTIMABLE, evaluated assignments=0. "
        "Raw/Holm p остаются NA, без подстановки p=1 и без уменьшения planned family.", "",
        "Для доступного контраста two-sided inclusive tail задавался как "
        "|T_perm| ≥ |T_observed|: p = число таких assignments / число всех assignments. "
        "Observed assignment входит в полный перебор; дополнительная +1 correction "
        "не применяется. Здесь ни одна из восьми planned primary tests фактически не оценена.", "",
        table(permutations, ["diagnostic_id", "k", "n_positive", "n_negative", "raw_exact_p",
                             "holm_adjusted_p", "permutations_possible", "permutations_evaluated",
                             "test_status", "holm_status"]), "",
        "Даже при доступном контрасте exact enumeration был бы описательной permutation "
        "reference: временная exchangeability не подтверждена, финансовые ряды имеют "
        "зависимость во времени, а k=3 окна перекрываются. Точность перебора не доказывает "
        "causal interpretation или калиброванный significance/FWER. Asymptotic p-values "
        "и бинарный success gate по p<0.05 не применялись.", "",
        "## Lead-time и каждый onset month", "",
        "Для всех шести onset months сохранены только заранее заданные t−1/t−2/t−3. "
        "Финансовые значения на origin с unknown event label можно показать как доступные "
        "реальные измерения; это не делает такую origin отрицательным контролем или "
        "пригодной для основного сравнения.", "",
        table(t1, ["onset_period", "event_count", "affected_municipalities", "forecast_origin",
                   "known_label_k1", "known_label_k3", *PRIMARY.values()]), "",
        table(windows, ["onset_period", "lead_months", "forecast_origin", "financial_available",
                        "known_label_k1", "known_label_k3", "previous_eligible_origin_k1",
                        "previous_eligible_origin_k3"]), "",
    ]
    for row in t1.itertuples():
        values = "; ".join(f"{name}={getattr(row, feature):.6g}" for name, feature in PRIMARY.items())
        text.append(f"- **{row.onset_period}**: {row.event_count} зарегистрированных events, "
                    f"{row.affected_municipalities} МО; t−1 origin {row.forecast_origin}. {values}. "
                    "Это описание реального предшествующего financial vector. Отдельный precursor "
                    "effect относительно negative controls для этой даты не идентифицирован.")
    sign_descriptions = []
    for diagnostic, feature in list(PRIMARY.items())[:3]:
        observed = t1.loc[t1.financial_available, feature]
        sign_descriptions.append(f"{diagnostic}: положительных значений {int(observed.gt(0).sum())}, "
                                 f"отрицательных {int(observed.lt(0).sum())}, нулевых {int(observed.eq(0).sum())}")
    text += ["", "Знаки на t−1 различаются между onset months: " + "; ".join(sign_descriptions) + ". "
             "Это описание заранее заданных signed indicators, без выбора окна/threshold. "
             "Variation volatility и совпадение отдельных extremes с отдельными events "
             "не доказывают общий precursor pattern или его устойчивость относительно controls."]
    text += [
        "", "## Leave-one-onset-out", "",
        "Для каждого omitted onset month удаляются первоначально known windows, "
        "содержащие этот onset; оставшиеся labels сохраняются. Positive не превращаются "
        "в negative после удаления даты. REMOVED/NOOP обозначают действие с окнами, "
        "а NONESTIMABLE — доступность контраста. NOOP не подтверждает независимую устойчивость.", "",
        table(removals, list(removals.columns)), "",
        f"Сохранено {len(loo)} primary signed/magnitude sensitivity rows. Контраст, его sign "
        "и ratio к full effect не определены при отсутствии negative class. Нельзя "
        "назвать результат устойчивым или ONE-EVENT-DRIVEN: исходный effect не оценён.", "",
        "## Event intensity — descriptive only", "",
        "Spearman рассчитывается на уникальных известных датах для observed registry "
        "counts/affected municipalities/share, с фильтрацией совместно конечных пар и "
        "average ranks при ties. Это small-N описание вариации интенсивности среди "
        "positive dates, а не оценка discrimination и не causal evidence. Повторения "
        "national features по МО не увеличивают N; неизвестные dates не добавляются как zero events.", "",
        table(selected_intensity, ["feature", "k", "outcome", "n_dates", "spearman_rho", "status"]), "",
        "## Две фигуры из реальных значений", "",
        f"![Financial indicators by real origin]({relative_link(project_path(figures[0]['path']), report)})", "",
        "Каждая панель содержит 12 сохранённых financial values. Пунктир обозначает "
        "условные month-end onset markers; внутримесячный день события неизвестен. "
        "Линии соединяют измеренные точки, сглаживания и fitting нет.", "",
        f"![Real event-centered financial trajectories]({relative_link(project_path(figures[1]['path']), report)})", "",
        "Каждая линия показывает только три реальные предшествующие calendar origins. "
        "Окна разных onset months перекрываются; общие national observations повторяются "
        "в представлении и не считаются независимыми. Ни interpolated, ни synthetic points нет.", "",
        "## Связь с E08c/E07 и вывод", "",
        "E08c дал **NO STABLE FORECASTING UPLIFT**. E08d задаёт отдельный вопрос о real "
        "financial precursors перед weak-event onsets; forecasting MAE не смешивается "
        "с event-study diagnostics. Реальный E07 feasibility gate остаётся в силе: "
        "classifiers не обучались из-за недостаточного временного числа событий. "
        "Этот diagnostic не отменяет тот вывод и не доказывает real early warning.", "",
        f"**{conclusion}** в этом запуске означает отсутствие установленного precursor "
        "evidence вследствие неидентифицируемого positive-vs-negative сравнения. "
        "Positive-only summaries и descriptive intensity correlation не восстанавливают "
        "контрольный класс и не позволяют выбрать evidence level по случайному signal. "
        "Это не доказательство отсутствия pattern; production-ready раннее предупреждение не установлено.", "",
        "Ограничения: weak/не независимые labels; residual history начинается позже "
        "истории исходной цели; left insufficiency, right censoring и municipality gaps; "
        "L=0 и vintages цели не подтверждены; conditional official archive trust; "
        "June2024 FX methodology boundary; мало независимых календарных дат и "
        "перекрытие k=3 windows. Thresholds не подбирались, classifier-style metrics не рассчитывались.", "",
        "## Проверки, артефакты и runtime", "",
        "Проверены unique origins, сохранение unknown, own-origin financial source "
        "cutoffs, совпадение всех event-window values с реальной E08b matrix, "
        "неизменность входных SHA и завершённых output SHA. Независимая real-data "
        f"проверка: [PASS]({relative_link(validation, report)}). Synthetic tests/benchmarks "
        "и classifier fits отсутствуют.", "",
        "```powershell", *commands, "```", "",
        f"Runtime основного расчёта: {float(manifest['runtime_seconds']):.3f} s. "
        "Report/figure rendering — отдельная команда, без models или изменения исходных labels.", "",
        f"[Run manifest]({relative_link(output / 'run_manifest.json', report)}); "
        f"[figure provenance]({relative_link(output / 'figures/figure_manifest.json', report)}).", "",
    ]
    text += [f"- [{name}.csv]({relative_link(output / (name + '.csv'), report)})" for name in TABLE_NAMES]
    text += ["", "F1–F7, E08a/b/c, README, final methodology report и presentation не обновляются. "
             "Следующий шаг — отдельно решить, нужна ли дополнительная реальная история "
             "с сопоставимыми known negative dates; новые sources/models этим diagnostic не разрешаются.", "",
             "REAL DATA ONLY; NO SYNTHETIC DATA; NO CLASSIFIER FITS; NO THRESHOLD TUNING; NO COMMIT/PUSH.", ""]
    return "\n".join(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e08d_real_financial_early_warning.yaml")
    parser.add_argument("--validation", default="outputs/e08d_checks/independent_validation.json")
    args = parser.parse_args()
    cfg = yaml.safe_load(project_path(args.config).read_text(encoding="utf-8"))
    validation = project_path(args.validation)
    executable = Path(sys.executable).resolve()
    if not executable.is_relative_to(ROOT):
        raise ValueError("Use this project's .venv/Scripts/python.exe for E08d rendering.")
    command = executable.relative_to(ROOT).as_posix() + " -B -X utf8 scripts/report_real_financial_early_warning.py"
    if sys.argv[1:]:
        command += " " + " ".join(sys.argv[1:])
    output, manifest, proof, frames = load_checked(cfg, validation)
    commands = []
    additional_records = {}
    smoke_manifest = project_path("outputs/real_financial_early_warning_e08d_smoke_v1/run_manifest.json")
    if smoke_manifest.exists():
        smoke = json.loads(smoke_manifest.read_text(encoding="utf-8"))
        if smoke.get("real_data_only") is True:
            commands.extend(smoke.get("run_commands", []))
            additional_records[smoke_manifest.relative_to(ROOT).as_posix()] = sha256(smoke_manifest)
    commands.extend(manifest["run_commands"])
    real_validation = project_path("outputs/e08d_checks/real_validation.json")
    if real_validation.exists():
        real_check = json.loads(real_validation.read_text(encoding="utf-8"))
        if real_check.get("status") == "PASS" and real_check.get("command"):
            commands.append(real_check["command"])
            additional_records[real_validation.relative_to(ROOT).as_posix()] = sha256(real_validation)
    if proof.get("command"):
        commands.append(proof["command"])
    commands.append(command)
    commands = list(dict.fromkeys(commands))
    figures = make_figures(output, frames)
    report = project_path(cfg["report_path"])
    content = render(cfg, output, manifest, frames, figures, validation, commands)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(content, encoding="utf-8")
    figure_manifest = dict(
        real_data_only=True, no_synthetic_data=True, no_fitting=True,
        figures=figures, figure_count=len(figures), maximum_figures=2,
        source_sha256={name + ".csv": manifest["artifact_sha256"][name + ".csv"] for name in TABLE_NAMES},
        completed_diagnostic_manifest_sha256=sha256(output / "run_manifest.json"),
        independent_validation_sha256=sha256(validation), script_sha256=sha256(Path(__file__)),
        independent_observed_run_manifest_sha256=next(proof[name] for name in (
            "observed_run_manifest_sha256", "complete_manifest_sha256", "run_manifest_sha256", "manifest_sha256") if name in proof),
        independent_validation_binding="all_seven_numerical_table_sha256",
        command=command,
        commands=commands, additional_command_record_sha256=additional_records,
        report_path=report.relative_to(ROOT).as_posix(), report_sha256=sha256(report),
        onset_dates_are_month_end_markers=True, intra_month_onset_days_known=False,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    save_json(output / "figures/figure_manifest.json", figure_manifest)
    save_json(project_path(cfg["audit_dir"]) / "figure_manifest.json", figure_manifest)
    print(f"Report saved: {report.relative_to(ROOT).as_posix()}; figures=2; {manifest['primary_conclusion']}")


if __name__ == "__main__":
    main()
