"""E08b acquisition/cache audit and financial features, without target access."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sberforecast.backtest import origins_from_config
from sberforecast.data import period_end, sha256_file
from sberforecast.leading_financial import (
    FEATURE_COLUMNS, build_financial_features, join_financial_features,
    origin_timestamp, parse_key_rate_xml, parse_usd_rub_xml, reconstruct_key_events,
)
from sberforecast.leading_financial_sources import collect_key_decisions


def project_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("E08b paths must stay in the project")
    return path


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def get_sources(cfg: dict, download: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    settings = cfg["sources"]
    cache = project_path(settings["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    sources = []
    for kind, file_key, url_key in [("key_rate", "key_xml", "key_url"), ("usd_rub", "fx_xml", "fx_url")]:
        path = cache / settings[file_key]
        url = settings[url_key]
        parsed = urlsplit(url)
        if (parsed.scheme != "https" or parsed.hostname not in {"cbr.ru", "www.cbr.ru"}
                or parsed.username or parsed.password):
            raise ValueError("Only official public CBR URLs may enter the financial loader")
        # Raw financial snapshots must retain the existing F7 ignored status.
        if subprocess.run(["git", "check-ignore", "-q", str(path.relative_to(ROOT))], cwd=ROOT).returncode:
            raise ValueError("Raw financial cache must be ignored by Git")
        meta_path = path.with_suffix(".source.json")
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"Missing {path.relative_to(ROOT)}; --download explicitly acquires the A sources")
            headers = {"User-Agent": "sberforecast E08b causal source audit"}
            body = None
            if kind == "key_rate":
                headers.update({"Content-Type": "text/xml; charset=utf-8", "SOAPAction": "http://web.cbr.ru/KeyRate"})
                body = ('<?xml version="1.0" encoding="utf-8"?>'
                        '<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">'
                        '<soap:Body><KeyRate xmlns="http://web.cbr.ru/">'
                        f'<fromDate>{settings["start"]}T00:00:00</fromDate>'
                        f'<ToDate>{settings["end"]}T00:00:00</ToDate>'
                        '</KeyRate></soap:Body></soap:Envelope>').encode("utf-8")
            with urlopen(Request(url, data=body, headers=headers), timeout=40) as response:
                final = urlsplit(response.geturl())
                if final.scheme != "https" or final.hostname not in {"cbr.ru", "www.cbr.ru"}:
                    raise ValueError("Unexpected nonofficial source redirect")
                content = response.read(3 * 1024 * 1024 + 1)
                if len(content) > 3 * 1024 * 1024:
                    raise ValueError("Financial XML exceeds bounded size")
                metadata = dict(source_url=url, response_url=response.geturl(),
                                content_type=response.headers.get("Content-Type"),
                                retrieved_at=datetime.now(timezone.utc).isoformat(),
                                retrieval_time_basis="HTTP_response_completed", method="POST" if body else "GET")
            path.write_bytes(content)
            metadata["sha256"] = sha256_file(path)
            save_json(meta_path, metadata)
        digest = sha256_file(path)
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if metadata["sha256"] != digest or metadata["source_url"] != url:
                raise ValueError("Financial cache hash disagrees with retrieval metadata")
        else:
            # Initial acquisition used PowerShell. File completion time is an
            # explicit approximation, never called a publication timestamp.
            metadata = dict(source_url=settings[url_key], sha256=digest,
                            retrieved_at=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                            retrieval_time_basis="initial_download_file_completion_time", method="POST" if kind == "key_rate" else "GET",
                            content_type=None)
            save_json(meta_path, metadata)
        sources.append(dict(metadata, cache_path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size))
    daily = parse_key_rate_xml((cache / settings["key_xml"]).read_bytes())
    fx = parse_usd_rub_xml((cache / settings["fx_xml"]).read_bytes())
    for frame in (daily, fx):
        if frame.effective_at.max() > origin_timestamp(settings["end"]):
            raise ValueError("Downloaded history contains observations after December 2024")
    decisions = pd.DataFrame(collect_key_decisions(cache / settings["decision_dir"], download=download))
    events = reconstruct_key_events(daily, decisions)
    return daily, fx, events, sources


def check_invariants(events: pd.DataFrame, fx: pd.DataFrame, matrix: pd.DataFrame) -> dict:
    """Actual downloaded inputs, future counterfactuals, no target values."""
    checks = {}
    for column in ("max_source_date_used", "max_source_available_at_used"):
        if not matrix[column].le(matrix.forecast_origin).all():
            raise AssertionError(f"{column} exceeds an origin")
    checks["all_operands_at_or_before_own_origin"] = True
    identical = build_financial_features(events, fx, matrix.forecast_origin)
    pd.testing.assert_frame_equal(matrix, identical, check_exact=True)
    shuffled = build_financial_features(events.sample(frac=1, random_state=42), fx.sample(frac=1, random_state=42), matrix.forecast_origin)
    pd.testing.assert_frame_equal(matrix, shuffled, check_exact=True)
    checks["deterministic_including_input_permutation"] = True
    for cutoff in matrix.forecast_origin:
        expected = build_financial_features(events, fx, [cutoff])
        changed_key, changed_fx = events.copy(), fx.copy()
        future_key = changed_key.effective_at.gt(cutoff) | changed_key.available_at.gt(cutoff)
        future_fx = changed_fx.effective_at.gt(cutoff) | changed_fx.available_at.gt(cutoff)
        changed_key.loc[future_key, "rate"] += 70
        changed_key.loc[future_key, "delta_pp"] = -33
        changed_fx.loc[future_fx, "value"] *= 11
        pd.testing.assert_frame_equal(expected, build_financial_features(changed_key, changed_fx, [cutoff]), check_exact=True)
        past_key = events.loc[events.effective_at.le(cutoff) & events.available_at.le(cutoff)]
        past_fx = fx.loc[fx.effective_at.le(cutoff) & fx.available_at.le(cutoff)]
        pd.testing.assert_frame_equal(expected, build_financial_features(past_key, past_fx, [cutoff]), check_exact=True)
    checks["future_mutation_invariance_origins"] = len(matrix)
    checks["future_removal_invariance_no_backfill_origins"] = len(matrix)
    # Technical replication check uses two synthetic MO identifiers, no panel.
    samples = pd.DataFrame([dict(municipality_id=uid, forecast_origin=cutoff)
                            for cutoff in matrix.forecast_origin for uid in ["SYNTHETIC_MO_A", "SYNTHETIC_MO_B"]])
    joined = join_financial_features(samples, matrix)
    assert joined.groupby("forecast_origin")[list(FEATURE_COLUMNS)].nunique(dropna=False).le(1).all().all()
    checks["national_features_equal_across_two_synthetic_municipalities"] = True
    june = matrix.loc[matrix.forecast_origin.dt.strftime("%Y-%m").eq("2024-06")]
    checks["june_method_boundary_in_audit"] = "not_applicable_no_june_origin" if june.empty else bool(
        june.usd_rub_method_regime.eq("bank_otc_reporting").all()
        and june.usd_rub_method_setting_boundary.eq("2024-06-13").all())
    if checks["june_method_boundary_in_audit"] is False:
        raise AssertionError("June FX methodology metadata missing")
    return checks


def manual_anchors(events: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    cases = [
        ("2022_date_only_decision_day", "2022-02-28"),
        ("2022_date_only_available_next_day", "2022-03-01"),
        ("2023_same_day_before", "2023-08-15T10:29:00+03:00"),
        ("2023_same_day_after", "2023-08-15T10:30:00+03:00"),
        ("2023_weekend_month_end_actual_origin", "2023-12-31"),
        ("2024_fx_before_announcement", "2024-06-13"),
        ("2024_fx_announcement_not_yet_effective", "2024-06-13T14:40:00+03:00"),
        ("2024_fx_first_effective", "2024-06-14"),
        ("2024_weekend_month_end_actual_origin", "2024-06-30"),
        ("2024_rate_announcement_future_effective", "2024-07-26T13:30:00+03:00"),
        ("2024_rate_effective", "2024-07-29"),
        ("2024_month_end_actual_origin", "2024-07-31"),
    ]
    rows = build_financial_features(events, fx, [stamp for _, stamp in cases])
    rows.insert(0, "anchor_case", [name for name, _ in cases])
    return rows


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    def cell(value):
        if pd.isna(value):
            return "NaN"
        if isinstance(value, (float, np.floating)):
            return f"{value:.6g}"
        return str(value)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(cell(row[c]) for c in columns) + " |" for _, row in frame.iterrows())
    return "\n".join(lines)


def write_report(cfg: dict, manifest: dict, matrix: pd.DataFrame, anchors: pd.DataFrame, coverage: pd.DataFrame) -> None:
    lines = [
        "# E08b — causal loading and origin-aware leading financial features", "",
        f"Ветка `{manifest['branch']}`, исходный HEAD `{manifest['git_commit']}`. "
        f"Seed {cfg['seed']}. Команда: `{manifest['command']}`.", "",
        "**Код подготовлен; источники загружены; матрица и автоматические audit checks выполнены.** "
        "Статус unit tests и окончательная готовность записаны в секции проверки ниже.", "",
        "## Источники и реконструкция", "",
        f"Запрошено {cfg['sources']['start']}…{cfg['sources']['end']}. "
        f"Ключевая ставка: {manifest['key_daily_observations']} daily observations "
        f"({manifest['key_daily_first']}…{manifest['key_daily_last']}), "
        f"{manifest['key_changes']} реальных изменений + initial anchor. "
        f"USD/RUB: {manifest['fx_observations']} settings "
        f"({manifest['fx_first']}…{manifest['fx_last']}). После декабря 2024 значений нет.", "",
        "Ключевая ставка сверена с архивными решениями: decision_date, announced_at, "
        "effective_date, rate, source URL/hash и точность времени сохранены отдельно в cache ledger. "
        "Effective date берётся из прямого текста либо наблюдаемого перехода официального daily history, "
        "с явным basis; понедельник не вычисляется по расписанию. Для каждого изменения требуется "
        "единственное совпадение rate/effective date; unmatched change останавливает сборку. "
        "Точный timestamp используется только при прямом подтверждении; date-only decision доступен "
        "со следующего calendar midnight. In-force level требует одновременно availability и effective≤O. "
        "Первый daily anchor допускается со следующего дня, с неизвестным prior delta; "
        "он не устанавливает дату последнего изменения. Unchanged decisions не сбрасывают last change.", "",
        f"Ledger: {manifest['decision_ledger']['decisions']} решений, "
        f"{manifest['decision_ledger']['unchanged']} unchanged; точное время "
        f"{manifest['decision_ledger']['exact_timestamps']} / date-only "
        f"{manifest['decision_ledger']['date_only']}. Среди 23 changes: "
        f"{manifest['decision_ledger']['changed_exact_timestamps']} exact / "
        f"{manifest['decision_ledger']['changed_date_only']} date-only. Effective date "
        f"{manifest['decision_ledger']['explicit_effective_dates']} изменений подтверждена прямо в core, "
        f"{manifest['decision_ledger']['observed_transition_effective_dates']} — официальным daily transition. "
        "Неподтверждённые 00:00 footer не считаются доказанным midnight publication; "
        "их причина неизвестна. Внеочередное 28.02.2022 допускается только 01.03, "
        "15.08.2023 сохраняет фактическое 10:30.", "",
        "FX Record.Date — effective date. available_at = effective midnight Europe/Moscow — "
        "консервативная верхняя граница при официальной публикации до вступления в силу, "
        "не выдуманный actual published_at. Рубли за 1 USD нормализуются через Nominal; "
        "VunitRate, дубли, неположительные значения проверяются. Holidays/weekends используют "
        "последний допустимый setting, без future backfill. Снимки архива получены сейчас; "
        "допуск A опирается на nonrevision/archive trust E08a, независимых historical SHA нет.", "",
        "[Ставка](https://www.cbr.ru/hd_base/KeyRate/), "
        "[календарь решений](https://www.cbr.ru/dkp/cal_mp/), "
        "[SOAP KeyRate](https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx?op=KeyRate), "
        "[FX XML documentation](https://www.cbr.ru/development/SXML/), "
        "[FX FAQ](https://www.cbr.ru/faq/foreign_exchange_market/), "
        "[E08a evidence/catalog](../../data/metadata/leading_financial_sources.json).", "",
        "## Origins и признаки", "",
        f"Матрица содержит {len(matrix)} существующих forecasting origins "
        f"{matrix.forecast_origin.min()}…{matrix.forecast_origin.max()}. "
        "Они получены через backtest.origins_from_config и data.period_end из неизменного "
        "configs/macro_forecast.yaml; target не читается. Cutoff — 00:00 Москвы последнего "
        "календарного дня, как в news feature layer. Intraday manual anchors ниже — проверки "
        "источников, не новые backtest origins. Горизонты/выборка/разбиение не менялись.", "",
        "| Признак | Определение |", "| --- | --- |",
        "| key_rate_level | Последняя известная действующая ставка, % |",
        "| key_rate_delta_last | Последнее известное ненулевое изменение, процентные пункты |",
        "| key_rate_change_3m / key_rate_change_6m | level(O) − level(O−3/6 календарных месяцев); нижний anchor доступен на собственную дату |",
        "| months_since_rate_change | Разница календарных month indices O и последнего effective change; не число дней/30 |",
        "| usd_rub_last | Последний действующий/допустимый курс, RUB за 1 USD |",
        "| usd_rub_change_1m / usd_rub_change_3m | ln(last(O)/last(O−1/3 календарных месяцев)); без умножения на 100 |",
        "| usd_rub_vol_1m / usd_rub_vol_3m | std(ddof=1) log returns между соседними settings с ending effective date в (lower,O]; без annualisation |", "",
        "Для change окон нужны оба causal endpoints; для volatility дополнительно минимум 2 "
        "returns и historical anchor≤lower. Начало первого return может предшествовать lower, "
        "поскольку return датируется вторым setting. Holiday daily zeros не добавляются. "
        "При недостатке истории — NaN и отдельный <feature>_missing; замена нулём или "
        "future value отсутствует. Полнота источника предполагает полноту официального ответа: "
        "отсутствие setting не доказывает торговый день без публикации. Max staleness "
        "не придумывается; anchor dates и return counts видны в аудите. DateOffset сохраняет "
        "номер календарного дня с ограничением длиной месяца: lower для 29.02−1m=29.01, "
        "а не предыдущая monthly origin 31.01. delta_last пересчитывается из двух доступных "
        "уровней на собственную origin, не копируется из глобального precomputed delta; "
        "оба operands включены в audit.", "",
        markdown_table(coverage, ["feature", "origins", "nonmissing", "missing"]), "",
        "Эти десять национальных признаков повторяются для МО на одной origin. Дублирование "
        "по 2190 МО не добавляет независимых финансовых наблюдений. Репликация проверена "
        "на двух synthetic MO identifiers, без чтения реального target/panel.", "",
        "## Июнь 2024 и audit operands", "",
        "[Релиз ЦБ 13.06.2024 14:40](https://cbr.ru/press/pr/?id=39834) задаёт "
        "изменение установления USD/EUR с 13 июня. При правиле next-calendar-day effective "
        "граница settings 13.06 соответствует first effective 14.06; это явно отмеченный "
        "вывод из двух правил, не timestamp XML publication. До объявления boundary metadata "
        "не раскрывается. Окна, пересекающие режимы, не корректируются и не отбрасываются; "
        "сравнимость volatility через границу ограничена. 15:30 в релизе — cutoff "
        "исходных сделок, не время публикации курса.", "",
        "Аудит сохраняет last available/effective dates и timestamps для обоих источников, "
        "нижние anchors, return counts, regime и maxima effective/availability **всех "
        "использованных operands**. Требование ≤ own origin проверяется для обеих maxima.", "",
        markdown_table(matrix, ["forecast_origin", "key_rate_last_available_date", "key_rate_level",
                                "usd_rub_last_available_date", "usd_rub_last", "max_source_date_used"]), "",
        "## Ручные anchors", "",
        "Каждая строка рассчитана по сохранённым sources, не вписана вручную. Таблица "
        "показывает, какой effective setting известен на cutoff, и часть значений; "
        "в manual_anchors.csv сохранены все десять features и missing flags.", "",
        markdown_table(anchors, ["anchor_case", "forecast_origin", "key_rate_level", "key_rate_delta_last",
                                 "key_rate_change_3m", "key_rate_change_6m", "usd_rub_last",
                                 "usd_rub_change_1m", "usd_rub_vol_1m", "usd_rub_last_effective_date",
                                 "usd_rub_method_regime"]), "",
        "## Автоматические проверки", "",
        "```json", json.dumps(manifest["checks"], ensure_ascii=False, indent=2), "```", "",
        "Future mutation меняет future key levels/deltas и FX values на каждой actual origin, "
        "затем сравнивает всю feature/audit row. Дополнительно будущие строки удаляются целиком. "
        "Loader и feature builder повторно читаются/выполняются, перестановка inputs проверяется "
        "на точное совпадение. Это проверки явной утечки, не доказательство полноты historical archive.", "",
        "## Артефакты, ограничения и следующий этап", "",
        "Raw XML/HTML, ledger и нормализованные public financial snapshots остаются в ignored "
        "outputs/leading_financial_e08b_v1/. Матрица/audit и report остаются в ignored reports/results/. "
        "Code/config/tests и две project docs пригодны для review; новые raw финансовые "
        "наблюдения в Git не добавляются. Manifest содержит hashes, относительные пути, seed, "
        "версии, command, commit, dirty state и coverage. Секреты/absolute local paths не сохраняются.", "",
        "E08b не оценивает leading predictive value или causal economic effect. Ограничения "
        "цели сохраняются: лишь 24 target months, L=0 — допущение, target vintages неизвестны, "
        "holdout уже просмотрен. Национальные features не снимают feasibility gate реального "
        "early warning. Announcement features для ещё не действующей ставки отдельно не добавлены. "
        "Будущая сборка training pair должна вызывать этот builder на собственной historical r, "
        "а не переносить row общей training origin O назад.", "",
        "Следующий конкретный шаг после прохождения tests: отдельно согласовать E08c, "
        "фиксированный forecasting ablation на прежних cases с features на own r. "
        "До согласования моделей/метрик/подбора не выполнялось.", "",
        "NO MODEL FITS; NO FORECASTING METRICS; NO EARLY-WARNING TRAINING; NO COMMIT/PUSH.", "",
    ]
    path = project_path(cfg["report_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/leading_financial_e08b.yaml")
    parser.add_argument("--download", action="store_true", help="Acquire missing official A-source snapshots")
    parser.add_argument("--smoke", action="store_true", help="First two existing origins, separate output, no final report")
    args = parser.parse_args()
    if git("branch", "--show-current") != "research/e08-leading-indicators":
        raise ValueError("E08b is authorised only on research/e08-leading-indicators")
    cfg = yaml.safe_load(project_path(args.config).read_text(encoding="utf-8"))
    source_cfg = yaml.safe_load(project_path(cfg["origin_config"]).read_text(encoding="utf-8"))
    origins = [period_end(o).tz_localize("Europe/Moscow") for o in origins_from_config(source_cfg)]
    if len(origins) != cfg["expected_origins"]:
        raise ValueError("Origin count differs from the unchanged protocol")
    started = datetime.now(timezone.utc).isoformat()
    initial_dirty = git("status", "--short")
    daily, fx, events, sources = get_sources(cfg, args.download)
    ledger_path = project_path(cfg["sources"]["cache_dir"]) / cfg["sources"]["decision_dir"] / "key_decisions.json"
    decisions = json.loads(ledger_path.read_text(encoding="utf-8"))
    changes = [d for d in decisions if d["action"] != "unchanged"]
    ledger_info = dict(path=ledger_path.relative_to(ROOT).as_posix(), sha256=sha256_file(ledger_path),
                       decisions=len(decisions), unchanged=len(decisions)-len(changes),
                       exact_timestamps=sum(d["announced_at"] is not None for d in decisions),
                       date_only=sum(d["announced_at"] is None for d in decisions),
                       changed_exact_timestamps=sum(d["announced_at"] is not None for d in changes),
                       changed_date_only=sum(d["announced_at"] is None for d in changes),
                       explicit_effective_dates=sum(d["effective_date_basis"] == "explicit_in_decision_core" for d in changes),
                       observed_transition_effective_dates=sum(d["effective_date_basis"] == "observed_transition_in_official_daily_history" for d in changes))
    if args.smoke:
        origins = origins[:2]
    matrix = build_financial_features(events, fx, origins)
    # Deterministic cached parsing/reconstruction, not only deterministic features.
    daily2, fx2, events2, sources2 = get_sources(cfg, False)
    for left, right in [(daily, daily2), (fx, fx2), (events, events2)]:
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    assert sources == sources2
    checks = check_invariants(events, fx, matrix)
    checks["cached_loader_deterministic"] = True
    anchors = manual_anchors(events, fx)
    coverage = pd.DataFrame([dict(feature=name, origins=len(matrix), nonmissing=int(matrix[name].notna().sum()),
                                 missing=int(matrix[name].isna().sum())) for name in FEATURE_COLUMNS])
    output = project_path(cfg["smoke_output_dir"] if args.smoke else cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in [("financial_features_by_origin.csv", matrix), ("manual_anchors.csv", anchors),
                        ("feature_coverage.csv", coverage), ("key_rate_daily.csv", daily),
                        ("key_rate_events.csv", events), ("usd_rub_settings.csv", fx)]:
        frame.to_csv(output / name, index=False)
    (output / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    manifest = dict(stage="E08b", mode="smoke" if args.smoke else "complete_feature_audit",
                    started_at=started, completed_at=datetime.now(timezone.utc).isoformat(),
                    branch=git("branch", "--show-current"), git_commit=git("rev-parse", "HEAD"),
                    main_commit=git("rev-parse", "main"), dirty_at_start=bool(initial_dirty),
                    git_status_at_start=initial_dirty, seed=cfg["seed"],
                    command=".\\.venv\\Scripts\\python.exe -B -X utf8 scripts/build_leading_financial_features.py " + " ".join(sys.argv[1:]),
                    versions={"python": sys.version.split()[0], **{name: importlib.metadata.version(name) for name in ["pandas", "numpy", "PyYAML"]}},
                    key_daily_observations=len(daily), key_daily_first=str(daily.effective_at.min().date()),
                    key_daily_last=str(daily.effective_at.max().date()), key_changes=len(events)-1,
                    fx_observations=len(fx), fx_first=str(fx.effective_at.min().date()), fx_last=str(fx.effective_at.max().date()),
                    origins=len(matrix), source_catalog=sources, decision_ledger=ledger_info, checks=checks,
                    features=list(FEATURE_COLUMNS), no_model_fits=True, no_forecasting_metrics=True,
                    no_early_warning_training=True, no_commit_push=True)
    manifest["code_config_hashes"] = {p: sha256_file(project_path(p)) for p in [
        args.config, cfg["origin_config"], "src/sberforecast/leading_financial.py",
        "src/sberforecast/leading_financial_sources.py", "scripts/build_leading_financial_features.py"]}
    manifest["artifact_hashes"] = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in output.glob("*") if p.is_file() and p.name != "run_manifest.json"}
    save_json(output / "run_manifest.json", manifest)
    if not args.smoke:
        audit = project_path(cfg["audit_dir"])
        audit.mkdir(parents=True, exist_ok=True)
        for name, frame in [("financial_features_by_origin.csv", matrix), ("manual_anchors.csv", anchors), ("feature_coverage.csv", coverage)]:
            frame.to_csv(audit / name, index=False)
        save_json(audit / "causality_checks.json", checks)
        write_report(cfg, manifest, matrix, anchors, coverage)
    print(json.dumps(dict(mode=manifest["mode"], key_daily=len(daily), key_changes=len(events)-1, fx=len(fx),
                          origins=len(matrix), features=len(FEATURE_COLUMNS), missing=int(coverage.missing.sum()), checks=checks), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
