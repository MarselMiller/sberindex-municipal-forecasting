"""E08b point-in-time national financial features; no targets or model fitting.

Naive origins mean Moscow midnight (month strings/Periods mean month end).
Every level, endpoint and volatility operand is selected on its own cutoff.
FX volatility is the sample std of log returns between published settings,
not daily filled returns, not annualised and not intraday market volatility.
"""
from __future__ import annotations

from collections.abc import Iterable
import math
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

MOSCOW = "Europe/Moscow"
FEATURE_COLUMNS = (
    "key_rate_level", "key_rate_delta_last", "key_rate_change_3m",
    "key_rate_change_6m", "months_since_rate_change", "usd_rub_last",
    "usd_rub_change_1m", "usd_rub_change_3m", "usd_rub_vol_1m", "usd_rub_vol_3m",
)
FX_METHOD_ANNOUNCED_AT = pd.Timestamp("2024-06-13T14:40:00+03:00")
# June 13 setting takes effect on June 14: keep both dates, never conflate them.
FX_METHOD_FIRST_EFFECTIVE_AT = pd.Timestamp("2024-06-14T00:00:00+03:00")


def origin_timestamp(value: object) -> pd.Timestamp:
    if isinstance(value, pd.Period):
        value = value.end_time.normalize()
    elif isinstance(value, str) and len(value.strip()) == 7:
        value = pd.Period(value.strip(), freq="M").end_time.normalize()
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Missing forecast origin")
    return stamp.tz_localize(MOSCOW) if stamp.tzinfo is None else stamp.tz_convert(MOSCOW)


def _dates(values: pd.Series) -> pd.Series:
    return pd.Series([origin_timestamp(x) for x in values], index=values.index,
                     dtype=f"datetime64[ns, {MOSCOW}]")


def _sorted_observations(frame: pd.DataFrame, value: str, *, availability: bool) -> pd.DataFrame:
    required = {"effective_at", value} | ({"available_at"} if availability else set())
    if not required.issubset(frame):
        raise ValueError(f"Missing financial columns: {sorted(required - set(frame))}")
    out = frame.copy()
    out["effective_at"] = _dates(out.effective_at)
    if availability:
        out["available_at"] = _dates(out.available_at)
    out[value] = pd.to_numeric(out[value], errors="raise")
    if (~np.isfinite(out[value].to_numpy(dtype=float))).any() or out[value].le(0).any():
        raise ValueError("Financial levels must be finite and positive")
    if out.effective_at.duplicated().any():
        raise ValueError("Duplicate financial effective date")
    return out.sort_values("effective_at", kind="stable").reset_index(drop=True)


def parse_key_rate_xml(content: bytes) -> pd.DataFrame:
    """SOAP KR/DT/Rate dates describe the rate in force, not its announcement."""
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise ValueError("Malformed key-rate XML") from exc
    records = [{"effective_at": origin_timestamp(node.findtext("DT")),
                "rate": float(node.findtext("Rate"))} for node in root.iter("KR")]
    if not records:
        raise ValueError("No KR observations in official key-rate response")
    return _sorted_observations(pd.DataFrame(records), "rate", availability=False)


def _decimal(value: str | None) -> float:
    if value is None:
        raise ValueError("Missing XML numeric field")
    return float(value.replace("\u00a0", "").replace(" ", "").replace(",", "."))


def parse_usd_rub_xml(content: bytes) -> pd.DataFrame:
    """CBR Record.Date is effective date; availability is its conservative bound."""
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise ValueError("Malformed USD/RUB XML") from exc
    if root.tag != "ValCurs" or root.get("ID") != "R01235":
        raise ValueError("Expected official USD R01235 XML")
    records = []
    for node in root.findall("Record"):
        if node.get("Id") != "R01235":
            raise ValueError("Non-USD observation in USD XML")
        stamp = pd.Timestamp(pd.to_datetime(node.get("Date"), format="%d.%m.%Y")).tz_localize(MOSCOW)
        nominal = _decimal(node.findtext("Nominal"))
        if not math.isfinite(nominal) or nominal <= 0:
            raise ValueError("Invalid FX nominal")
        value = _decimal(node.findtext("Value")) / nominal
        vunit = node.findtext("VunitRate")
        if vunit is not None and not math.isclose(value, _decimal(vunit), rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("FX Value/Nominal disagrees with VunitRate")
        records.append({"effective_at": stamp, "available_at": stamp, "value": value,
                        "published_at": pd.NaT, "availability_basis": "effective_midnight_upper_bound"})
    if not records:
        raise ValueError("No USD observations in XML")
    return _sorted_observations(pd.DataFrame(records), "value", availability=True)


def reconstruct_key_events(daily: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    """Require a release for every observed change; no weekday/13:30 guessing.

    A date-only decision is admitted from the following calendar midnight.
    The first daily observation is a bootstrap anchor with unknown prior delta,
    conservatively admitted the next day. It does not claim a last change date.
    Unchanged decisions are retained in the source ledger, not change events.
    """
    history = _sorted_observations(daily, "rate", availability=False)
    required = {"decision_date", "announced_at", "effective_date", "rate", "source_url"}
    if not required.issubset(decisions):
        raise ValueError(f"Missing decision fields: {sorted(required - set(decisions))}")
    # Unchanged releases may have no separate effective date in the core text.
    # They remain in source metadata and do not reset the last change.
    ledger = decisions.loc[decisions.effective_date.notna()].copy()
    ledger["effective_at"] = _dates(ledger.effective_date)
    ledger["rate"] = pd.to_numeric(ledger.rate, errors="raise")
    rows = []
    previous = None
    for record in history.itertuples(index=False):
        if previous is not None and record.rate == previous:
            continue
        if previous is None:
            rows.append(dict(effective_at=record.effective_at,
                             available_at=record.effective_at + pd.Timedelta(days=1),
                             announced_at=pd.NaT, decision_date=None, rate=record.rate,
                             delta_pp=np.nan, source_url="https://www.cbr.ru/hd_base/KeyRate/",
                             time_precision="bootstrap", availability_basis="first_daily_anchor_next_day"))
        else:
            match = ledger.loc[ledger.effective_at.eq(record.effective_at) & ledger.rate.eq(record.rate)]
            if len(match) != 1:
                raise ValueError(f"Unmatched/ambiguous key-rate change: {record.effective_at.date()}")
            decision = match.iloc[0]
            decision_day = origin_timestamp(decision.decision_date).normalize()
            announced = decision.announced_at
            if pd.isna(announced) or str(announced).strip() == "":
                announced = pd.NaT
                available = decision_day + pd.Timedelta(days=1)
                precision = "date"
            else:
                announced = origin_timestamp(announced)
                if announced.normalize() != decision_day:
                    raise ValueError("Announcement timestamp disagrees with decision date")
                available, precision = announced, "exact"
            if decision_day > record.effective_at.normalize():
                raise ValueError("Decision day follows effective date")
            rows.append(dict(effective_at=record.effective_at, available_at=max(available, record.effective_at),
                             announced_at=announced, decision_date=str(decision_day.date()), rate=record.rate,
                             delta_pp=record.rate - previous, source_url=decision.source_url,
                             time_precision=precision, availability_basis="max_announcement_bound_and_effective"))
        previous = record.rate
    out = _sorted_observations(pd.DataFrame(rows), "rate", availability=True)
    if not out.available_at.is_monotonic_increasing:
        raise ValueError("Key change availability must follow effective change order")
    return out


def _eligible(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    return frame.loc[frame.effective_at.le(cutoff) & frame.available_at.le(cutoff)]


def _last(frame: pd.DataFrame, cutoff: pd.Timestamp):
    prefix = _eligible(frame, cutoff)
    return None if prefix.empty else prefix.iloc[-1]


def build_financial_features(key_events: pd.DataFrame, fx: pd.DataFrame,
                             origins: Iterable[object]) -> pd.DataFrame:
    key = _sorted_observations(key_events, "rate", availability=True)
    if "delta_pp" not in key:
        raise ValueError("Key events require an explicit delta_pp (bootstrap may be NaN)")
    foreign = _sorted_observations(fx, "value", availability=True)
    if foreign.available_at.lt(foreign.effective_at).any():
        raise ValueError("FX availability must conservatively be at or after effective date")
    stamps = [origin_timestamp(o) for o in origins]
    if len(set(stamps)) != len(stamps):
        raise ValueError("Duplicate forecast origins")
    rows = []
    for cutoff in stamps:
        row = {"forecast_origin": cutoff, **{name: np.nan for name in FEATURE_COLUMNS}}
        used_dates, used_availability = [], []

        def audit(operands: pd.DataFrame) -> None:
            if not operands.empty:
                used_dates.extend(operands.effective_at.tolist())
                used_availability.extend(operands.available_at.tolist())

        kp, fp = _eligible(key, cutoff), _eligible(foreign, cutoff)
        last_key, last_fx = _last(key, cutoff), _last(foreign, cutoff)
        for prefix, last, label in [(kp, last_key, "key_rate"), (fp, last_fx, "usd_rub")]:
            row[f"{label}_last_available_date"] = pd.NaT if last is None else last.available_at.normalize()
            row[f"{label}_last_available_at"] = pd.NaT if last is None else last.available_at
            row[f"{label}_last_effective_date"] = pd.NaT if last is None else last.effective_at
            if last is not None:
                audit(prefix.iloc[[-1]])
        if last_key is not None:
            row["key_rate_level"] = float(last_key.rate)
            # Recompute on this origin's eligible prefix. A precomputed delta
            # may depend on an unavailable predecessor; it is not a feature
            # operand. The first known rate alone cannot establish a change.
            deltas = kp.rate.astype(float).diff()
            changes = kp.loc[deltas.notna() & deltas.ne(0)]
            if not changes.empty:
                change = changes.iloc[-1]
                row["key_rate_delta_last"] = float(deltas.loc[change.name])
                row["months_since_rate_change"] = (cutoff.year - change.effective_at.year) * 12 + cutoff.month - change.effective_at.month
                position = kp.index.get_loc(change.name)
                audit(kp.iloc[[position - 1, position]])
        for months in (3, 6):
            anchor = _last(key, cutoff - pd.DateOffset(months=months))
            row[f"key_rate_anchor_{months}m_date"] = pd.NaT if anchor is None else anchor.effective_at
            if last_key is not None and anchor is not None:
                row[f"key_rate_change_{months}m"] = float(last_key.rate - anchor.rate)
                audit(pd.DataFrame([anchor]))
        if last_fx is not None:
            row["usd_rub_last"] = float(last_fx.value)
        for months in (1, 3):
            lower = cutoff - pd.DateOffset(months=months)
            anchor = _last(foreign, lower)
            row[f"usd_rub_anchor_{months}m_date"] = pd.NaT if anchor is None else anchor.effective_at
            row[f"usd_rub_return_count_{months}m"] = 0
            if last_fx is None or anchor is None:
                continue
            row[f"usd_rub_change_{months}m"] = float(np.log(last_fx.value / anchor.value))
            audit(pd.DataFrame([anchor]))
            # Work only on the cutoff prefix. Adjacent settings have no invented
            # zeros on holidays; the return is dated by its second operand.
            returns = np.log(fp.value.astype(float)).diff()
            mask = fp.effective_at.gt(lower) & fp.effective_at.le(cutoff) & returns.notna()
            selected = returns.loc[mask]
            row[f"usd_rub_return_count_{months}m"] = len(selected)
            if len(selected) >= 2:
                row[f"usd_rub_vol_{months}m"] = float(selected.std(ddof=1))
                indices = fp.index[mask]
                positions = fp.index.get_indexer(indices)
                audit(fp.iloc[np.unique(np.r_[positions, positions - 1])])
        # Regime metadata is available only after its actual announcement.
        known = cutoff >= FX_METHOD_ANNOUNCED_AT
        row["usd_rub_method_change_known"] = known
        row["usd_rub_method_regime"] = (
            "bank_otc_reporting" if known and last_fx is not None and last_fx.effective_at >= FX_METHOD_FIRST_EFFECTIVE_AT
            else "pre_june_2024_setting_method" if last_fx is not None else "missing")
        row["usd_rub_method_setting_boundary"] = "2024-06-13" if known else None
        row["usd_rub_method_first_effective_date"] = "2024-06-14" if known else None
        for months in (1, 3):
            anchor_date = row[f"usd_rub_anchor_{months}m_date"]
            row[f"usd_rub_window_crosses_method_{months}m"] = bool(
                known and pd.notna(anchor_date) and anchor_date < FX_METHOD_FIRST_EFFECTIVE_AT
                and last_fx is not None and last_fx.effective_at >= FX_METHOD_FIRST_EFFECTIVE_AT)
        if known:
            used_dates.append(FX_METHOD_ANNOUNCED_AT.normalize())
            used_availability.append(FX_METHOD_ANNOUNCED_AT)
        row["max_source_date_used"] = max(used_dates, default=pd.NaT)
        row["max_source_available_at_used"] = max(used_availability, default=pd.NaT)
        if any(s > cutoff for s in used_dates + used_availability):
            raise AssertionError("A financial operand is newer than its own origin")
        for name in FEATURE_COLUMNS:
            row[f"{name}_missing"] = bool(pd.isna(row[name]))
        rows.append(row)
    return pd.DataFrame(rows)


def join_financial_features(samples: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    """National features repeat across MO; this adds no independent observations."""
    if not {"municipality_id", "forecast_origin"}.issubset(samples):
        raise ValueError("Samples require municipality_id and forecast_origin")
    left, right = samples.copy(), matrix.copy()
    left["forecast_origin"] = _dates(left.forecast_origin)
    right["forecast_origin"] = _dates(right.forecast_origin)
    if right.forecast_origin.duplicated().any():
        raise ValueError("National matrix must have unique origins")
    out = left.merge(right, on="forecast_origin", how="left", validate="many_to_one", sort=False, indicator=True)
    if out["_merge"].ne("both").any():
        raise ValueError("Missing national feature origin")
    return out.drop(columns="_merge")
