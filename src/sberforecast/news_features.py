"""Small point-in-time news features for E06b; no prediction or shock labels.

Windows are (O - timedelta(days=n), O], on version ``available_at`` in UTC.
Naive forecast origins (including E01 month-end dates) mean Moscow local time;
an E01 date therefore means midnight, not the end of that day. Normalized
document timestamps mean UTC. Counts describe the collected corpus, not all
events: absence of a document does not establish absence of an event.

Deduplication is performed AFTER the origin cutoff, BEFORE geography/windows.
The first available version of each document and then the first available
document of each canonical event supplies its fixed labels and availability.
Later copies do not create another event or rewrite its first known labels.
Global ``duplicate_count`` is audit metadata and is never used in features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = (
    "news_count_7d", "news_count_30d", "news_count_90d",
    "negative_count_30d", "positive_count_30d", "emergency_count_30d",
    "regulation_count_30d", "macro_policy_count_30d", "income_wages_count_30d",
    "prices_inflation_count_30d", "retail_consumption_count_30d",
    "regional_news_count_30d", "national_news_count_30d", "municipal_news_count_30d",
    "unique_event_count_30d", "duplicate_share_30d", "news_intensity_ratio_30d_90d",
    "negative_share_30d", "days_since_last_negative_event",
    "days_since_last_emergency", "days_since_last_regulation_event",
    "news_count_change_30d_vs_prev30d", "negative_count_change_30d_vs_prev30d",
    "emergency_count_change_30d_vs_prev30d", "topic_entropy_30d", "topic_entropy_change",
)
MISSING_COLUMNS = tuple(f"{column}_missing" for column in FEATURE_COLUMNS)
EVENT_COLUMNS = (
    "document_id", "canonical_event_id", "published_at", "available_at",
    "geography_level", "region_id", "municipality_id", "topic", "event_type",
    "direction", "availability_status",
)
SAMPLE_COLUMNS = ("municipality_id", "region_id", "forecast_origin")
DAY = pd.Timedelta(days=1)


def _identifier(value: object) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return ""
    text = str(value).strip()
    if text.isdigit():
        return str(int(text))
    if text.endswith(".0") and text[:-2].isdigit():
        return str(int(text[:-2]))
    return text


def _origin(value: object) -> pd.Timestamp:
    if isinstance(value, pd.Period):
        value = value.end_time.normalize()
    elif isinstance(value, str) and len(value.strip()) == 7:
        value = pd.Period(value.strip(), freq="M").end_time.normalize()
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("forecast_origin must not be missing")
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("Europe/Moscow")
    return stamp.tz_convert("UTC")


def _utc(values: pd.Series) -> pd.Series:
    # Inputs are normalized UTC timestamps; mixed accepts offsets and blanks.
    return pd.to_datetime(values.replace("", pd.NaT), utc=True, errors="raise", format="mixed")


def _validate_samples(samples: pd.DataFrame, *, region_required: bool = True) -> pd.DataFrame:
    required = set(SAMPLE_COLUMNS if region_required else ("municipality_id", "forecast_origin"))
    if not required.issubset(samples):
        raise ValueError(f"Missing sample columns: {sorted(required - set(samples))}")
    queries = samples.copy()
    queries["_municipality"] = queries["municipality_id"].map(_identifier)
    if queries["_municipality"].eq("").any():
        raise ValueError("Sample municipality_id must not be missing")
    if region_required:
        queries["_region"] = queries["region_id"].map(_identifier)
        if queries["_region"].eq("").any():
            raise ValueError("Sample region_id must not be missing")
        if queries.groupby("_municipality")["_region"].nunique().gt(1).any():
            raise ValueError("Ambiguous municipality-to-region sample mapping")
    queries["_origin"] = queries["forecast_origin"].map(_origin)
    return queries


def _validate_events(events: pd.DataFrame) -> pd.DataFrame:
    if not set(EVENT_COLUMNS).issubset(events):
        raise ValueError(f"Missing event columns: {sorted(set(EVENT_COLUMNS) - set(events))}")
    result = events.copy()
    for column in ("document_id", "canonical_event_id"):
        if result[column].isna().any() or result[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing event identity: {column}")
        result[column] = result[column].astype(str)
    for column in ("published_at", "available_at"):
        result[column] = _utc(result[column])
    dated = result["published_at"].notna() & result["available_at"].notna()
    if result.loc[dated, "published_at"].gt(result.loc[dated, "available_at"]).any():
        raise ValueError("A document version cannot be available before publication")
    result["_municipality"] = result["municipality_id"].map(_identifier)
    result["_region"] = result["region_id"].map(_identifier)
    for column in ("geography_level", "topic", "event_type", "direction"):
        result[column] = result[column].fillna("unknown").astype(str)
    # Unknown time/status is not made historically available by imputation.
    result = result.loc[dated].copy()
    region = result["geography_level"].eq("region")
    municipal = result["geography_level"].eq("municipality")
    if result.loc[region | municipal, "_region"].eq("").any():
        raise ValueError("Regional and municipal events require an explicit region_id")
    if result.loc[municipal, "_municipality"].eq("").any():
        raise ValueError("Municipal events require an explicit municipality_id")
    return result


def _known_events(events: pd.DataFrame, origin: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lock first snapshots without allowing later global dedup metadata in."""
    known = events.loc[events["available_at"].le(origin)].copy()
    # Stable ties use document identity and the complete feature-defining labels.
    sort = ["available_at", "published_at", "document_id", "canonical_event_id",
            "geography_level", "_region", "_municipality", "topic", "event_type", "direction"]
    known = known.sort_values(sort, kind="stable").drop_duplicates("document_id", keep="first")
    canonical = known.drop_duplicates("canonical_event_id", keep="first").copy()
    # A reprint's own geographic guesses cannot move the original event.
    labels = canonical.set_index("canonical_event_id")[["geography_level", "_region", "_municipality"]]
    for column in labels:
        known[column] = known["canonical_event_id"].map(labels[column])
    return canonical, known


def _scope(table: pd.DataFrame, municipality: str, region: str) -> pd.DataFrame:
    selected = table["geography_level"].eq("national")
    selected |= table["geography_level"].eq("region") & table["_region"].eq(region)
    selected |= (table["geography_level"].eq("municipality") &
                 table["_region"].eq(region) & table["_municipality"].eq(municipality))
    return table.loc[selected]


def _window(table: pd.DataFrame, origin: pd.Timestamp, days: int, offset: int = 0) -> pd.DataFrame:
    end = origin - offset * DAY
    return table.loc[table["available_at"].gt(end - days * DAY) & table["available_at"].le(end)]


def _emergency(table: pd.DataFrame) -> pd.Series:
    return table["event_type"].eq("emergency") | table["topic"].eq("disaster_emergency")


def _entropy(table: pd.DataFrame) -> float:
    if table.empty:
        return np.nan
    probability = table["topic"].value_counts(normalize=True).to_numpy()
    return float(-(probability * np.log(probability)).sum())


def _feature_row(events: pd.DataFrame, documents: pd.DataFrame, origin: pd.Timestamp,
                 history_start: pd.Timestamp) -> dict[str, float]:
    values = dict.fromkeys(FEATURE_COLUMNS, np.nan)
    history = events.loc[events["available_at"].ge(history_start)]
    current = _window(history, origin, 30)
    previous = _window(history, origin, 30, offset=30)
    full = lambda days: origin - days * DAY >= history_start
    for days in (7, 30, 90):
        if full(days):
            values[f"news_count_{days}d"] = float(len(_window(history, origin, days)))
    if full(30):
        values.update(
            negative_count_30d=float(current["direction"].eq("negative").sum()),
            positive_count_30d=float(current["direction"].eq("positive").sum()),
            emergency_count_30d=float(_emergency(current).sum()),
            unique_event_count_30d=float(len(current)),
            topic_entropy_30d=_entropy(current),
        )
        for topic in ("regulation", "macro_policy", "income_wages", "prices_inflation", "retail_consumption"):
            values[f"{topic}_count_30d"] = float(current["topic"].eq(topic).sum())
        for feature, scope in (("regional", "region"), ("national", "national"), ("municipal", "municipality")):
            values[f"{feature}_news_count_30d"] = float(current["geography_level"].eq(scope).sum())
        raw_n = len(_window(documents, origin, 30))
        if raw_n:
            values["duplicate_share_30d"] = float((raw_n - len(current)) / raw_n)
        if len(current):
            values["negative_share_30d"] = values["negative_count_30d"] / len(current)
    if full(90) and values["news_count_90d"] > 0:
        values["news_intensity_ratio_30d_90d"] = 3 * values["news_count_30d"] / values["news_count_90d"]
    if full(60):
        values.update(
            news_count_change_30d_vs_prev30d=float(len(current) - len(previous)),
            negative_count_change_30d_vs_prev30d=float(current["direction"].eq("negative").sum() - previous["direction"].eq("negative").sum()),
            emergency_count_change_30d_vs_prev30d=float(_emergency(current).sum() - _emergency(previous).sum()),
        )
        old_entropy = _entropy(previous)
        if np.isfinite(values["topic_entropy_30d"]) and np.isfinite(old_entropy):
            values["topic_entropy_change"] = values["topic_entropy_30d"] - old_entropy
    for feature, mask in (
        ("days_since_last_negative_event", history["direction"].eq("negative")),
        ("days_since_last_emergency", _emergency(history)),
        ("days_since_last_regulation_event", history["topic"].eq("regulation")),
    ):
        if mask.any():
            values[feature] = float((origin - history.loc[mask, "available_at"].max()) / DAY)
    return values


def build_news_features(events: pd.DataFrame, samples: pd.DataFrame,
                        history_start: object = "2023-01-01") -> pd.DataFrame:
    """Keep every sample row; append 26 numeric features and 26 missing flags.

    ``history_start`` is the declared start of corpus collection, independent of
    the first observed document. Incomplete windows yield NaN. Complete empty
    windows yield zero counts; shares, entropy and unseen-event recency remain
    NaN. Corpus/source coverage gaps are audited separately. No backfill occurs.
    """
    queries = _validate_samples(samples)
    documents = _validate_events(events)
    start = _origin(history_start)
    collisions = set(FEATURE_COLUMNS + MISSING_COLUMNS) & set(samples)
    if collisions:
        raise ValueError(f"Samples already contain news features: {sorted(collisions)}")
    by_origin = {}
    cache = {}
    rows = []
    for query in queries.to_dict("records"):
        origin = query["_origin"]
        if origin not in by_origin:
            canonical, known_documents = _known_events(documents, origin)
            levels = set(canonical["geography_level"])
            by_origin[origin] = (canonical, known_documents, levels)
        canonical, known_documents, levels = by_origin[origin]
        # National/empty corpora have identical rows within an origin; regional
        # corpora have identical rows inside a region until a municipal event.
        key = (query["_municipality"] if "municipality" in levels else "",
               query["_region"] if levels & {"region", "municipality"} else "", origin)
        if key not in cache:
            cache[key] = _feature_row(
                _scope(canonical, query["_municipality"], query["_region"]),
                _scope(known_documents, query["_municipality"], query["_region"]), origin, start,
            )
        rows.append(cache[key])
    result = samples.copy()
    features = pd.DataFrame(rows, columns=FEATURE_COLUMNS, index=samples.index, dtype=float)
    for column in FEATURE_COLUMNS:
        result[column] = features[column]
        result[f"{column}_missing"] = features[column].isna()
    return result


def safe_asof_merge(samples: pd.DataFrame, states: pd.DataFrame,
                    prefix: str = "detector_") -> pd.DataFrame:
    """Append the latest explicitly available ONLINE state for each municipality.

    States must be wide (one row per municipality/available_at); independently
    named detector fields should be pivoted upstream. An observation period or
    historical offline breakpoint date never substitutes for ``available_at``.
    Offline/hindsight rows and backdated dependency timestamps are rejected.
    """
    queries = _validate_samples(samples, region_required=False)
    required = {"municipality_id", "available_at"}
    if not required.issubset(states):
        raise ValueError(f"Missing detector state columns: {sorted(required - set(states))}")
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("A nonempty detector column prefix is required")
    data = states.copy()
    data["_municipality"] = data["municipality_id"].map(_identifier)
    if data["_municipality"].eq("").any():
        raise ValueError("Detector municipality_id must not be missing")
    data["available_at"] = _utc(data["available_at"])
    if data["available_at"].isna().any():
        raise ValueError("Detector states require explicit available_at")
    for column in ("method", "mode", "detector_type", "analysis_type"):
        if column in data and data[column].astype(str).str.lower().isin(
            ["pelt", "binseg", "binarysegmentation", "offline", "retrospective", "hindsight"]
        ).any():
            raise ValueError("Offline segmentation cannot be merged as an online detector state")
    if "is_hindsight_diagnostic" in data:
        hindsight = data["is_hindsight_diagnostic"].astype(str).str.lower().isin(["true", "1"])
        if hindsight.any():
            raise ValueError("Hindsight diagnostics cannot be merged as online detector states")
    if "future_access" in data:
        offline_access = data["future_access"].astype(str).str.lower().str.contains("whole|full_series|future", regex=True)
        if offline_access.any():
            raise ValueError("Offline whole-series future access is not an online detector state")
    for column in ("analyzed_available_date", "analysis_available_at", "max_input_available_at", "state_computed_at", "signal_date"):
        if column in data:
            dependencies = _utc(data[column])
            if (dependencies.notna() & dependencies.gt(data["available_at"])).any():
                raise ValueError(f"Detector available_at precedes its dependency: {column}")
    if data.duplicated(["_municipality", "available_at"]).any():
        raise ValueError("Ambiguous detector states; pivot methods before the as-of join")
    columns = [column for column in states if column != "municipality_id"]
    names = [prefix + column for column in columns] + [prefix + "missing"]
    if len(names) != len(set(names)) or set(names) & set(samples):
        raise ValueError("Detector output columns collide with sample columns")
    rows = []
    by_municipality = {key: sub.sort_values("available_at") for key, sub in data.groupby("_municipality")}
    for query in queries.to_dict("records"):
        known = by_municipality.get(query["_municipality"])
        known = known.loc[known["available_at"].le(query["_origin"])] if known is not None else None
        row = {prefix + column: np.nan for column in columns}
        row[prefix + "missing"] = known is None or known.empty
        if not row[prefix + "missing"]:
            latest = known.iloc[-1]
            row.update({prefix + column: latest[column] for column in columns})
        rows.append(row)
    appended = pd.DataFrame(rows, columns=names, index=samples.index)
    result = samples.copy()
    for column in names:
        result[column] = appended[column]
    result[prefix + "missing"] = result[prefix + "missing"].astype(bool)
    return result


def feature_dictionary() -> pd.DataFrame:
    """Definitions for every value and missing flag, suitable for a saved CSV."""
    common = "First available document/version per canonical_event_id after published_at <= available_at <= O; no global duplicate_count, event_date or future labels."
    definitions = {
        **{f"news_count_{days}d": (f"Number of distinct first-known canonical events in {days} trailing days", days) for days in (7, 30, 90)},
        **{f"{label}_count_30d": (f"Distinct canonical events with direction={label}", 30) for label in ("negative", "positive")},
        "emergency_count_30d": ("Distinct events with event_type=emergency OR topic=disaster_emergency", 30),
        **{f"{topic}_count_30d": (f"Distinct canonical events with topic={topic}", 30) for topic in ("regulation", "macro_policy", "income_wages", "prices_inflation", "retail_consumption")},
        **{f"{label}_news_count_30d": (f"Distinct events of geography_level={scope}; no copying between scopes", 30) for label, scope in (("regional", "region"), ("national", "national"), ("municipal", "municipality"))},
        "unique_event_count_30d": ("Distinct first-known canonical events; equal to news_count_30d", 30),
        "duplicate_share_30d": ("(Distinct first-known documents arriving in 30d - canonical events first known in 30d) / documents arriving in 30d; late reprints remain duplicates", 30),
        "news_intensity_ratio_30d_90d": ("(news_count_30d / 30) / (news_count_90d / 90)", 90),
        "negative_share_30d": ("negative_count_30d / news_count_30d", 30),
        "days_since_last_negative_event": ("Elapsed 24-hour days since last available first-known event with direction=negative", None),
        "days_since_last_emergency": ("Elapsed 24-hour days since last available event_type=emergency OR topic=disaster_emergency", None),
        "days_since_last_regulation_event": ("Elapsed 24-hour days since last available first-known event with topic=regulation", None),
        "news_count_change_30d_vs_prev30d": ("Count in (O-30d,O] minus count in (O-60d,O-30d]", 60),
        "negative_count_change_30d_vs_prev30d": ("Negative event count in (O-30d,O] minus (O-60d,O-30d]", 60),
        "emergency_count_change_30d_vs_prev30d": ("Emergency event count in (O-30d,O] minus (O-60d,O-30d]", 60),
        "topic_entropy_30d": ("-sum(p_topic * ln(p_topic)) over distinct events; natural-log units, unknown/other included", 30),
        "topic_entropy_change": ("Entropy in (O-30d,O] minus entropy in (O-60d,O-30d]", 60),
    }
    fields = "document_id,canonical_event_id,published_at,available_at,availability_status,geography_level,region_id,municipality_id,topic,event_type,direction"
    scope = "Union of national, own region, own municipality, counted once; unknown geography excluded"
    rows = []
    for feature in FEATURE_COLUMNS:
        definition, days = definitions[feature]
        specific_scope = scope
        for label, level in (("regional", "region"), ("national", "national"), ("municipal", "municipality")):
            if feature == f"{label}_news_count_30d":
                specific_scope = level
        window = f"(O-{days}d,O]" if days else "[declared history_start,O]"
        missing = "NaN when the complete window predates history_start; no backfill."
        if "share" in feature or "ratio" in feature or "entropy" in feature:
            missing += " NaN when a required denominator/event distribution is empty."
        if feature.startswith("days_since"):
            missing = "NaN if no matching event has been observed since history_start (left-censored)."
        unit = "elapsed_24h_days" if feature.startswith("days_since") else "nats" if "entropy" in feature else "fraction_or_ratio" if "share" in feature or "ratio" in feature else "canonical_events"
        row = dict(feature=feature, definition=definition, window=window,
                   geographic_scope=specific_scope, source_fields=fields,
                   availability_rule=common, missing_rule=missing, unit=unit)
        rows.append(row)
        rows.append(dict(row, feature=feature + "_missing", definition=f"True exactly when {feature} is NaN", unit="boolean"))
    return pd.DataFrame(rows)
