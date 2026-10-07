"""Synthetic, offline checks of E08b financial availability and calendar windows.

All rates, exchange rates, decisions and dates below are invented fixtures.
No historical CBR numerical observations, target values or models are used.
"""
from __future__ import annotations

from math import log

import numpy as np
import pandas as pd
import pytest

from sberforecast.leading_financial import (
    FEATURE_COLUMNS,
    build_financial_features,
    join_financial_features,
    parse_key_rate_xml,
    parse_usd_rub_xml,
    reconstruct_key_events,
)


TZ = "Europe/Moscow"


def stamp(value):
    result = pd.Timestamp(value)
    return result.tz_localize(TZ) if result.tzinfo is None else result.tz_convert(TZ)


def key_xml(rows):
    records = "".join(
        f"<KR><DT>{date}T00:00:00</DT><Rate>{rate}</Rate></KR>"
        for date, rate in rows
    )
    return (
        '<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">'
        '<soap:Body><KeyRateResponse xmlns="http://web.cbr.ru/">'
        f'<KeyRateResult><NewDataSet xmlns="">{records}</NewDataSet></KeyRateResult>'
        "</KeyRateResponse></soap:Body></soap:Envelope>"
    ).encode("utf-8")


def fx_xml(rows, *, currency_id="R01235"):
    records = "".join(
        f'<Record Date="{date}" Id="{currency_id}"><Nominal>{nominal}</Nominal>'
        f"<Value>{value}</Value></Record>"
        for date, nominal, value in rows
    )
    return (f'<ValCurs ID="{currency_id}">{records}</ValCurs>').encode("utf-8")


def events(rows):
    """Make explicit policy events; bootstrap is not a policy change."""
    result = pd.DataFrame(rows, columns=["effective_at", "available_at", "rate"])
    for column in ("effective_at", "available_at"):
        result[column] = result[column].map(stamp)
    result["announced_at"] = result["available_at"]
    result["decision_date"] = result["available_at"].dt.date.astype(str)
    result["delta_pp"] = result["rate"].diff()
    result["source_url"] = "https://example.invalid/synthetic-policy"
    return result


def fx_rows(rows):
    result = pd.DataFrame(rows, columns=["effective_at", "value"])
    result["effective_at"] = result["effective_at"].map(stamp)
    result["available_at"] = result["effective_at"]
    return result


@pytest.fixture
def policy():
    return events([
        ("2039-09-01", "2039-09-01", 2.0),
        ("2039-12-01", "2039-11-30 13:30", 3.0),
        ("2040-03-01", "2040-02-29 13:30", 5.0),
        ("2040-03-20", "2040-03-19 13:30", 5.0),
    ])


@pytest.fixture
def currency():
    return fx_rows([
        ("2039-09-29", 80.0),
        ("2039-12-29", 100.0),
        ("2040-01-04", 110.0),
        ("2040-01-08", 99.0),
        ("2040-01-16", 118.8),
        ("2040-01-31", 118.8),
        ("2040-02-03", 130.68),
        ("2040-02-10", 117.612),
        ("2040-03-30", 141.1344),
    ])


def test_feature_contract_has_only_the_ten_requested_numeric_features():
    assert FEATURE_COLUMNS == (
        "key_rate_level", "key_rate_delta_last", "key_rate_change_3m", "key_rate_change_6m",
        "months_since_rate_change", "usd_rub_last", "usd_rub_change_1m", "usd_rub_change_3m",
        "usd_rub_vol_1m", "usd_rub_vol_3m",
    )


def test_key_xml_parses_namespaced_records_sorts_and_retains_local_midnight():
    result = parse_key_rate_xml(key_xml([("2040-01-09", "3.25"), ("2040-01-04", "2.00")]))
    assert result["rate"].tolist() == [2.0, 3.25]
    assert result["effective_at"].tolist() == [stamp("2040-01-04"), stamp("2040-01-09")]
    assert str(result["effective_at"].dt.tz) == TZ


@pytest.mark.parametrize("rate", ["0", "-1", "NaN", "Infinity", "not-a-rate"])
def test_key_xml_rejects_nonpositive_nonfinite_or_malformed_rates(rate):
    with pytest.raises(ValueError):
        parse_key_rate_xml(key_xml([("2040-01-04", rate)]))


def test_key_xml_rejects_duplicate_dates_and_malformed_document():
    with pytest.raises(ValueError):
        parse_key_rate_xml(key_xml([("2040-01-04", "2"), ("2040-01-04", "2")]))
    with pytest.raises(ValueError):
        parse_key_rate_xml(b"<broken>")


def test_fx_nominal_normalization_and_decimal_comma_are_per_usd():
    result = parse_usd_rub_xml(fx_xml([
        ("09.01.2040", "10", "1250,00"),
        ("04.01.2040", "1", "100,50"),
    ]))
    assert result["value"].tolist() == [100.5, 125.0]
    assert result["effective_at"].tolist() == [stamp("2040-01-04"), stamp("2040-01-09")]
    pd.testing.assert_series_equal(result["available_at"], result["effective_at"], check_names=False)


@pytest.mark.parametrize("nominal,value", [
    ("0", "100"), ("-10", "100"), ("1", "0"), ("1", "-100"),
    ("1", "NaN"), ("1", "Infinity"), ("wrong", "100"), ("1", "wrong"),
])
def test_fx_xml_rejects_invalid_nominal_and_value(nominal, value):
    with pytest.raises(ValueError):
        parse_usd_rub_xml(fx_xml([("04.01.2040", nominal, value)]))


def test_fx_xml_rejects_other_currency_duplicate_dates_and_malformed_document():
    with pytest.raises(ValueError):
        parse_usd_rub_xml(fx_xml([("04.01.2040", "1", "100")], currency_id="SYNTHETIC_OTHER"))
    with pytest.raises(ValueError):
        parse_usd_rub_xml(fx_xml([("04.01.2040", "1", "100")] * 2))
    with pytest.raises(ValueError):
        parse_usd_rub_xml(b"<broken>")


def test_fx_xml_checks_record_currency_id_and_optional_vunit_consistency():
    wrong_record = fx_xml([("04.01.2040", "1", "100")]).replace(
        b'Id="R01235"', b'Id="SYNTHETIC_OTHER"',
    )
    with pytest.raises(ValueError):
        parse_usd_rub_xml(wrong_record)
    inconsistent = fx_xml([("04.01.2040", "10", "1000")]).replace(
        b"</Record>", b"<VunitRate>101</VunitRate></Record>",
    )
    with pytest.raises(ValueError):
        parse_usd_rub_xml(inconsistent)


@pytest.mark.parametrize("date", ["not-a-date", "31.02.2040"])
def test_fx_xml_rejects_malformed_calendar_dates(date):
    with pytest.raises(ValueError):
        parse_usd_rub_xml(fx_xml([(date, "1", "100")]))


def decision(date, effective, rate, *, announced=None, precision="exact"):
    return {
        "decision_date": date,
        "announced_at": announced or f"{date}T13:30:00+03:00",
        "time_precision": precision,
        "effective_date": effective,
        "rate": rate,
        "source_url": "https://example.invalid/synthetic-decision",
    }


def test_policy_events_reconcile_changes_and_unchanged_decision_does_not_reset_event():
    daily = parse_key_rate_xml(key_xml([
        ("2040-01-04", "2"), ("2040-01-05", "2"),
        ("2040-02-01", "3"), ("2040-02-20", "3"),
    ]))
    decisions = pd.DataFrame([
        decision("2040-01-31", "2040-02-01", 3.0),
        decision("2040-02-19", "2040-02-20", 3.0),
    ])
    result = reconstruct_key_events(daily, decisions)
    assert result["rate"].tolist() == [2.0, 3.0]
    assert pd.isna(result["delta_pp"].iloc[0])
    assert result["delta_pp"].iloc[1:].tolist() == [1.0]
    assert result["announced_at"].iloc[1] == stamp("2040-01-31 13:30")
    assert result["available_at"].iloc[1] == stamp("2040-02-01")
    assert result["effective_at"].iloc[1] == stamp("2040-02-01")


def test_policy_date_only_publication_is_available_next_calendar_midnight():
    daily = parse_key_rate_xml(key_xml([("2040-01-04", "2"), ("2040-03-31", "4")]))
    row = decision("2040-03-31", "2040-03-31", 4.0, precision="date")
    row["announced_at"] = None
    result = reconstruct_key_events(daily, pd.DataFrame([row]))
    assert result["available_at"].iloc[-1] == stamp("2040-04-01")
    assert pd.isna(result["announced_at"].iloc[-1])


@pytest.mark.parametrize("change", ["missing", "wrong_rate", "wrong_effective"])
def test_unreconciled_daily_policy_changes_are_explicit_errors(change):
    daily = parse_key_rate_xml(key_xml([("2040-01-04", "2"), ("2040-02-01", "3")]))
    row = decision("2040-01-31", "2040-02-01", 3.0)
    if change == "wrong_rate":
        row["rate"] = 4.0
    if change == "wrong_effective":
        row["effective_date"] = "2040-02-02"
    decisions = pd.DataFrame([row]).iloc[:0] if change == "missing" else pd.DataFrame([row])
    with pytest.raises(ValueError):
        reconstruct_key_events(daily, decisions)


def test_calendar_policy_levels_changes_and_months_ignore_unchanged_decision(policy, currency):
    result = build_financial_features(policy, currency, ["2040-03-31", "2040-04-30"])
    assert result["key_rate_level"].tolist() == [5.0, 5.0]
    assert result["key_rate_delta_last"].tolist() == [2.0, 2.0]
    assert result["key_rate_change_3m"].tolist() == [2.0, 2.0]
    assert result["key_rate_change_6m"].tolist() == [3.0, 3.0]
    assert result["months_since_rate_change"].tolist() == [0.0, 1.0]


def test_policy_last_change_requires_a_previous_level_available_at_its_own_origin(currency):
    delayed = events([
        ("2040-01-01", "2040-03-01", 2.0),
        ("2040-02-01", "2040-02-01", 5.0),
    ])
    result = build_financial_features(delayed, currency, ["2040-02-15"])
    row = result.iloc[0]
    assert row["key_rate_level"] == 5.0
    assert pd.isna(row["key_rate_delta_last"])
    assert pd.isna(row["months_since_rate_change"])
    assert row["key_rate_delta_last_missing"]
    assert row["months_since_rate_change_missing"]
    assert row["max_source_available_at_used"] <= row["forecast_origin"]

    # An unavailable predecessor cannot establish the change or its magnitude.
    mutated = delayed.copy(deep=True)
    mutated.loc[0, "rate"] = 999.0
    mutated.loc[1, "delta_pp"] = -994.0
    pd.testing.assert_frame_equal(
        result, build_financial_features(mutated, currency, ["2040-02-15"]),
    )


def test_policy_last_change_recomputes_difference_instead_of_trusting_cached_delta(policy, currency):
    corrupted = policy.copy(deep=True)
    corrupted["delta_pp"] = 999.0
    expected = build_financial_features(policy, currency, ["2040-03-31"])
    actual = build_financial_features(corrupted, currency, ["2040-03-31"])
    assert actual["key_rate_delta_last"].iloc[0] == 2.0
    pd.testing.assert_frame_equal(actual, expected)


def test_policy_last_change_audits_availability_of_both_rate_operands():
    policy = events([
        ("2040-01-01", "2040-02-10", 2.0),
        ("2040-02-01", "2040-02-02", 5.0),
    ])
    empty_currency = pd.DataFrame(columns=["effective_at", "available_at", "value"])
    row = build_financial_features(policy, empty_currency, ["2040-02-15"]).iloc[0]
    assert row["key_rate_delta_last"] == 3.0
    assert row["months_since_rate_change"] == 0.0
    assert row["key_rate_last_available_at"] == stamp("2040-02-02")
    assert row["max_source_available_at_used"] == stamp("2040-02-10")


def test_policy_uses_both_announcement_and_in_force_cutoff(currency):
    policy = events([
        ("2040-01-01", "2040-01-01", 2.0),
        ("2040-03-31", "2040-03-31 13:30", 4.0),
        ("2040-04-02", "2040-04-01 13:30", 8.0),
    ])
    result = build_financial_features(policy, currency, [
        "2040-03-31", stamp("2040-03-31 13:30"), stamp("2040-04-01 18:00"), "2040-04-02",
    ])
    assert result["key_rate_level"].tolist() == [2.0, 4.0, 4.0, 8.0]


def test_calendar_key_anchors_use_their_own_information_cutoff(currency):
    policy = events([
        ("2039-08-01", "2039-08-01", 2.0),
        ("2039-12-31", "2040-01-02", 3.0),
        ("2040-03-01", "2040-02-29 13:30", 5.0),
    ])
    row = build_financial_features(policy, currency, ["2040-03-31"]).iloc[0]
    # The December event was still unavailable at the December 31 anchor.
    assert row["key_rate_change_3m"] == 3.0


def test_bootstrap_does_not_imply_a_known_previous_policy_change(currency):
    daily = parse_key_rate_xml(key_xml([("2039-09-01", "2"), ("2040-03-31", "2")]))
    decisions = pd.DataFrame(columns=[
        "decision_date", "announced_at", "effective_date", "rate", "source_url",
    ])
    policy = reconstruct_key_events(daily, decisions)
    row = build_financial_features(policy, currency, ["2040-03-31"]).iloc[0]
    assert row["key_rate_level"] == 2.0
    assert pd.isna(row["key_rate_delta_last"])
    assert pd.isna(row["months_since_rate_change"])
    assert row["key_rate_change_3m"] == 0.0
    assert row["key_rate_change_6m"] == 0.0


def test_short_history_never_backfills_missing_policy_or_fx_anchors():
    policy = events([("2040-03-01", "2040-03-01", 2.0)])
    currency = fx_rows([("2040-03-15", 100.0), ("2040-03-20", 110.0)])
    result = build_financial_features(policy, currency, ["2040-02-29", "2040-03-31"])
    assert result.loc[0, list(FEATURE_COLUMNS)].isna().all()
    assert result[[f"{name}_missing" for name in FEATURE_COLUMNS]].iloc[0].all()
    assert result[[f"{name}_missing" for name in FEATURE_COLUMNS]].iloc[1].any()
    assert result["key_rate_delta_last"].isna().all()
    assert result["months_since_rate_change"].isna().all()
    for column in ("key_rate_change_3m", "key_rate_change_6m", "usd_rub_change_1m",
                   "usd_rub_change_3m", "usd_rub_vol_1m", "usd_rub_vol_3m"):
        assert result[column].isna().all()


def test_fx_calendar_log_changes_and_original_observation_volatility(policy, currency):
    row = build_financial_features(policy, currency, ["2040-03-31"]).iloc[0]
    assert row["usd_rub_last"] == pytest.approx(141.1344)
    assert row["usd_rub_change_1m"] == pytest.approx(log(1.2))
    assert row["usd_rub_change_3m"] == pytest.approx(log(141.1344 / 100.0))
    observed_returns = [log(1.1), log(0.9), log(1.2), 0.0, log(1.1), log(0.9), log(1.2)]
    assert row["usd_rub_vol_3m"] == pytest.approx(np.std(observed_returns, ddof=1))
    # One original March return is insufficient; weekends are not zero-return copies.
    assert pd.isna(row["usd_rub_vol_1m"])


def test_two_original_returns_give_sample_std_without_annualisation(policy):
    currency = fx_rows([
        ("2040-02-28", 100.0), ("2040-03-04", 110.0), ("2040-03-28", 99.0),
    ])
    row = build_financial_features(policy, currency, ["2040-03-31"]).iloc[0]
    assert row["usd_rub_vol_1m"] == pytest.approx(np.std([log(1.1), log(0.9)], ddof=1))


def test_weekends_and_a_synthetic_holiday_keep_last_setting_without_new_observations(policy):
    friday = pd.date_range("2040-03-01", periods=1, freq="W-FRI", tz=TZ)[0]
    currency = fx_rows([
        (friday - pd.Timedelta(days=1), 100.0),
        (friday, 110.0),
        (friday + pd.Timedelta(days=5), 200.0),
    ])
    # Saturday, Sunday, and a made-up Monday/Tuesday publication holiday.
    origins = [friday + pd.Timedelta(days=offset) for offset in (1, 2, 3, 4)]
    result = build_financial_features(policy, currency, origins)
    assert result["usd_rub_last"].tolist() == [110.0] * 4
    assert result["usd_rub_last_effective_date"].tolist() == [friday] * 4
    assert result["usd_rub_return_count_1m"].tolist() == [0] * 4


def test_calendar_month_lower_boundary_clamps_to_leap_february(policy):
    currency = fx_rows([
        ("2040-02-28", 100.0), ("2040-02-29", 200.0), ("2040-03-30", 300.0),
    ])
    row = build_financial_features(policy, currency, ["2040-03-31"]).iloc[0]
    assert row["usd_rub_anchor_1m_date"] == stamp("2040-02-29")
    assert row["usd_rub_change_1m"] == pytest.approx(log(1.5))


def test_fx_log_change_anchor_is_selected_as_of_lower_calendar_cutoff(policy):
    currency = fx_rows([
        ("2040-01-31", 100.0), ("2040-02-28", 200.0), ("2040-03-30", 300.0),
    ])
    currency.loc[1, "available_at"] = stamp("2040-03-02")
    row = build_financial_features(policy, currency, ["2040-03-31"]).iloc[0]
    assert row["usd_rub_change_1m"] == pytest.approx(log(3.0))
    assert row["usd_rub_anchor_1m_date"] == stamp("2040-01-31")


def test_monthly_period_origin_uses_last_calendar_day_midnight(policy, currency):
    expected = build_financial_features(policy, currency, ["2040-03-31"])
    for origin in (pd.Period("2040-03", freq="M"), "2040-03"):
        actual = build_financial_features(policy, currency, [origin])
        pd.testing.assert_frame_equal(actual, expected)


def test_date_only_origin_midnight_and_aware_instants_are_equivalent(policy, currency):
    expected = build_financial_features(policy, currency, ["2040-03-31"])
    for equivalent in (stamp("2040-03-31"), pd.Timestamp("2040-03-30T21:00:00Z")):
        actual = build_financial_features(policy, currency, [equivalent])
        pd.testing.assert_frame_equal(actual, expected)


def test_duplicate_origins_are_rejected_after_timezone_normalisation(policy, currency):
    with pytest.raises(ValueError):
        build_financial_features(policy, currency, ["2040-03-31", pd.Timestamp("2040-03-30T21:00:00Z")])


def test_future_appends_and_mutations_leave_past_features_and_audit_unchanged(policy, currency):
    origin = ["2040-03-31"]
    before = build_financial_features(policy, currency, origin)
    future_policy = events([("2040-06-01", "2040-05-31 13:30", 40.0)])
    future_policy["delta_pp"] = 35.0
    future_fx = fx_rows([("2040-04-01", 10000.0), ("2040-04-02", 1.0)])
    extended_policy = pd.concat([policy, future_policy], ignore_index=True)
    extended_fx = pd.concat([currency, future_fx], ignore_index=True)
    pd.testing.assert_frame_equal(before, build_financial_features(extended_policy, extended_fx, origin))
    extended_policy.loc[extended_policy["effective_at"] > stamp(origin[0]), "rate"] = 400.0
    extended_fx.loc[extended_fx["effective_at"] > stamp(origin[0]), "value"] = 50000.0
    pd.testing.assert_frame_equal(before, build_financial_features(extended_policy, extended_fx, origin))


def test_unavailable_future_publication_cannot_change_past_fx_features(policy, currency):
    before = build_financial_features(policy, currency, ["2040-03-31"])
    delayed = fx_rows([("2040-03-10", 10000.0)])
    delayed["available_at"] = stamp("2040-04-01")
    extended = pd.concat([currency, delayed], ignore_index=True)
    pd.testing.assert_frame_equal(before, build_financial_features(policy, extended, ["2040-03-31"]))


def test_audit_source_maxima_never_exceed_each_rows_own_origin(policy, currency):
    result = build_financial_features(policy, currency, ["2040-02-29", "2040-03-31"])
    for column in ("max_source_date_used", "max_source_available_at_used",
                   "key_rate_last_available_at", "usd_rub_last_available_at"):
        assert (result[column] <= result["forecast_origin"]).all()
    for name in FEATURE_COLUMNS:
        assert result[f"{name}_missing"].tolist() == result[name].isna().tolist()


def test_empty_sources_preserve_origins_and_explicit_feature_missing_flags():
    policy = pd.DataFrame(columns=["effective_at", "available_at", "rate", "delta_pp"])
    currency = pd.DataFrame(columns=["effective_at", "available_at", "value"])
    result = build_financial_features(policy, currency, ["2040-03-31"])
    assert len(result) == 1
    assert result.loc[0, list(FEATURE_COLUMNS)].isna().all()
    assert result.loc[0, [f"{name}_missing" for name in FEATURE_COLUMNS]].all()


def test_fx_method_metadata_is_not_known_before_announcement_or_active_before_effective():
    policy = events([("2024-01-01", "2024-01-01", 2.0)])
    currency = fx_rows([("2024-06-12", 100.0), ("2024-06-13", 110.0), ("2024-06-14", 120.0)])
    result = build_financial_features(policy, currency, [
        "2024-06-13", stamp("2024-06-13 14:39"), stamp("2024-06-13 14:40"), "2024-06-14",
    ])
    assert result["usd_rub_method_change_known"].tolist() == [False, False, True, True]
    assert result["usd_rub_method_setting_boundary"].tolist() == [None, None, "2024-06-13", "2024-06-13"]
    assert result["usd_rub_method_regime"].tolist() == [
        "pre_june_2024_setting_method", "pre_june_2024_setting_method",
        "pre_june_2024_setting_method", "bank_otc_reporting",
    ]


def test_deterministic_results_with_unsorted_source_observations(policy, currency):
    origins = ["2040-03-31", "2040-04-30"]
    expected = build_financial_features(policy, currency, origins)
    actual = build_financial_features(
        policy.sample(frac=1, random_state=9), currency.sample(frac=1, random_state=3), origins,
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_national_join_matches_each_samples_own_origin_and_preserves_rows(policy, currency):
    matrix = build_financial_features(policy, currency, ["2040-03-31", "2040-04-30"])
    samples = pd.DataFrame({
        "municipality_id": ["synthetic-a", "synthetic-b", "synthetic-a"],
        "forecast_origin": ["2040-03-31", "2040-03-31", "2040-04-30"],
        "sample_id": ["sample-1", "sample-2", "sample-3"],
    })
    joined = join_financial_features(samples, matrix)
    assert len(joined) == len(samples)
    assert joined["sample_id"].tolist() == samples["sample_id"].tolist()
    pd.testing.assert_series_equal(joined.loc[0, list(FEATURE_COLUMNS)],
                                   joined.loc[1, list(FEATURE_COLUMNS)], check_names=False)
    assert joined["months_since_rate_change"].tolist() == [0.0, 0.0, 1.0]


def test_national_join_normalises_equivalent_timezone_origins(policy, currency):
    matrix = build_financial_features(policy, currency, ["2040-03-31"])
    samples = pd.DataFrame({
        "municipality_id": ["synthetic-a", "synthetic-b"],
        "forecast_origin": ["2040-03-31", pd.Timestamp("2040-03-30T21:00:00Z")],
    })
    joined = join_financial_features(samples, matrix)
    assert joined["key_rate_level"].tolist() == [5.0, 5.0]


def test_national_join_rejects_duplicated_matrix_origins_instead_of_multiplying_rows(policy, currency):
    matrix = build_financial_features(policy, currency, ["2040-03-31"])
    samples = pd.DataFrame({"municipality_id": ["synthetic-a"], "forecast_origin": ["2040-03-31"]})
    with pytest.raises(ValueError, match="unique origins"):
        join_financial_features(samples, pd.concat([matrix, matrix], ignore_index=True))


def test_uncovered_sample_origin_is_not_backfilled_from_future_matrix(policy, currency):
    matrix = build_financial_features(policy, currency, ["2040-03-31"])
    samples = pd.DataFrame({"municipality_id": ["synthetic-a"], "forecast_origin": ["2040-02-29"]})
    with pytest.raises(ValueError, match="Missing national feature origin"):
        join_financial_features(samples, matrix)
