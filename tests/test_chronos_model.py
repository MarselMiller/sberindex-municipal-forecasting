"""Синтетические проверки zero-shot адаптера без torch, сети и весов."""
import numpy as np
import pandas as pd
import pytest

from sberforecast.chronos_model import ChronosModel, prepare_history


class RecordingPipeline:
    def __init__(self):
        self.calls = []

    def fit(self, *args, **kwargs):
        pytest.fail("zero-shot adapter must not fit or fine-tune")

    def predict_quantiles(self, **kwargs):
        self.calls.append({
            **kwargs,
            "inputs": [values.copy() for values in kwargs["inputs"]],
        })
        steps = kwargs["prediction_length"]
        quantiles = [
            np.arange(1, steps + 1, dtype=np.float32).reshape(1, steps, 1)
            for _ in kwargs["inputs"]
        ]
        # Намеренно другой второй результат: адаптер обязан выбрать q=0.5 явно.
        point = [np.full((1, steps), 999999., dtype=np.float32) for _ in kwargs["inputs"]]
        return quantiles, point


@pytest.fixture
def pipeline():
    return RecordingPipeline()


def predict(pipeline, history, origin="2023-12", horizons=None, lag=0, batch_size=1):
    return ChronosModel(pipeline, batch_size=batch_size, context_length=8192).predict(
        history,
        pd.Period(origin, "M"),
        [1, 3, 6, 12] if horizons is None else horizons,
        lag,
    )


def assert_failed_without_reserve(result):
    assert result.forecasts.status.eq("failed").all()
    assert result.forecasts.y_pred.isna().all()
    assert result.forecasts.effective_model.eq("Chronos-2").all()
    assert result.forecasts.reason.fillna("").str.len().gt(0).all()
    assert not result.forecasts.status.str.contains("fallback", case=False).any()
    assert result.errors


def test_prepare_history_keeps_month_grid_cutoff_and_requested_ids(panel):
    original = panel.copy(deep=True)
    sparse = panel.drop(pd.to_datetime(["2023-04-01", "2023-09-01"]))
    history = prepare_history(sparse, pd.Period("2023-12", "M"), 1, ["2", "1"])
    pd.testing.assert_index_equal(history.index, pd.date_range("2023-01-01", "2023-11-01", freq="MS"))
    assert list(history.columns) == ["2", "1"]
    assert history.loc["2023-04-01"].isna().all()
    assert history.loc["2023-09-01"].isna().all()
    pd.testing.assert_frame_equal(panel, original)


@pytest.mark.parametrize("lag", [0, 1])
def test_future_changes_do_not_change_context_or_forecast(panel, pipeline, lag):
    origin = pd.Period("2023-12", "M")
    cutoff = (origin - lag).to_timestamp()
    changed = panel.copy(deep=True)
    changed.loc[changed.index > cutoff] = 1000000.
    first = prepare_history(panel, origin, lag, ["1", "2"])
    second = prepare_history(changed, origin, lag, ["1", "2"])
    pd.testing.assert_frame_equal(first, second)
    before = predict(pipeline, first, lag=lag)
    after = predict(pipeline, second, lag=lag)
    pd.testing.assert_frame_equal(before.forecasts, after.forecasts)
    pd.testing.assert_frame_equal(before.paths, after.paths)


@pytest.mark.parametrize("lag", [0, 1])
def test_calendar_horizons_and_publication_lag_use_cutoff_steps(panel, pipeline, lag):
    origin = pd.Period("2023-12", "M")
    history = prepare_history(panel, origin, lag, ["1"])
    result = predict(pipeline, history, lag=lag)
    forecasts = result.forecasts.set_index("horizon")
    assert set(forecasts.index) == {1, 3, 6, 12}
    for horizon in [1, 3, 6, 12]:
        assert forecasts.loc[horizon, "target_period"] == (origin + horizon).to_timestamp().strftime("%Y-%m-%d")
        assert forecasts.loc[horizon, "y_pred"] == lag + horizon
    assert forecasts.status.eq("native").all()
    assert forecasts.effective_model.eq("Chronos-2").all()
    assert len(result.paths) == lag + 12
    assert list(result.paths.target_period) == [
        (origin - lag + step).to_timestamp().strftime("%Y-%m-%d")
        for step in range(1, lag + 13)
    ]
    assert pipeline.calls[0]["prediction_length"] == lag + 12
    assert pipeline.calls[0]["quantile_levels"] == [0.5]
    assert not result.errors


def test_median_is_selected_in_original_scale_instead_of_second_result(panel, pipeline):
    history = prepare_history(panel, pd.Period("2023-12", "M"), 0, ["1"])
    result = predict(pipeline, history, horizons=[1, 3])
    np.testing.assert_array_equal(result.forecasts.sort_values("horizon").y_pred, [1., 3.])
    assert not result.forecasts.y_pred.eq(999999.).any()
    # Контекст около 100 рублей: прибавлять anchor или повторно нормировать выход нельзя.
    assert pipeline.calls[0]["inputs"][0][-1] == 111.


def test_interior_and_trailing_nan_preserve_calendar_positions(panel, pipeline):
    incomplete = panel.copy(deep=True)
    incomplete.loc["2023-04-01", "1"] = np.nan
    incomplete.loc["2023-12-01", "1"] = np.nan
    history = prepare_history(incomplete, pd.Period("2023-12", "M"), 0, ["1"])
    original = history.copy(deep=True)
    result = predict(pipeline, history, horizons=[1, 3, 12])
    context = pipeline.calls[0]["inputs"][0]
    assert len(context) == 12
    assert np.isnan(context[3]) and np.isnan(context[11])
    assert context[4] == 104. and context[10] == 110.
    # Последний факт — ноябрь; отсчёт прогноза всё равно идёт от cutoff=декабрь.
    row = result.forecasts.loc[result.forecasts.horizon.eq(1)].iloc[0]
    assert row.target_period == "2024-01-01" and row.y_pred == 1.
    assert result.forecasts.status.eq("native").all()
    pd.testing.assert_frame_equal(history, original)


def test_each_municipality_is_a_separate_float32_input_without_cross_learning(panel, pipeline):
    history = prepare_history(panel, pd.Period("2023-12", "M"), 0, ["1", "2"])
    result = predict(pipeline, history, horizons=[1], batch_size=2)
    contexts = [values for call in pipeline.calls for values in call["inputs"]]
    assert len(contexts) == 2
    for municipality_id, context in zip(["1", "2"], contexts):
        assert context.ndim == 1 and context.dtype == np.float32
        np.testing.assert_array_equal(context, history[municipality_id].to_numpy(dtype=np.float32))
    assert all(call["cross_learning"] is False for call in pipeline.calls)
    assert all(call["batch_size"] == 2 for call in pipeline.calls)
    assert all(call["context_length"] == 8192 for call in pipeline.calls)
    assert set(result.forecasts.municipality_id) == {"1", "2"}
    assert result.forecasts.status.eq("native").all()


def test_another_municipality_changes_do_not_change_independent_forecast(panel, pipeline, monkeypatch):
    def context_dependent(**kwargs):
        assert kwargs["cross_learning"] is False
        steps = kwargs["prediction_length"]
        quantiles = [
            (np.nanmean(values) + np.arange(1, steps + 1)).astype(np.float32).reshape(1, steps, 1)
            for values in kwargs["inputs"]
        ]
        return quantiles, [np.zeros((1, steps), dtype=np.float32) for _ in quantiles]

    monkeypatch.setattr(pipeline, "predict_quantiles", context_dependent)
    origin = pd.Period("2023-12", "M")
    first = prepare_history(panel, origin, 0, ["1", "2"])
    second = first.copy(deep=True)
    second["2"] *= 1000.
    before = predict(pipeline, first, horizons=[1, 3], batch_size=2).forecasts
    after = predict(pipeline, second, horizons=[1, 3], batch_size=2).forecasts
    pd.testing.assert_frame_equal(
        before.loc[before.municipality_id.eq("1")].reset_index(drop=True),
        after.loc[after.municipality_id.eq("1")].reset_index(drop=True),
    )
    assert not np.array_equal(
        before.loc[before.municipality_id.eq("2"), "y_pred"],
        after.loc[after.municipality_id.eq("2"), "y_pred"],
    )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_selected_forecast_is_failed_and_logged(panel, pipeline, monkeypatch, value):
    def nonfinite(**kwargs):
        # h=3 действительно запрошен; не задаём поведение для неоцениваемых шагов.
        values = np.ones((1, kwargs["prediction_length"], 1), dtype=np.float32)
        values[0, 2, 0] = value
        return [values], [np.zeros((1, kwargs["prediction_length"]), dtype=np.float32)]

    monkeypatch.setattr(pipeline, "predict_quantiles", nonfinite)
    history = prepare_history(panel, pd.Period("2023-12", "M"), 0, ["1"])
    assert_failed_without_reserve(predict(pipeline, history, horizons=[3]))


@pytest.mark.parametrize("shape", [(3, 1), (2, 3, 1), (1, 2, 1), (1, 3, 2)])
def test_malformed_prediction_shape_is_failed_without_fallback(panel, pipeline, monkeypatch, shape):
    monkeypatch.setattr(pipeline, "predict_quantiles", lambda **kwargs: ([np.ones(shape)], []))
    history = prepare_history(panel, pd.Period("2023-12", "M"), 0, ["1"])
    assert_failed_without_reserve(predict(pipeline, history, horizons=[3]))


def test_pipeline_exception_is_logged_and_does_not_replace_model(panel, pipeline, monkeypatch):
    def failed(**kwargs):
        raise RuntimeError("synthetic inference failure")

    monkeypatch.setattr(pipeline, "predict_quantiles", failed)
    history = prepare_history(panel, pd.Period("2023-12", "M"), 0, ["1"])
    result = predict(pipeline, history, horizons=[1, 3])
    assert_failed_without_reserve(result)
    assert "synthetic inference failure" in str(result.errors)


@pytest.mark.parametrize("invalid_history", ["all_missing", "infinity"])
def test_invalid_history_fails_before_pipeline_call(panel, pipeline, invalid_history):
    history = prepare_history(panel, pd.Period("2023-12", "M"), 0, ["1"])
    if invalid_history == "all_missing":
        history["1"] = np.nan
    else:
        history.loc["2023-04-01", "1"] = np.inf
    result = predict(pipeline, history, horizons=[1, 3])
    assert_failed_without_reserve(result)
    assert not pipeline.calls


def test_failure_of_one_municipality_does_not_remove_another(panel, pipeline, monkeypatch):
    native = pipeline.predict_quantiles

    def sometimes_failed(**kwargs):
        assert len(kwargs["inputs"]) == 1
        if kwargs["inputs"][0][-1] == 222.:
            raise RuntimeError("synthetic failure for second municipality")
        return native(**kwargs)

    monkeypatch.setattr(pipeline, "predict_quantiles", sometimes_failed)
    history = prepare_history(panel, pd.Period("2023-12", "M"), 0, ["1", "2"])
    result = predict(pipeline, history, horizons=[1, 3], batch_size=1)
    assert len(result.forecasts) == 4
    valid = result.forecasts.loc[result.forecasts.municipality_id.eq("1")]
    invalid = result.forecasts.loc[result.forecasts.municipality_id.eq("2")]
    assert valid.status.eq("native").all() and np.isfinite(valid.y_pred).all()
    assert invalid.status.eq("failed").all() and invalid.y_pred.isna().all()
    assert invalid.effective_model.eq("Chronos-2").all()
    assert "synthetic failure for second municipality" in str(result.errors)
