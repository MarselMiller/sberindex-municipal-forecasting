"""TRAIN-only fixed linear models for the E07c controlled synthetic benchmark.

There is no E07b feasibility gate here: independently generated cohorts are a
different experiment. Threshold selection and evaluation belong to the caller.
No TEST or VALIDATION labels are accepted by the fitting entry point.

The existing E07b median/filter/population-standard-deviation preprocessor is
reused unchanged. If sklearn is absent, already installed SciPy solves the same
binary L2 objective: sum(log loss) + ||weights||**2 / (2*C). The intercept is
unpenalized. Failure to converge aborts fitting, rather than producing a nominal
successful model or replacing its predictions with a constant.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import importlib.util
import warnings

import numpy as np
import pandas as pd

from .early_warning_full_panel_models import TrainOnlyPreprocessor


KEYS = ("series_id", "cohort", "month_index")
MODEL_GROUPS = {
    "S0": (), "S1": ("history",), "S2": ("history", "detector"),
    "S3": ("history", "detector", "external"),
}
OBJECTIVE = "sum_binary_log_loss_plus_squared_weights_over_2C_unpenalized_intercept"
BACKEND_POLICY = "prefer_installed_sklearn_else_scipy_equivalent_L2_logistic"
COEFFICIENT_COLUMNS = (
    "model", "k", "feature", "group", "coefficient", "coefficient_scale",
    "coefficient_original_units", "training_median",
    "training_mean_after_imputation", "training_scale",
)


class ModelConvergenceError(RuntimeError):
    """Optimization failed; diagnostics can be saved by the experiment runner."""

    def __init__(self, diagnostics: dict):
        self.diagnostics = diagnostics
        super().__init__(f"Logistic model did not converge: {diagnostics}")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    result = np.empty_like(z)
    positive = z >= 0
    result[positive] = 1 / (1 + np.exp(-z[positive]))
    exponential = np.exp(z[~positive])
    result[~positive] = exponential / (1 + exponential)
    return result


def logistic_objective_and_gradient(parameters, X, y, C=1.0):
    """Numerically stable summed binary loss and its analytic gradient.

    Parameters are feature weights followed by the unpenalized intercept.
    This pure function also supports a zero-column (intercept-only) matrix.
    """
    matrix, truth, params = (np.asarray(value, dtype=float) for value in (X, y, parameters))
    if (matrix.ndim != 2 or truth.shape != (len(matrix),)
            or params.shape != (matrix.shape[1] + 1,) or not len(truth)
            or not np.isfinite(matrix).all() or not np.isfinite(params).all()
            or not np.isin(truth, [0, 1]).all() or not np.isfinite(C) or C <= 0):
        raise ValueError("Finite numerical matrix, binary labels, weights/intercept and positive C required")
    weight, intercept = params[:-1], params[-1]
    score = matrix @ weight + intercept
    residual = _sigmoid(score) - truth
    loss = float(np.sum(np.logaddexp(0, score) - truth * score) + weight @ weight / (2 * C))
    gradient = np.r_[matrix.T @ residual + weight / C, residual.sum()]
    return loss, gradient


def _fixed_config(config: Mapping) -> dict:
    provided = config.get("models", config)
    if not isinstance(provided, Mapping):
        raise ValueError("Model configuration must be a mapping")
    fixed = dict(C=1.0, max_iter=2000, random_state=42, class_weight=None,
                 n_jobs=1, backend=BACKEND_POLICY)
    for name, expected in fixed.items():
        if name in provided and provided[name] != expected:
            raise ValueError(f"E07c fixed model parameter {name} must be {expected!r}")
    if "names" in provided and list(provided["names"]) != list(MODEL_GROUPS):
        raise ValueError("E07c model names/order are fixed S0,S1,S2,S3")
    return fixed


def _groups(groups: Mapping[str, Sequence[str]]) -> tuple[dict, dict]:
    if set(groups) != {"history", "detector", "external"}:
        raise ValueError("Explicit history/detector/external feature groups required")
    cleaned, owners = {}, {}
    forbidden = ("onset", "event_id", "event_time", "time_to_event", "time_until",
                 "shock_in", "future", "offline", "pelt", "binseg", "breakpoint",
                 "label", "truth", "target", "y_true", "has_event", "anticipated",
                 "precursor_class", "noise_class", "series_id", "cohort", "month_index")
    for group in ("history", "detector", "external"):
        if isinstance(groups[group], (str, bytes)):
            raise ValueError("Feature group must contain a sequence of feature names")
        cleaned[group] = list(groups[group])
        for name in cleaned[group]:
            if (not isinstance(name, str) or not name.startswith(group + "_")
                    or any(token in name.lower() for token in forbidden)):
                raise ValueError(f"Noncausal/target metadata or wrong namespace in feature matrix: {name}")
            if name in owners:
                raise ValueError(f"Duplicate feature in explicit groups: {name}")
            owners[name] = group
    return cleaned, owners


def _keys(frame: pd.DataFrame, *, cases=False) -> pd.DataFrame:
    needed = set(KEYS) | ({"k"} if cases else set())
    if not needed.issubset(frame):
        raise ValueError(f"Missing case/feature keys: {sorted(needed - set(frame))}")
    result = frame.copy()
    if result[list(KEYS)].isna().any().any():
        raise ValueError("Missing series/cohort/month key")
    result["series_id"] = result.series_id.astype(str)
    if result.series_id.eq("").any() or not result.cohort.isin(["train", "validation", "test"]).all():
        raise ValueError("Nonempty series IDs and explicit train/validation/test cohorts required")
    month = pd.to_numeric(result.month_index, errors="raise")
    if not np.isfinite(month).all() or month.lt(0).any() or not np.equal(month, np.floor(month)).all():
        raise ValueError("month_index must be a nonnegative integer")
    result["month_index"] = month.astype(int)
    unique = list(KEYS)
    if cases:
        horizon = pd.to_numeric(result.k, errors="raise")
        if not horizon.isin([1, 3]).all():
            raise ValueError("E07c horizons must be k=1 or k=3")
        result["k"] = horizon.astype(int)
        unique += ["k"]
    if result.duplicated(unique).any():
        raise ValueError("Duplicate series/cohort/month case or feature keys")
    return result


def _flag(frame: pd.DataFrame, name: str, default=None) -> pd.Series:
    if name not in frame:
        if default is None:
            raise ValueError(f"Missing explicit flag {name}")
        return pd.Series(default, index=frame.index, dtype=bool)
    value = frame[name]
    if value.isna().any() or not value.isin([True, False, 0, 1]).all():
        raise ValueError(f"Flag {name} must be explicit boolean")
    return value.astype(bool)


def _monitoring_flags(frame: pd.DataFrame) -> pd.Series:
    eligible_name = "eligible" if "eligible" in frame else "eligible_at_origin"
    return (_flag(frame, "fully_known") & _flag(frame, "at_risk")
            & _flag(frame, eligible_name) & ~_flag(frame, "active_regime", False))


def _training_cases(cases: pd.DataFrame) -> pd.DataFrame:
    train = _keys(cases, cases=True)
    if train.empty or not train.cohort.eq("train").all():
        raise ValueError("fit_training_models accepts a nonempty TRAIN cohort only")
    if "label" not in train:
        raise ValueError("Training cases require label")
    known = _flag(train, "fully_known")
    label = pd.to_numeric(train.label, errors="raise")
    if label.loc[~known].notna().any():
        raise ValueError("Unknown/censored training labels must remain missing")
    if not label.loc[known].isin([0, 1]).all():
        raise ValueError("Fully known training labels must be binary")
    train["label"] = label
    return train.loc[_monitoring_flags(train)].sort_values(["k", *KEYS], kind="stable").reset_index(drop=True)


def _align(cases: pd.DataFrame, features: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if not set(columns).issubset(features):
        raise ValueError(f"Missing candidate features: {sorted(set(columns) - set(features))}")
    if set(columns) & set(cases.columns):
        raise ValueError("Feature names collide with case/target metadata")
    result = cases.merge(features[list(KEYS) + list(columns)], on=list(KEYS), how="left",
                         validate="many_to_one", indicator="_feature_join", sort=False)
    if not result._feature_join.eq("both").all():
        raise ValueError("Every prediction/training case needs its own-origin feature row")
    return result.drop(columns="_feature_join")


def _backend() -> str:
    return "sklearn" if importlib.util.find_spec("sklearn") is not None else "scipy"


def _fit_logistic(X: np.ndarray, y: np.ndarray, parameters: dict) -> tuple[np.ndarray, float, dict]:
    prior = float(y.mean())
    if not 0 < prior < 1:
        raise ValueError("Logistic training requires both classes; no silent model substitution")
    if X.shape[1] == 0:
        return np.empty(0), float(np.log(prior / (1 - prior))), dict(
            backend="analytic_intercept_only", converged=True, iterations=0,
            optimizer_status=0, optimizer_message="No varying TRAIN features; unpenalized intercept optimum",
            objective_value=float(-np.sum(y * np.log(prior) + (1 - y) * np.log(1 - prior))),
            gradient_max_abs=0.0)
    backend = _backend()
    if backend == "sklearn":
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(C=parameters["C"], max_iter=parameters["max_iter"],
                                        random_state=parameters["random_state"], class_weight=None,
                                        n_jobs=1, solver="lbfgs")
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(X, y)
        params = np.r_[classifier.coef_[0], classifier.intercept_[0]]
        iterations = int(np.max(classifier.n_iter_))
        failed = [str(item.message) for item in notices if issubclass(item.category, ConvergenceWarning)]
        converged = not failed and iterations < parameters["max_iter"]
        diagnostics = dict(backend=backend, converged=converged, iterations=iterations,
                           optimizer_status=0 if converged else 1,
                           optimizer_message="sklearn lbfgs completed without convergence warning" if converged else "; ".join(failed) or "Iteration limit reached")
    else:
        from scipy.optimize import minimize
        initial = np.r_[np.zeros(X.shape[1]), np.log(prior / (1 - prior))]
        result = minimize(logistic_objective_and_gradient, initial, args=(X, y, parameters["C"]),
                          jac=True, method="L-BFGS-B",
                          options=dict(maxiter=parameters["max_iter"], gtol=1e-6, ftol=1e-12))
        params = np.asarray(result.x, dtype=float)
        diagnostics = dict(backend=backend, converged=bool(result.success), iterations=int(result.nit),
                           optimizer_status=int(result.status), optimizer_message=str(result.message))
    if not np.isfinite(params).all():
        diagnostics.update(converged=False, objective_value=None, gradient_max_abs=None,
                           reason="Optimizer returned nonfinite parameters")
        raise ModelConvergenceError(diagnostics)
    loss, gradient = logistic_objective_and_gradient(params, X, y, parameters["C"])
    diagnostics.update(objective_value=loss, gradient_max_abs=float(np.max(np.abs(gradient))))
    if not diagnostics["converged"] or not np.isfinite(loss) or not np.isfinite(gradient).all():
        diagnostics["converged"] = False
        raise ModelConvergenceError(diagnostics)
    return params[:-1], float(params[-1]), diagnostics


def fit_training_models(train_cases: pd.DataFrame, train_features: pd.DataFrame,
                        groups: Mapping[str, Sequence[str]], config: Mapping) -> dict:
    """Fit S0--S3 using admitted TRAIN rows only, separately for k=1/3.

    The bundle includes live preprocessors, CSV-ready tables and JSON-safe
    ``parameters``. ``predict_models`` can also use a JSON-restored bundle
    containing only that parameters list; no package-specific pickle is needed.
    """
    parameters = _fixed_config(config)
    feature_groups, owner = _groups(groups)
    train = _training_cases(train_cases)
    matrix = _keys(train_features)
    if not matrix.cohort.eq("train").all():
        raise ValueError("Training feature matrix must contain TRAIN only")
    if train.empty:
        raise ValueError("No fully known eligible at-risk TRAIN cases")
    columns = [name for group in ("history", "detector", "external") for name in feature_groups[group]]
    joined = _align(train, matrix, columns)
    models, saved, filters, coefficients, statuses, preprocessing = {}, [], [], [], [], []
    for k in sorted(train.k.unique()):
        current = joined.loc[joined.k.eq(k)].reset_index(drop=True)
        truth = current.label.to_numpy(dtype=float)
        if len(np.unique(truth)) != 2:
            raise ValueError(f"k={k} requires both TRAIN classes before fitting S0--S3")
        for model, included in MODEL_GROUPS.items():
            candidates = [name for group in included for name in feature_groups[group]]
            processor = None
            inference = dict(model=model, k=int(k), training_rows=len(current),
                             training_positives=int(truth.sum()), feature_names=[], medians={}, means={}, scales={},
                             weights=[], intercept=0.0, constant_risk=None, objective_definition=OBJECTIVE,
                             C=parameters["C"], random_state=42, class_weight=None, n_jobs=1,
                             threshold_selection="VALIDATION_only_external_to_model_fit")
            if model == "S0":
                inference["constant_risk"] = float(truth.mean())
                diagnostics = dict(backend="analytic_constant_prior", converged=True, iterations=0,
                                   optimizer_status=0, optimizer_message="TRAIN positive rate", objective_value=None,
                                   gradient_max_abs=None)
                coefficients.append(dict(model=model, k=int(k), feature="train_positive_rate", group="constant",
                                         coefficient=inference["constant_risk"], coefficient_scale="probability"))
            else:
                processor = TrainOnlyPreprocessor(candidates).fit(current)
                audit = processor.audit()
                audit["model"], audit["k"] = model, int(k)
                audit["group"] = audit.feature.map(owner)
                filters.extend(audit.to_dict("records"))
                X = processor.transform(current)
                try:
                    weight, intercept, diagnostics = _fit_logistic(X, truth, parameters)
                except ModelConvergenceError as exc:
                    exc.diagnostics.update(model=model, k=int(k), training_rows=len(current))
                    raise
                inference.update(feature_names=list(processor.selected_columns_), weights=weight.tolist(),
                                 intercept=intercept, medians=processor.medians_.to_dict(),
                                 means=processor.means_.to_dict(), scales=processor.scales_.to_dict())
                for name, value in zip(processor.selected_columns_, weight):
                    coefficients.append(dict(model=model, k=int(k), feature=name, group=owner[name],
                                             coefficient=float(value), coefficient_scale="standardized_train_feature",
                                             coefficient_original_units=float(value / processor.scales_[name]),
                                             training_median=float(processor.medians_[name]),
                                             training_mean_after_imputation=float(processor.means_[name]),
                                             training_scale=float(processor.scales_[name])))
                original_intercept = intercept - sum(float(w * processor.means_[name] / processor.scales_[name])
                                                      for name, w in zip(processor.selected_columns_, weight))
                coefficients.append(dict(model=model, k=int(k), feature="intercept", group="intercept",
                                         coefficient=intercept, coefficient_scale="standardized_feature_intercept",
                                         coefficient_original_units=float(original_intercept)))
            inference.update(backend=diagnostics["backend"], optimizer_diagnostics=diagnostics)
            models[(model, int(k))] = dict(preprocessor=processor, inference_metadata=inference)
            saved.append(inference)
            preprocessing.append({key: inference[key] for key in (
                "model", "k", "training_rows", "feature_names", "medians", "means", "scales")})
            statuses.append(dict(model=model, k=int(k), status="fitted", fit_performed=True,
                                 retained_features=len(inference["feature_names"]), training_rows=len(current),
                                 training_positives=int(truth.sum()), objective_definition=OBJECTIVE, **diagnostics))
    return dict(models=models, parameters=saved, coefficients=pd.DataFrame(coefficients, columns=COEFFICIENT_COLUMNS),
                feature_filter=pd.DataFrame(filters), model_status=pd.DataFrame(statuses),
                preprocessing=preprocessing, config=parameters,
                training_ids=sorted(train.series_id.unique().tolist()), fit_count=len(saved),
                classifier_fit_count=sum(row["backend"] in {"scipy", "sklearn"} for row in saved),
                constant_fit_count=sum(row["model"] == "S0" for row in saved))


def _thresholds(values) -> dict:
    if values is None:
        return {}
    if isinstance(values, pd.DataFrame):
        if not {"model", "k", "threshold"}.issubset(values) or values.duplicated(["model", "k"]).any():
            raise ValueError("Threshold table requires unique model,k,threshold")
        values = {(row.model, int(row.k)): row.threshold for row in values.itertuples()}
    if not isinstance(values, Mapping):
        raise ValueError("Thresholds must be a model/k mapping or threshold table")
    result = {}
    for key, threshold in values.items():
        if not isinstance(key, tuple) or len(key) != 2 or key[0] not in MODEL_GROUPS or key[1] not in (1, 3):
            raise ValueError("Threshold keys must be (S0--S3,k)")
        threshold = float(threshold)
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Threshold must be finite in [0,1]")
        result[(key[0], int(key[1]))] = threshold
    return result


def predict_models(fitted: Mapping, cases: pd.DataFrame, features: pd.DataFrame,
                   thresholds=None) -> pd.DataFrame:
    """Predict from frozen TRAIN parameters; labels are copied without reading.

    All passed rows receive risk, including audit rows if the caller supplies
    them. Alerts require explicit fully-known/eligible/at-risk/non-active flags.
    Pass VALIDATION-selected thresholds for TEST; None is temporary 0.5 only.
    No feature selection or imputation/scaling fit occurs here. JSON-restored
    ``{"parameters": fitted["parameters"]}`` gives the same inference path.
    """
    current, matrix = _keys(cases, cases=True), _keys(features)
    limits = _thresholds(thresholds)
    records = fitted.get("parameters")
    if records is None:
        raise ValueError("Fitted bundle requires frozen parameters")
    if len({(row["model"], row["k"]) for row in records}) != len(records):
        raise ValueError("Frozen model/horizon parameters must be unique")
    if thresholds is not None:
        required = {(row["model"], int(row["k"])) for row in records if current.k.eq(row["k"]).any()}
        if not required.issubset(limits):
            raise ValueError("Every predicted model/horizon requires its VALIDATION-selected threshold")
    outputs = []
    for record in records:
        model, k = record["model"], int(record["k"])
        selected = list(record["feature_names"])
        subset = current.loc[current.k.eq(k)].copy()
        aligned = _align(subset, matrix, selected)
        if record["constant_risk"] is not None:
            probability = np.full(len(aligned), float(record["constant_risk"]))
        else:
            numeric = aligned[selected].apply(pd.to_numeric, errors="raise").astype(float)
            if np.isinf(numeric.to_numpy()).any():
                raise ValueError("Infinite features cannot be median-imputed at inference")
            # JSON writers may sort dictionary keys. Frozen feature_names,
            # rather than dictionary insertion order, defines coefficient order.
            means = pd.Series(record["means"], dtype=float).reindex(selected)
            scales = pd.Series(record["scales"], dtype=float).reindex(selected)
            transformed = ((numeric.fillna(record["medians"]) - means) / scales).reindex(columns=selected)
            probability = _sigmoid(transformed.to_numpy(dtype=float) @ np.asarray(record["weights"]) + record["intercept"])
        threshold = limits.get((model, k), 0.5)
        result = subset.reset_index(drop=True)
        result["model"], result["probability"], result["threshold"] = model, probability, threshold
        result["alert"] = (probability >= threshold) & _monitoring_flags(result).to_numpy()
        result["threshold_status"] = "temporary_default_for_validation" if thresholds is None else "provided_validation_selection"
        outputs.append(result)
    if not outputs:
        raise ValueError("No fitted models available for prediction")
    return pd.concat(outputs, ignore_index=True)
