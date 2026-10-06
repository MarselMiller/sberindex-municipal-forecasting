"""Чтение и календарная сетка. Пустой месяц — NaN, никогда не ноль."""
from __future__ import annotations
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_data(path: Path, category: str = "Все категории") -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Нет файла данных: {path}")
    df = pd.read_csv(path, dtype={"territory_id": "string", "oktmo": "string", "region_code": "string"})
    required = {"date", "territory_id", "value"}
    if not required.issubset(df.columns):
        raise ValueError(f"Не хватает столбцов: {sorted(required - set(df.columns))}")
    if "category" in df:
        df = df.loc[df["category"].eq(category)].copy()
    if df.empty:
        raise ValueError("После выбора категории таблица пуста.")
    if df[list(required)].isna().any().any():
        raise ValueError("В исходных ключах или имеющихся значениях есть пустые ячейки.")
    df["ds"] = pd.PeriodIndex(df["date"].astype(str), freq="M").to_timestamp()
    df["municipality_id"] = df["territory_id"].astype(str)
    df["y"] = pd.to_numeric(df["value"], errors="raise").astype(float)
    if not np.isfinite(df["y"]).all() or (df["y"] < 0).any():
        raise ValueError("Ожидались конечные неотрицательные расходы.")
    if df.duplicated(["municipality_id", "ds"]).any():
        raise ValueError("Дубликаты МО + месяц. Проверьте соединение исторических версий справочника.")
    return df.sort_values(["municipality_id", "ds"]).reset_index(drop=True)


def make_panel(df: pd.DataFrame) -> pd.DataFrame:
    panel = df.pivot(index="ds", columns="municipality_id", values="y")
    months = pd.date_range(panel.index.min(), panel.index.max(), freq="MS")
    return panel.reindex(months).sort_index(axis=1).astype(float)


def period_end(period: pd.Period) -> pd.Timestamp:
    return period.to_timestamp(how="end").normalize()


def get_prefix(panel: pd.DataFrame, origin: pd.Period, release_lag_months: int) -> pd.DataFrame:
    if release_lag_months < 0:
        raise ValueError("Лаг публикации не может быть отрицательным.")
    cutoff = (origin - release_lag_months).to_timestamp()
    prefix = panel.loc[panel.index <= cutoff].copy()
    if prefix.empty:
        raise ValueError("На эту дату нет доступной истории.")
    # Достраиваем только уже прошедшие, условно доступные месяцы.
    return prefix.reindex(pd.date_range(prefix.index.min(), cutoff, freq="MS"))


def eligibility(prefix: pd.DataFrame, min_observations: int, max_staleness: int) -> pd.DataFrame:
    finite = prefix.notna().to_numpy()
    n = finite.sum(axis=0)
    positions = np.arange(len(prefix))[:, None]
    last = np.where(finite, positions, -1).max(axis=0)
    stale = len(prefix) - 1 - last
    return pd.DataFrame({
        "municipality_id": prefix.columns.astype(str),
        "n_history": n, "staleness_months": stale,
        "eligible": (n >= min_observations) & (stale <= max_staleness),
    })


def audit(df: pd.DataFrame, panel: pd.DataFrame) -> dict:
    return {
        "observations": int(len(df)), "municipalities": int(panel.shape[1]),
        "months": int(panel.shape[0]), "first_month": str(panel.index.min().to_period("M")),
        "last_month": str(panel.index.max().to_period("M")),
        "complete_series": int(panel.notna().all().sum()),
        "missing_grid_cells": int(panel.isna().sum().sum()),
        "duplicates": int(df.duplicated(["municipality_id", "ds"]).sum()),
        "zero_values": int(df["y"].eq(0).sum()),
        "target": "Средние безналичные расходы, категория «Все категории», исходные рубли",
        "publication_dates_verified": False,
    }
