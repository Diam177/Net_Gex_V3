
# -*- coding: utf-8 -*-
"""
final_table.py — сборка финальных таблиц по страйкам окна.

Экспортируемые функции:
- build_final_tables_from_corr(df_corr, windows, cfg=FinalTableConfig())
- build_final_sum_from_corr(df_corr, windows, selected_exps, weight_mode, cfg=FinalTableConfig())
- process_from_raw(...)  [заглушка-переадресация, если в проекте используется полный конвейер]
- _series_ctx_from_corr(df_corr, windows) -> Dict[str, dict]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Внешние зависимости пайплайна
from lib.netgex_ag import NetGEXAGConfig, compute_netgex_ag_per_expiry
try:
    from lib.power_zone_er import compute_power_zone  # опционально
except Exception:  # pragma: no cover
    compute_power_zone = None  # тип: ignore

__all__ = [
    "FinalTableConfig",
    "build_final_tables_from_corr",
    "build_final_sum_from_corr",
    "process_from_raw",
    "_series_ctx_from_corr",
]

# ----------------------- Конфиг -----------------------

@dataclass
class FinalTableConfig:
    """Параметры сборки финальных таблиц."""
    scale_millions: float = 1e6  # делитель для *_M
    day_high: Optional[float] = None
    day_low: Optional[float] = None


# ----------------------- Внутренние хелперы -----------------------

_REQ_COLS = {
    "exp", "K", "side", "oi", "mult", "gamma_corr", "S",
}

def _ensure_required_columns(df: pd.DataFrame) -> None:
    miss = sorted(c for c in _REQ_COLS if c not in df.columns)
    if miss:
        raise KeyError(f"df_corr is missing required columns: {miss}")

def _norm_weights(weights: Dict[str, float]) -> Dict[str, float]:
    vals = np.array(list(weights.values()), dtype=float)
    s = float(np.nansum(vals))
    if not np.isfinite(s) or s == 0:
        n = len(vals) if len(vals) > 0 else 1
        return {k: 1.0 / n for k in weights.keys()}
    return {k: float(v) / s for k, v in weights.items()}

def _window_strikes_for_exp(df_corr: pd.DataFrame, exp: str, windows: Dict[str, Iterable[int]]) -> np.ndarray:
    """Вернуть массив страйков K, попадающих в окно для exp. Если окно пустое — вся сетка K."""
    Ks_sorted = np.array(sorted(pd.to_numeric(df_corr.loc[df_corr["exp"] == exp, "K"], errors="coerce").dropna().unique()), dtype=float)
    idx = np.asarray(list(windows.get(exp, [])), dtype=int)
    if Ks_sorted.size == 0:
        return Ks_sorted
    if idx.size == 0 or np.min(idx) < 0 or np.max(idx) >= Ks_sorted.size:
        return Ks_sorted
    return Ks_sorted[idx]

def _pivot_sum(g: pd.DataFrame, col: str, default: float = 0.0) -> pd.DataFrame:
    """Суммы по K и side → столбцы call_*, put_*."""
    if col not in g.columns:
        return pd.DataFrame()
    t = (
        g.groupby(["K", "side"], as_index=False)[col]
        .sum()
        .pivot(index="K", columns="side", values=col)
        .rename(columns={"C": "call", "P": "put"})
        .fillna(default)
    )
    t.index = pd.to_numeric(t.index, errors="coerce")
    t = t.sort_index().reset_index().rename(columns={"index": "K"})
    # Гарантируем наличие столбцов
    for c in ("call", "put"):
        if c not in t.columns:
            t[c] = default
    return t[["K", "call", "put"]]

def _to_side(series: pd.Series) -> str:
    s = str(series).upper()
    if s.startswith("C"):
        return "C"
    if s.startswith("P"):
        return "P"
    return s  # уже "C"/"P"

# ----------------------- Основные функции -----------------------

def _compute_per_exp_final(df_corr: pd.DataFrame, windows: Dict[str, Iterable[int]], exp: str, cfg: FinalTableConfig) -> pd.DataFrame:
    """
    Использует compute_netgex_ag_per_expiry для расчёта AG/NetGEX и собирает базовые поля.
    """
    _ensure_required_columns(df_corr)
    # Вызов стандартной функции проекта
    nt = compute_netgex_ag_per_expiry(
        df_corr=df_corr,
        exp=exp,
        windows=windows,
        cfg=NetGEXAGConfig(scale=cfg.scale_millions),
    )
    if nt is None or getattr(nt, "empty", True):
        return pd.DataFrame(columns=[
            "exp","K","S","F","call_oi","put_oi","call_vol","put_vol","iv_call","iv_put",
            "AG_1pct","NetGEX_1pct","AG_1pct_M","NetGEX_1pct_M","PZ","ER_Up","ER_Down"
        ])

    # Гарантируем наличие типовых столбцов
    cols_need = {
        "exp","K","S","F",
        "AG_1pct","NetGEX_1pct","AG_1pct_M","NetGEX_1pct_M",
        "call_oi","put_oi","call_vol","put_vol","iv_call","iv_put"
    }
    for c in cols_need:
        if c not in nt.columns:
            # Попробуем достроить OI/Vol/IV из df_corr при необходимости
            if c in ("call_oi","put_oi","call_vol","put_vol","iv_call","iv_put"):
                g = df_corr.loc[df_corr["exp"] == exp].copy()
                g["side"] = g["side"].map(_to_side)
                if c.endswith("_oi"):
                    t = _pivot_sum(g, "oi")
                elif c.endswith("_vol"):
                    t = _pivot_sum(g, "volume")
                elif c in ("iv_call","iv_put"):
                    if "iv" in g.columns:
                        t2 = (
                            g.groupby(["K","side"], as_index=False)["iv"]
                            .median()
                            .pivot(index="K", columns="side", values="iv")
                            .rename(columns={"C":"iv_call","P":"iv_put"})
                        )
                        t2.index = pd.to_numeric(t2.index, errors="coerce")
                        t2 = t2.sort_index().reset_index()
                        nt = nt.merge(t2, on="K", how="left")
                        continue
                    else:
                        nt[c] = np.nan
                        continue
                else:
                    nt[c] = np.nan
                    continue
                t = t.rename(columns={"call": "call_oi" if c.endswith("_oi") else "call_vol",
                                      "put":  "put_oi"  if c.endswith("_oi") else "put_vol"})
                nt = nt.merge(t, on="K", how="left")
            else:
                nt[c] = np.nan

    # Power Zone, если доступно ядро
    if compute_power_zone is not None and "PZ" not in nt.columns:
        try:
            nt["PZ"] = compute_power_zone(
                K=np.asarray(pd.to_numeric(nt["K"], errors="coerce")),
                S=float(pd.to_numeric(nt["S"], errors="coerce").dropna().median()) if "S" in nt.columns else np.nan,
                ag=np.asarray(pd.to_numeric(nt["AG_1pct"], errors="coerce")),
                net=np.asarray(pd.to_numeric(nt["NetGEX_1pct"], errors="coerce")),
            )
        except Exception:
            nt["PZ"] = np.nan

    nt["exp"] = exp
    return nt

def build_final_tables_from_corr(
    df_corr: pd.DataFrame,
    windows: Dict[str, Iterable[int]],
    cfg: FinalTableConfig = FinalTableConfig(),
) -> Dict[str, pd.DataFrame]:
    """
    Собрать финальные таблицы по каждой экспирации из уже «исправленных» данных.
    """
    if df_corr is None or getattr(df_corr, "empty", True):
        return {}
    df_corr = df_corr.copy()
    # Нормализация side
    if "side" in df_corr.columns:
        df_corr["side"] = df_corr["side"].map(_to_side)
    exps: List[str] = sorted({str(x) for x in df_corr["exp"].dropna().unique().tolist()})
    results: Dict[str, pd.DataFrame] = {}
    for exp in exps:
        try:
            results[exp] = _compute_per_exp_final(df_corr, windows, exp, cfg)
        except Exception:
            # Не роняем весь пайплайн на одной экспирации
            results[exp] = pd.DataFrame()
    return results

def build_final_sum_from_corr(
    df_corr: pd.DataFrame,
    windows: Dict[str, Iterable[int]],
    selected_exps: Optional[List[str]] = None,
    weight_mode: str = "1/T",   # "1/T" | "equal"
    cfg: FinalTableConfig = FinalTableConfig(),
) -> pd.DataFrame:
    """
    Суммарная таблица по нескольким экспирациям.
    AG/NetGEX — взвешенные суммы; OI — суммы; S/F — медиана по сериям.
    """
    all_tables = build_final_tables_from_corr(df_corr, windows, cfg=cfg)
    if not all_tables:
        return pd.DataFrame()
    if not selected_exps:
        selected_exps = sorted(all_tables.keys())

    # веса серий
    weights: Dict[str, float] = {}
    if weight_mode == "1/T" and "T" in df_corr.columns:
        # Берём медианный T по серии
        for e in selected_exps:
            tvals = pd.to_numeric(df_corr.loc[df_corr["exp"] == e, "T"], errors="coerce").dropna().values
            w = float(np.nanmedian(1.0 / np.maximum(tvals, 1e-9))) if tvals.size else 1.0
            weights[e] = w
    else:
        weights = {e: 1.0 for e in selected_exps}
    weights = _norm_weights(weights)

    # общая сетка K
    Ks: List[float] = sorted({
        float(k)
        for e in selected_exps
        for k in pd.to_numeric(all_tables.get(e, pd.DataFrame()).get("K", pd.Series(dtype=float)), errors="coerce").dropna().unique().tolist()
    })
    if not Ks:
        return pd.DataFrame()

    base = pd.DataFrame({"K": Ks})

    # S и F — медиана
    S_vals, F_vals = [], []
    for e in selected_exps:
        t = all_tables.get(e)
        if t is None or getattr(t, "empty", True):
            continue
        if "S" in t.columns:
            S_vals.extend(pd.to_numeric(t["S"], errors="coerce").dropna().tolist())
        if "F" in t.columns:
            F_vals.extend(pd.to_numeric(t["F"], errors="coerce").dropna().tolist())
    base["S"] = float(np.nanmedian(S_vals)) if S_vals else np.nan
    base["F"] = float(np.nanmedian(F_vals)) if F_vals else np.nan

    # AG/NetGEX — взвешенная сумма
    base["AG_1pct"] = 0.0
    base["NetGEX_1pct"] = 0.0
    for e in selected_exps:
        t = all_tables.get(e)
        if t is None or getattr(t, "empty", True):
            continue
        m = (
            t[["K","AG_1pct","NetGEX_1pct"]]
            .copy()
            .astype({"K":"float64"})
            .groupby("K", as_index=False)[["AG_1pct","NetGEX_1pct"]]
            .sum()
        )
        base = base.merge(m, on="K", how="left", suffixes=("","_add")).fillna(0.0)
        w = weights.get(e, 0.0)
        base["AG_1pct"] += w * base.pop("AG_1pct_add")
        base["NetGEX_1pct"] += w * base.pop("NetGEX_1pct_add")

    # *_M
    scale = float(cfg.scale_millions) if cfg.scale_millions else 1e6
    base["AG_1pct_M"] = base["AG_1pct"] / scale
    base["NetGEX_1pct_M"] = base["NetGEX_1pct"] / scale

    # OI суммы
    if {"K","side","oi"}.issubset(df_corr.columns):
        g = df_corr[df_corr["exp"].isin(selected_exps)].copy()
        g["side"] = g["side"].map(_to_side)
        oi = _pivot_sum(g, "oi", default=0.0)
        oi = oi.rename(columns={"call":"call_oi","put":"put_oi"})
        base = base.merge(oi, on="K", how="left")
    else:
        base["call_oi"] = np.nan
        base["put_oi"] = np.nan

    # Volume суммы, если есть
    if {"K","side","volume"}.issubset(df_corr.columns):
        g = df_corr[df_corr["exp"].isin(selected_exps)].copy()
        g["side"] = g["side"].map(_to_side)
        vol = _pivot_sum(g, "volume", default=0.0)
        vol = vol.rename(columns={"call":"call_vol","put":"put_vol"})
        base = base.merge(vol, on="K", how="left")
    else:
        base["call_vol"] = np.nan
        base["put_vol"] = np.nan

    # IV медианы по сторонам, если есть
    if {"K","side","iv","exp"}.issubset(df_corr.columns):
        g = df_corr[df_corr["exp"].isin(selected_exps)].copy()
        g["side"] = g["side"].map(_to_side)
        t2 = (
            g.groupby(["K","side"], as_index=False)["iv"]
            .median()
            .pivot(index="K", columns="side", values="iv")
            .rename(columns={"C":"iv_call","P":"iv_put"})
            .reset_index()
        )
        t2["K"] = pd.to_numeric(t2["K"], errors="coerce")
        base = base.merge(t2, on="K", how="left")
    else:
        base["iv_call"] = np.nan
        base["iv_put"] = np.nan

    # Power Zone
    if compute_power_zone is not None:
        try:
            base["PZ"] = compute_power_zone(
                K=np.asarray(pd.to_numeric(base["K"], errors="coerce")),
                S=float(pd.to_numeric(base["S"], errors="coerce")),
                ag=np.asarray(pd.to_numeric(base["AG_1pct"], errors="coerce")),
                net=np.asarray(pd.to_numeric(base["NetGEX_1pct"], errors="coerce")),
            )
        except Exception:
            base["PZ"] = np.nan
    else:
        base["PZ"] = np.nan

    return base

# ----------------------- Контекст серий -----------------------

def _series_ctx_from_corr(
    df_corr: pd.DataFrame,
    windows: Dict[str, Iterable[int]],
) -> Dict[str, dict]:
    """
    Возвращает по каждой экспирации агрегаты, полезные для UI и Power Zone.
    """
    _ensure_required_columns(df_corr)
    res: Dict[str, dict] = {}
    exps: List[str] = sorted({str(x) for x in df_corr["exp"].dropna().unique().tolist()})

    # Суммарные AG и NetGEX для нормировки долей
    totals: Dict[str, float] = {"AG": 0.0, "NET": 0.0}
    per_exp_vals: Dict[str, Tuple[float, float]] = {}

    for exp in exps:
        nt = compute_netgex_ag_per_expiry(
            df_corr=df_corr, exp=exp, windows=windows, cfg=NetGEXAGConfig(scale=1.0)
        )
        if nt is None or getattr(nt, "empty", True):
            per_exp_vals[exp] = (0.0, 0.0)
            continue
        ag = float(pd.to_numeric(nt["AG_1pct"], errors="coerce").abs().sum())
        net = float(pd.to_numeric(nt["NetGEX_1pct"], errors="coerce").sum())
        totals["AG"] += ag
        totals["NET"] += abs(net)
        per_exp_vals[exp] = (ag, net)

    # Контексты
    for exp in exps:
        g = df_corr.loc[df_corr["exp"] == exp].copy()
        g["side"] = g["side"].map(_to_side)
        call_oi = float(pd.to_numeric(g.loc[g["side"] == "C", "oi"], errors="coerce").sum())
        put_oi  = float(pd.to_numeric(g.loc[g["side"] == "P", "oi"], errors="coerce").sum())
        call_vol = float(pd.to_numeric(g.loc[g["side"] == "C", "volume"], errors="coerce").sum()) if "volume" in g.columns else np.nan
        put_vol  = float(pd.to_numeric(g.loc[g["side"] == "P", "volume"], errors="coerce").sum()) if "volume" in g.columns else np.nan
        iv_call = float(pd.to_numeric(g.loc[g["side"] == "C", "iv"], errors="coerce").median()) if "iv" in g.columns else np.nan
        iv_put  = float(pd.to_numeric(g.loc[g["side"] == "P", "iv"], errors="coerce").median()) if "iv" in g.columns else np.nan
        T_med   = float(pd.to_numeric(g["T"], errors="coerce").median()) if "T" in g.columns else np.nan

        ag, net = per_exp_vals.get(exp, (0.0, 0.0))
        ag_share  = (ag / totals["AG"]) if totals["AG"] > 0 else np.nan
        net_share = (abs(net) / totals["NET"]) if totals["NET"] > 0 else np.nan

        res[exp] = {
            "gamma_abs_share": ag_share,
            "gamma_net_share": net_share,
            "call_oi": call_oi,
            "put_oi": put_oi,
            "call_vol": call_vol,
            "put_vol": put_vol,
            "iv_call": iv_call,
            "iv_put": iv_put,
            "T": T_med,
        }
    return res

# ----------------------- Полный цикл (заглушка) -----------------------

def process_from_raw(*args, **kwargs):
    """
    В вашем проекте полный конвейер реализован в других модулях.
    Эта функция оставлена для совместимости с импортами; направляет на build_final_tables_from_corr,
    если переданы df_corr и windows в именованных аргументах.
    """
    df_corr = kwargs.get("df_corr")
    windows = kwargs.get("windows")
    if df_corr is not None and windows is not None:
        return build_final_tables_from_corr(df_corr, windows, cfg=kwargs.get("cfg", FinalTableConfig()))
    raise NotImplementedError("process_from_raw is not implemented in this lightweight module")
