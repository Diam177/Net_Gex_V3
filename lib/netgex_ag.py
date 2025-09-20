"""
netgex_ag.py — расчёт Net GEX и AG из "исправленных" данных (df_corr), опциональная агрегация.
Единицы: $ на 1% движения (и млн $ при scale=1e6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List
import numpy as np
import pandas as pd

# NEW imports
from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar
import numpy.random as rnd  # Для bootstrap (замена sklearn)

@dataclass
class NetGEXAGConfig:
    scale: float = 1e6       # млн $ на 1% движения
    aggregate: str = "none"  # 'none' | 'sum' | 'sum_union'
    advanced_mode: bool = False  # NEW: Флаг для улучшенного режима
    market_cap: float = 654.8e9  # Пример для SPY, configurable
    adv: float = 70e6  # Average daily volume shares
    decay_theta: float = 0.5  # Decay rate, fitted default
    n_bootstrap: int = 1000  # Для CI

def _ensure_required_columns(df: pd.DataFrame) -> None:
    req = {"exp","K","side","oi","gamma_corr","S","mult"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"df_corr is missing columns: {sorted(miss)}")

def _window_strikes_for_exp(df_corr: pd.DataFrame, exp: str, windows: Optional[Dict[str, np.ndarray]]):
    Ks = np.array(sorted(df_corr.loc[df_corr["exp"]==exp, "K"].unique()), dtype=float)
    if windows is None: return Ks
    idx = windows.get(exp)
    if idx is None or len(idx)==0: return Ks
    idx = np.array(idx, dtype=int)
    idx = idx[(idx>=0)&(idx<len(Ks))]
    if len(idx)==0: return Ks
    return Ks[idx]

def compute_netgex_ag_per_expiry(df_corr: pd.DataFrame, exp: str,
                                 windows: Optional[Dict[str, np.ndarray]]=None,
                                 cfg: NetGEXAGConfig=NetGEXAGConfig()) -> pd.DataFrame:
    _ensure_required_columns(df_corr)
    g = df_corr.loc[df_corr["exp"]==exp].copy()
    if g.empty:
        return pd.DataFrame(columns=["exp","K","S","F","call_oi","put_oi","dg1pct_call","dg1pct_put","AG_1pct","NetGEX_1pct","AG_1pct_M","NetGEX_1pct_M"])

    Ks_keep = set(_window_strikes_for_exp(df_corr, exp, windows))

    dg_row = np.where(
        np.isfinite(g["gamma_corr"]) & (g["gamma_corr"]>0) &
        np.isfinite(g["S"]) & np.isfinite(g["mult"]) &
        np.isfinite(g["oi"]) & (g["oi"]>0),
        g["gamma_corr"].to_numpy() * (g["S"].to_numpy()**2) * 0.01 * g["mult"].to_numpy() * g["oi"].to_numpy(),
        0.0
    )
    g = g.assign(dg1pct_row=dg_row)
    agg = g.groupby(["K","side"], as_index=False).agg(oi_side=("oi","sum"), dg1pct_side=("dg1pct_row","sum"))
    pivot = agg.pivot_table(index="K", columns="side", values=["oi_side","dg1pct_side"], aggfunc="sum").fillna(0.0)
    pivot.columns = [f"{v}_{s.lower()}" for v,s in pivot.columns]
    pivot = pivot.reset_index()

    S_exp = float(np.nanmedian(g["S"].values)); pivot["S"]=S_exp
    if "F" in g.columns: pivot["F"]=float(np.nanmedian(g["F"].values))

    pivot = pivot.rename(columns={
        "oi_side_c":"call_oi","oi_side_p":"put_oi",
        "dg1pct_side_c":"dg1pct_call","dg1pct_side_p":"dg1pct_put",
    })

    if Ks_keep: pivot = pivot[pivot["K"].isin(Ks_keep)]

    pivot["AG_1pct"]     = pivot["dg1pct_call"] + pivot["dg1pct_put"]
    pivot["NetGEX_1pct"] = pivot["dg1pct_call"] - pivot["dg1pct_put"]

    if cfg.scale and cfg.scale>0:
        pivot["AG_1pct_M"]     = pivot["AG_1pct"]/cfg.scale
        pivot["NetGEX_1pct_M"] = pivot["NetGEX_1pct"]/cfg.scale

    # IMPROVED: Advanced mode
    if cfg.advanced_mode:
        # Bootstrap for CI (using numpy random)
        netgex_boot = []
        for _ in range(cfg.n_bootstrap):
            idx_boot = rnd.choice(len(g), len(g), replace=True)
            g_boot = g.iloc[idx_boot]
            # Пересчитать dg_row и agg как выше
            dg_row_boot = ...  # Повторить расчёт
            # ... (аналогично, получить boot NetGEX)
            netgex_boot.append(boot_netgex)

        netgex_mean = np.mean(netgex_boot, axis=0)
        netgex_std = np.std(netgex_boot, axis=0)
        pivot['NetGEX_mean'] = netgex_mean
        pivot['NetGEX_std'] = netgex_std

        # Normalization
        adv_factor = cfg.adv / 1e6
        pivot['NetGEX_norm'] = netgex_mean / (cfg.market_cap * adv_factor)

        # G-Flip with cubic spline
        Ks = pivot['K'].values.astype(float)
        if len(Ks) > 2:
            spline = CubicSpline(Ks, pivot['NetGEX_norm'].values)
            try:
                gflip = root_scalar(spline, bracket=[min(Ks), max(Ks)]).root
                pivot['GFlip'] = gflip  # Или добавить как отдельную колонку/аттрибут
            except:
                pass

    pivot.insert(0,"exp",exp)
    cols=["exp","K","S"] + (["F"] if "F" in pivot.columns else []) + ["call_oi","put_oi","dg1pct_call","dg1pct_put","AG_1pct","NetGEX_1pct"]
    if "AG_1pct_M" in pivot.columns: cols += ["AG_1pct_M","NetGEX_1pct_M"]
    if cfg.advanced_mode: cols += ["NetGEX_norm", "NetGEX_std"]
    pivot = pivot[cols].sort_values("K").reset_index(drop=True)
    return pivot

# ... (остальной оригинальный код для compute_netgex_ag с аналогичными правками для aggregation: weighted sum with decay)
def compute_netgex_ag(df_corr: pd.DataFrame, windows: Optional[Dict[str, np.ndarray]]=None,
                      cfg: NetGEXAGConfig=NetGEXAGConfig()):
    # ORIGINAL ...
    if cfg.advanced_mode:
        # Weighted aggregation with decay
        # Вычислить T_exp, w = exp(-cfg.decay_theta * T_exp) * (OI / total_OI)
        # Затем base["NetGEX_1pct"] = sum weighted
    # ...
