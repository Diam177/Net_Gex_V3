
# -*- coding: utf-8 -*-
"""
final_table.py — единая сборка финальной таблицы по страйкам окна:
- Берёт "исправленные" данные и окна (sanitize_window.py)
- Считает Net GEX / AG (netgex_ag.py)
- Считает Power Zone / ER Up / ER Down (power_zone_er.py)
- Собирает одну финальную таблицу по каждому expiry: [exp, K, S, F, call_oi, put_oi,
  dg1pct_call, dg1pct_put, AG_1pct, NetGEX_1pct, AG_1pct_M, NetGEX_1pct_M, PZ, ER_Up, ER_Down]

Предусмотрены два варианта входа:
1) process_from_raw(raw_records, S, ...) — полный цикл "сырые -> финальные таблицы"
2) build_final_tables_from_corr(df_corr, windows, ...) — если у вас уже есть df_corr и окна
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Внешние модули пайплайна (должны быть в PYTHONPATH или рядом)
from sanitize_window import (
    SanitizerConfig,
    sanitize_and_window_pipeline,
)

# ---------------------------------------------------------------------
# Helper: flatten Polygon snapshot records into a flat dict structure.
#
# Polygon's v3 snapshot returns option data nested inside 'details',
# 'greeks' and 'day' sub-dictionaries.  The sanitization pipeline
# expects records where fields like 'option_type', 'strike_price' and
# 'expiration_date' live at the top level.  Without flattening, the
# sanitize_window.build_raw_table skips all records because it cannot
# find the strike price and thus the resulting DataFrame is empty.
#
# This helper inspects each record; if it contains a 'details' dict
# (characteristic of Polygon snapshots), it constructs a new flat
# dictionary with the keys expected by build_raw_table.  Records that
# do not have a 'details' dict are returned untouched.
#
def _flatten_polygon_snapshot_records(records: Iterable[dict]) -> List[dict]:
    """Flatten nested Polygon snapshot records.

    Parameters
    ----------
    records : Iterable[dict]
        A list or iterable of raw records from the snapshot.

    Returns
    -------
    List[dict]
        A list of dictionaries with top-level keys suitable for
        sanitize_window.build_raw_table.
    """
    flat_records: List[dict] = []
    for r in records:
        # Only flatten dictionaries that have the nested Polygon structure
        if isinstance(r, dict) and "details" in r:
            details = r.get("details", {}) or {}
            greeks  = r.get("greeks", {}) or {}
            day     = r.get("day", {}) or {}

            flat: dict = {}
            # option side: contract_type may be 'call' or 'put'
            contract_type = details.get("contract_type")
            if contract_type:
                flat["option_type"] = contract_type
            # strike price
            strike = details.get("strike_price")
            if strike is not None:
                flat["strike_price"] = strike
            # expiration date
            exp_date = details.get("expiration_date")
            if exp_date:
                flat["expiration_date"] = exp_date
            # shares per contract / contract size
            spc = details.get("shares_per_contract")
            if spc is not None:
                flat["shares_per_contract"] = spc
            # open interest; default to None if missing
            if "open_interest" in r:
                flat["open_interest"] = r.get("open_interest")
            # volume: use day.volume if present
            vol = None
            if day and isinstance(day, dict):
                vol = day.get("volume")
            if vol is not None:
                flat["volume"] = vol
            # implied volatility
            iv = r.get("implied_volatility")
            if iv is not None:
                flat["implied_volatility"] = iv
            # greeks: delta, gamma, vega
            for greek_name in ("delta", "gamma", "vega"):
                if greek_name in greeks:
                    flat[greek_name] = greeks.get(greek_name)
            # If nothing was extracted, fall back to the original record
            if flat:
                flat_records.append(flat)
            else:
                flat_records.append(r)
        else:
            # not a dict or not Polygon snapshot structure; append as-is
            flat_records.append(r)
    return flat_records
from netgex_ag import (
    NetGEXAGConfig,
    compute_netgex_ag_per_expiry,
)
from power_zone_er import compute_power_zone_and_er


# --------- конфиг ---------

@dataclass
class FinalTableConfig:
    # Масштаб для млн $ в колонках *_M (используется в netgex_ag)
    scale_millions: float = 1e6
    # Параметры для Power Zone / ER (оставляем значения по умолчанию из power_zone_er)
    day_high: Optional[float] = None
    day_low: Optional[float] = None


# --------- helpers ---------

def _window_strikes(df_corr: pd.DataFrame, exp: str, windows: Dict[str, np.ndarray]) -> np.ndarray:
    Ks = np.array(sorted(df_corr.loc[df_corr["exp"] == exp, "K"].unique()), dtype=float)
    idx = np.array(windows.get(exp, []), dtype=int)
    if Ks.size == 0:
        return Ks
    if idx.size == 0 or idx.max() >= Ks.size or idx.min() < 0:
        return Ks
    return Ks[idx]


def _series_ctx_from_corr(df_corr: pd.DataFrame, exp: str) -> Dict[str, dict]:
    """
    Конструирует all_series_ctx-словарь для power_zone_er из df_corr по конкретной экспирации.
    Возвращает dict с одним ключом exp -> контекст (совместимо с compute_power_zone_and_er).
    """
    g = df_corr[df_corr["exp"] == exp].copy()
    if g.empty:
        return {}

    # Страйки и сортировка
    Ks = np.array(sorted(g["K"].unique()), dtype=float)
    # OI / Volume по сторонам
    agg_oi = g.groupby(["K", "side"], as_index=False)["oi"].sum()
    agg_vol = g.groupby(["K", "side"], as_index=False)["vol"].sum()
    pivot_oi = agg_oi.pivot_table(index="K", columns="side", values="oi", aggfunc="sum").fillna(0.0)
    pivot_vol = agg_vol.pivot_table(index="K", columns="side", values="vol", aggfunc="sum").fillna(0.0)
    call_oi = {float(k): float(v) for k, v in pivot_oi.get("C", pd.Series(dtype=float)).items()}
    put_oi  = {float(k): float(v) for k, v in pivot_oi.get("P", pd.Series(dtype=float)).items()}
    call_vol= {float(k): float(v) for k, v in pivot_vol.get("C", pd.Series(dtype=float)).items()}
    put_vol = {float(k): float(v) for k, v in pivot_vol.get("P", pd.Series(dtype=float)).items()}

    # IV по сторонам (исправленная)
    iv_call = g[g["side"]=="C"].groupby("K")["iv_corr"].median().to_dict()
    iv_put  = g[g["side"]=="P"].groupby("K")["iv_corr"].median().to_dict()

    # Dollar gamma на 1% * OI по сторонам (из df_corr) — построим для формы профиля
    g["dg1pct_row"] = np.where(
        np.isfinite(g["gamma_corr"]) & (g["gamma_corr"] > 0) &
        np.isfinite(g["S"]) & np.isfinite(g["mult"]) &
        np.isfinite(g["oi"]) & (g["oi"] > 0),
        g["gamma_corr"].to_numpy() * (g["S"].to_numpy()**2) * 0.01 * g["mult"].to_numpy() * g["oi"].to_numpy(),
        0.0
    )
    agg_dg = g.groupby(["K","side"], as_index=False)["dg1pct_row"].sum()
    p_dg = agg_dg.pivot_table(index="K", columns="side", values="dg1pct_row", aggfunc="sum").fillna(0.0)
    dg_call = p_dg.get("C", pd.Series(dtype=float)).reindex(Ks, fill_value=0.0).to_numpy()
    dg_put  = p_dg.get("P", pd.Series(dtype=float)).reindex(Ks, fill_value=0.0).to_numpy()

    # gamma_abs_share / gamma_net_share — возьмём форму профилей от долларовой гаммы:
    # Важно: compute_power_zone_and_er далее нормализует эти ряды, поэтому абсолютный масштаб не критичен.
    gamma_abs_share = (dg_call + dg_put)  # прокси AG-профиля
    gamma_net_share = (dg_call - dg_put)  # прокси NetGEX-профиля

    # Время до экспирации (медиана T)
    T_med = float(np.nanmedian(g["T"].values)) if len(g) else 0.0

    ctx = {
        "strikes": Ks.tolist(),
        "gamma_abs_share": gamma_abs_share,
        "gamma_net_share": gamma_net_share,
        "call_oi": call_oi,
        "put_oi": put_oi,
        "call_vol": call_vol,
        "put_vol": put_vol,
        "iv_call": iv_call,
        "iv_put": iv_put,
        "T": T_med,
    }
    return {exp: ctx}


def build_final_tables_from_corr(
    df_corr: pd.DataFrame,
    windows: Dict[str, np.ndarray],
    cfg: FinalTableConfig = FinalTableConfig(),
) -> Dict[str, pd.DataFrame]:
    """
    Собирает финальные таблицы по каждой экспирации:
    exp, K, S, F, call_oi, put_oi, dg1pct_call, dg1pct_put, AG_1pct, NetGEX_1pct,
    AG_1pct_M, NetGEX_1pct_M, PZ, ER_Up, ER_Down
    """
    results: Dict[str, pd.DataFrame] = {}

    # Пройдём по экспирациям, для каждой построим таблицу NetGEX/AG и добавим PZ/ER
    for exp in sorted(df_corr["exp"].dropna().unique()):
        # 1) таблица NetGEX/AG по окну
        net_tbl = compute_netgex_ag_per_expiry(
            df_corr, exp, windows=windows,
            cfg=NetGEXAGConfig(scale=cfg.scale_millions, aggregate="none")
        )
        # 1.b) добавим объёмы по сторонам из df_corr
        g_exp = df_corr[df_corr["exp"] == exp].copy()
        agg_vol = g_exp.groupby(["K","side"], as_index=False)["vol"].sum()
        pv = agg_vol.pivot_table(index="K", columns="side", values="vol", aggfunc="sum").fillna(0.0)
        call_vol_map = {float(k): float(v) for k, v in pv.get("C", pd.Series(dtype=float)).items()}
        put_vol_map  = {float(k): float(v) for k, v in pv.get("P", pd.Series(dtype=float)).items()}
        net_tbl["call_vol"] = net_tbl["K"].map(call_vol_map).fillna(0.0)
        net_tbl["put_vol"]  = net_tbl["K"].map(put_vol_map).fillna(0.0)

        if net_tbl.empty:
            results[exp] = net_tbl
            continue

        # 2) strikes окна и контекст для PZ/ER
        strikes_eval = _window_strikes(df_corr, exp, windows)
        series_ctx_map = _series_ctx_from_corr(df_corr, exp)
        if exp not in series_ctx_map:
            # если не удалось собрать контекст — вернём без PZ/ER
            net_tbl["PZ"] = 0.0; net_tbl["ER_Up"] = 0.0; net_tbl["ER_Down"] = 0.0
            results[exp] = net_tbl
            continue

        # 3) PZ/ER по формуле проекта
        pz, er_up, er_down = compute_power_zone_and_er(
            S=float(np.nanmedian(df_corr.loc[df_corr["exp"]==exp, "S"].values)),
            strikes_eval=strikes_eval,
            all_series_ctx=[series_ctx_map[exp]],
            day_high=cfg.day_high,
            day_low=cfg.day_low,
        )

        # 4) привязка PZ/ER к таблице по K
        pz_map   = {float(k): float(v) for k, v in zip(strikes_eval, pz)}
        erup_map = {float(k): float(v) for k, v in zip(strikes_eval, er_up)}
        erdn_map = {float(k): float(v) for k, v in zip(strikes_eval, er_down)}
        net_tbl["PZ"]      = net_tbl["K"].map(pz_map).fillna(0.0)
        net_tbl["ER_Up"]   = net_tbl["K"].map(erup_map).fillna(0.0)
        net_tbl["ER_Down"] = net_tbl["K"].map(erdn_map).fillna(0.0)

        # Упорядочим колонки
        cols = ["exp","K","S"] + (["F"] if "F" in net_tbl.columns else []) + \
               ["call_oi","put_oi","call_vol","put_vol","dg1pct_call","dg1pct_put","AG_1pct","NetGEX_1pct"]
        if "AG_1pct_M" in net_tbl.columns:
            cols += ["AG_1pct_M","NetGEX_1pct_M"]
        cols += ["PZ","ER_Up","ER_Down"]
        net_tbl = net_tbl[cols].sort_values("K").reset_index(drop=True)

        results[exp] = net_tbl

    return results


def process_from_raw(
    raw_records: List[dict],
    S: float,
    sanitizer_cfg: Optional[dict] = None,
    final_cfg: FinalTableConfig = FinalTableConfig(),
) -> Dict[str, pd.DataFrame]:
    """
    Полный цикл: сырые записи -> санитайз -> окна -> NetGEX/AG -> PZ/ER -> финальные таблицы.
    Возвращает словарь {exp: DataFrame}.
    """
    sanitizer_cfg = sanitizer_cfg or {}
    s_cfg = SanitizerConfig(**sanitizer_cfg)
    # Flatten raw records from Polygon snapshots into the structure
    # expected by sanitize_window.build_raw_table.  If flattening
    # fails for any reason, fall back to using raw_records directly.
    try:
        flat_records = _flatten_polygon_snapshot_records(raw_records)
    except Exception:
        flat_records = raw_records

    res = sanitize_and_window_pipeline(flat_records, S=S, cfg=s_cfg)
    df_corr = res["df_corr"]
    windows = res["windows"]

    return build_final_tables_from_corr(df_corr, windows, cfg=final_cfg)
