
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
__all__ = ['FinalTableConfig', 'build_final_tables_from_corr', 'build_final_sum_from_corr', 'process_from_raw', '_series_ctx_from_corr']


from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import math

# Внешние модули пайплайна (должны быть в PYTHONPATH или рядом)
from lib.sanitize_window import (
    SanitizerConfig,
    sanitize_and_window_pipeline,
)
from lib.netgex_ag import (
    NetGEXAGConfig,
    compute_netgex_ag_per_expiry,
)
from lib.power_zone_er import compute_power_zone


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
            net_tbl["PZ"] = 0.0
            results[exp] = net_tbl
            continue

        # 3) PZ/ER по формуле проекта
        pz = compute_power_zone(
            S=float(np.nanmedian(df_corr.loc[df_corr["exp"]==exp, "S"].values)),
            strikes_eval=strikes_eval,
            all_series_ctx=[series_ctx_map[exp]],
            day_high=cfg.day_high,
            day_low=cfg.day_low,
        )

        # 4) привязка PZ/ER к таблице по K
        pz_map   = {float(k): float(v) for k, v in zip(strikes_eval, pz)}
        net_tbl["PZ"]      = net_tbl["K"].map(pz_map).fillna(0.0)

        # Упорядочим колонки
        cols = ["exp","K","S"] + (["F"] if "F" in net_tbl.columns else []) + \
               ["call_oi","put_oi","call_vol","put_vol","dg1pct_call","dg1pct_put","AG_1pct","NetGEX_1pct"]
        if "AG_1pct_M" in net_tbl.columns:
            cols += ["AG_1pct_M","NetGEX_1pct_M"]
        cols += ["PZ"]
        net_tbl = net_tbl[cols].sort_values("K").reset_index(drop=True)

        results[exp] = net_tbl

    return results



def build_final_sum_from_corr(
    df_corr: pd.DataFrame,
    windows: Dict[str, np.ndarray],
    selected_exps: Optional[List[str]] = None,
    weight_mode: str = "1/√T",
    cfg: FinalTableConfig = FinalTableConfig(),
) -> pd.DataFrame:
    """Суммарная финальная таблица по нескольким экспирациям (Multi) на единой сетке K.

    Возвращает DataFrame с колонками:
      K, S[, F], call_oi, put_oi, call_vol, put_vol, AG_1pct, NetGEX_1pct, [AG_1pct_M, NetGEX_1pct_M], PZ.
    """
    # 0) Список экспираций
    exps_all = sorted(df_corr.get("exp", pd.Series(dtype=str)).dropna().unique().tolist())
    exp_list = [e for e in (selected_exps or exps_all) if e in exps_all]
    if not exp_list:
        return pd.DataFrame(columns=["K","S","call_oi","put_oi","call_vol","put_vol","AG_1pct","NetGEX_1pct","PZ"])

    # 1) Весовой коэффициент по T
    t_map: Dict[str, float] = {}
    for e in exp_list:
        g = df_corr[df_corr["exp"] == e]
        T_med = float(np.nanmedian(g.get("T", pd.Series(dtype=float)).values)) if len(g) else 1.0/252.0
        if not np.isfinite(T_med) or T_med <= 0:
            T_med = 1.0/252.0
        t_map[e] = T_med

    w_raw: Dict[str, float] = {}
    for e in exp_list:
        if weight_mode == "1/T":
            w_raw[e] = 1.0 / max(t_map[e], 1.0/252.0)
        elif weight_mode in ("1/√T","1/sqrt(T)","1/sqrtT","1/sqT"):
            w_raw[e] = 1.0 / np.sqrt(max(t_map[e], 1.0/252.0))
        else:
            w_raw[e] = 1.0
    w_sum = sum(w_raw.values()) or 1.0
    weights: Dict[str, float] = {e: w_raw[e]/w_sum for e in exp_list}

    # 2) Пер‑эксп. таблицы NetGEX/AG на окне
    per_exp: Dict[str, pd.DataFrame] = {}
    for e in exp_list:
        net_tbl = compute_netgex_ag_per_expiry(
            df_corr, e, windows=windows,
            cfg=NetGEXAGConfig(scale=cfg.scale_millions, aggregate="none")
        )
        keep = [c for c in ["K","AG_1pct","NetGEX_1pct","S","F"] if c in net_tbl.columns]
        per_exp[e] = net_tbl[keep].copy()

    # 3) Единая сетка K = объединение окон по выбранным экспирациям
    K_sets = []
    for e in exp_list:
        Ks = _window_strikes(df_corr, e, windows)
        if Ks.size:
            K_sets.append(set(map(float, Ks.tolist())))
    if not K_sets:
        # нет окон -> берём все страйки из df_corr
        K_union = sorted(set(map(float, df_corr.loc[df_corr["exp"].isin(exp_list), "K"].unique().tolist())))
    else:
        K_union = sorted(set().union(*K_sets))
    base = pd.DataFrame({"K": list(K_union)}, dtype=float)

    # 4) Медианные S/F по сериям
    S_vals = []
    F_vals = []
    for e, nt in per_exp.items():
        if "S" in nt.columns and nt["S"].notna().any():
            S_vals.append(float(np.nanmedian(nt["S"].astype(float))))
        if "F" in nt.columns and nt["F"].notna().any():
            F_vals.append(float(np.nanmedian(nt["F"].astype(float))))
    base["S"] = float(np.nanmedian(S_vals)) if S_vals else np.nan
    if F_vals:
        base["F"] = float(np.nanmedian(F_vals))

    # 5) Сумма AG/NetGEX с весами на единой сетке
    base["AG_1pct"] = 0.0
    base["NetGEX_1pct"] = 0.0
    for e, nt in per_exp.items():
        m = nt.groupby("K")[ ["AG_1pct","NetGEX_1pct"] ].sum()
        base = base.merge(m, left_on="K", right_index=True, how="left", suffixes=("","_add")).fillna(0.0)
        base["AG_1pct"] += weights[e] * base.pop("AG_1pct_add")
        base["NetGEX_1pct"] += weights[e] * base.pop("NetGEX_1pct_add")

    # 6) Совокупные OI и Volume по сторонам без весов
    g_all = df_corr[df_corr["exp"].isin(exp_list)].copy()
    agg_oi  = g_all.groupby(["K","side"], as_index=False)["oi"].sum()
    agg_vol = g_all.groupby(["K","side"], as_index=False)["vol"].sum()
    piv_oi  = agg_oi.pivot_table(index="K", columns="side", values="oi", aggfunc="sum").fillna(0.0)
    piv_vol = agg_vol.pivot_table(index="K", columns="side", values="vol", aggfunc="sum").fillna(0.0)
    base["call_oi"] = piv_oi.get("C", pd.Series(dtype=float)).reindex(base["K"], fill_value=0.0).to_numpy()
    base["put_oi"]  = piv_oi.get("P", pd.Series(dtype=float)).reindex(base["K"], fill_value=0.0).to_numpy()
    base["call_vol"] = piv_vol.get("C", pd.Series(dtype=float)).reindex(base["K"], fill_value=0.0).to_numpy()
    base["put_vol"]  = piv_vol.get("P", pd.Series(dtype=float)).reindex(base["K"], fill_value=0.0).to_numpy()

    # 7) Масштабные колонки в млн $
    if cfg.scale_millions and cfg.scale_millions > 0:
        base["AG_1pct_M"] = base["AG_1pct"] / cfg.scale_millions
        base["NetGEX_1pct_M"] = base["NetGEX_1pct"] / cfg.scale_millions

    # 8) Power Zone на объединённой сетке: gamma_*_share домножаем на веса
    all_ctx: List[dict] = []
    for e in exp_list:
        series_map = _series_ctx_from_corr(df_corr, e)
        if e not in series_map:
            continue
        ctx = dict(series_map[e])
        if "gamma_abs_share" in ctx:
            ctx["gamma_abs_share"] = (np.array(ctx["gamma_abs_share"], dtype=float) * weights[e]).tolist()
        if "gamma_net_share" in ctx:
            ctx["gamma_net_share"] = (np.array(ctx["gamma_net_share"], dtype=float) * weights[e]).tolist()
        all_ctx.append(ctx)
    strikes_eval = base["K"].astype(float).tolist()
    pz = compute_power_zone(
        S=(float(np.nanmedian(base["S"].astype(float))) if ("S" in base.columns and pd.notna(base["S"]).any()) else float("nan")),
        strikes_eval=strikes_eval,
        all_series_ctx=all_ctx,
        day_high=getattr(cfg, "day_high", None),
        day_low=getattr(cfg, "day_low", None),
    )
    base["PZ"] = pd.Series(pz, index=base.index).astype(float)

    # 9) Финальные колонки и порядок
    cols = ["K","S"] + (["F"] if "F" in base.columns else []) + ["call_oi","put_oi","call_vol","put_vol","AG_1pct","NetGEX_1pct"]
    if "AG_1pct_M" in base.columns:
        cols += ["AG_1pct_M","NetGEX_1pct_M"]
    cols += ["PZ"]
    return base[cols].sort_values("K").reset_index(drop=True)
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
    res = sanitize_and_window_pipeline(raw_records, S=S, cfg=s_cfg)
    df_corr = res["df_corr"]; windows = res["windows"]

    return build_final_tables_from_corr(df_corr, windows, cfg=final_cfg)



# ===== Patched implementation: build_final_sum_from_corr (Multi fix) =====
def build_final_sum_from_corr(
    df_corr: pd.DataFrame,
    windows: Dict[str, np.ndarray],
    selected_exps: Optional[List[str]] = None,
    weight_mode: str = "1/√T",
    cfg: FinalTableConfig = FinalTableConfig(),
) -> pd.DataFrame:
    """
    Суммирует по всем выбранным экспирациям на объединённой сетке K окна.
    Весовая схема: равные | 1/T | 1/√T применяется к AG и NetGEX. OI/Vol суммируются без весов.
    PZ пересчитывается на объединённой сетке с учётом весов (через compute_power_zone).
    """
    if df_corr is None or len(df_corr) == 0:
        return pd.DataFrame(columns=["K","S","call_oi","put_oi","call_vol","put_vol","AG_1pct","NetGEX_1pct","PZ"])

    # 0) список экспираций
    exps_all = sorted(pd.Series(df_corr.get("exp", pd.Series(dtype=str))).dropna().unique().tolist())
    exp_list = [e for e in (selected_exps or exps_all) if e in exps_all]
    if not exp_list:
        return pd.DataFrame(columns=["K","S","call_oi","put_oi","call_vol","put_vol","AG_1pct","NetGEX_1pct","PZ"])

    # 1) веса по T
    t_map: Dict[str, float] = {}
    for e in exp_list:
        g = df_corr[df_corr["exp"] == e]
        T_vals = pd.to_numeric(g.get("T", pd.Series(dtype=float)), errors="coerce").dropna().values
        T_med = float(np.nanmedian(T_vals)) if T_vals.size else 1.0/252.0
        if not (np.isfinite(T_med) and T_med > 0):
            T_med = 1.0/252.0
        t_map[e] = T_med

    w_raw: Dict[str, float] = {}
    for e in exp_list:
        if weight_mode == "1/T":
            w_raw[e] = 1.0 / max(t_map[e], 1.0/252.0)
        elif weight_mode in ("1/√T","1/sqrt(T)","1/sqrtT"):
            w_raw[e] = 1.0 / math.sqrt(max(t_map[e], 1.0/252.0))
        else:
            w_raw[e] = 1.0
    w_sum = sum(w_raw.values()) or 1.0
    weights: Dict[str, float] = {e: w_raw[e]/w_sum for e in exp_list}

    # 2) объединённая сетка K по окнам
    Ks_union: List[float] = []
    for e in exp_list:
        arr = windows.get(e, None)
        if arr is None:
            continue
        try:
            Ks_union.extend([float(x) for x in (arr.tolist() if hasattr(arr, "tolist") else list(arr))])
        except Exception:
            pass
    Ks = sorted(set(Ks_union))
    if not Ks:
        return pd.DataFrame(columns=["K","S","call_oi","put_oi","call_vol","put_vol","AG_1pct","NetGEX_1pct","PZ"])

    # 3) базовая таблица с объединёнными метриками
    base = pd.DataFrame({"K": Ks})
    # S и F как медиана по всем экспирациям на выбранном K
    def _median_series(col: str) -> pd.Series:
        vals = []
        for e in exp_list:
            g = df_corr[(df_corr["exp"]==e) & (df_corr["K"].isin(Ks))]
            if col in g.columns:
                vals.append(pd.to_numeric(g[col], errors="coerce"))
        if not vals:
            return pd.Series([np.nan]*len(Ks))
        s = pd.concat(vals, axis=0)
        med = float(np.nanmedian(pd.to_numeric(s, errors="coerce").dropna().values)) if not s.dropna().empty else np.nan
        return pd.Series([med]*len(Ks))
    base["S"] = _median_series("S")
    if "F" in df_corr.columns:
        base["F"] = _median_series("F")

    # 4) агрегирование по экспирациям: OI/Vol суммой, AG/NetGEX с весами
    for col in ["call_oi","put_oi","call_vol","put_vol","AG_1pct","NetGEX_1pct"]:
        base[col] = 0.0

    for e in exp_list:
        g = df_corr[(df_corr["exp"]==e) & (df_corr["K"].isin(Ks))].copy()
        # группировка по страйку
        if not len(g):
            continue
        g["K"] = pd.to_numeric(g["K"], errors="coerce")
        g = g.dropna(subset=["K"])
        grouped = g.groupby("K", as_index=True).agg({
            "call_oi":"sum","put_oi":"sum","call_vol":"sum","put_vol":"sum",
            "AG_1pct":"sum","NetGEX_1pct":"sum"
        })
        grouped = grouped.reset_index()
        grouped.rename(columns={"K":"K_key"}, inplace=True)
        # merge
        base = base.merge(grouped, left_on="K", right_on="K_key", how="left", suffixes=("","__add"))
        # add
        for col in ["call_oi","put_oi","call_vol","put_vol"]:
            if f"{col}" in grouped.columns:
                base[col] = base[col] + base[f"{col}__add"].fillna(0.0)
        # weighted add for gamma metrics
        for col in ["AG_1pct","NetGEX_1pct"]:
            if f"{col}" in grouped.columns:
                base[col] = base[col] + weights[e] * base[f"{col}__add"].fillna(0.0)
        # drop helper
        drop_cols = [c for c in base.columns if c.endswith("__add") or c=="K_key"]
        base.drop(columns=drop_cols, inplace=True)

    # 5) масштабные столбцы
    if getattr(cfg, "scale_millions", None):
        scale = float(cfg.scale_millions)
        if scale > 0:
            base["AG_1pct_M"] = base["AG_1pct"] / scale
            base["NetGEX_1pct_M"] = base["NetGEX_1pct"] / scale

    # 6) Power Zone: соберём контексты по экспирациям и взвесим шары гаммы
    all_ctx: List[dict] = []
    try:
        series_map_all = {}
        for e in exp_list:
            try:
                series_map = _series_ctx_from_corr(df_corr, e)  # {e: {...}}
            except Exception:
                series_map = {}
            if e in series_map:
                ctx = dict(series_map[e])
                # домножаем шары на веса
                if "gamma_abs_share" in ctx:
                    ctx["gamma_abs_share"] = (np.array(ctx["gamma_abs_share"], dtype=float) * weights[e]).tolist()
                if "gamma_net_share" in ctx:
                    ctx["gamma_net_share"] = (np.array(ctx["gamma_net_share"], dtype=float) * weights[e]).tolist()
                all_ctx.append(ctx)
        strikes_eval = base["K"].astype(float).tolist()
        S_med = float(np.nanmedian(pd.to_numeric(base["S"], errors="coerce").dropna().values)) if "S" in base.columns else np.nan
        pz = compute_power_zone(
            S=S_med,
            strikes_eval=strikes_eval,
            all_series_ctx=all_ctx,
            day_high=getattr(cfg, "day_high", None),
            day_low=getattr(cfg, "day_low", None),
        )
        base["PZ"] = pd.Series(pz, index=base.index).astype(float)
    except Exception:
        base["PZ"] = np.nan

    # 7) Порядок колонок
    cols = ["K","S"] + (["F"] if "F" in base.columns else []) + ["call_oi","put_oi","call_vol","put_vol","AG_1pct","NetGEX_1pct"]
    if "AG_1pct_M" in base.columns:
        cols += ["AG_1pct_M","NetGEX_1pct_M"]
    cols += ["PZ"]
    base = base[cols].sort_values("K").reset_index(drop=True)
    return base
# ===== End patched implementation =====

# --- override build_final_sum_from_corr (appended) ---

def build_final_sum_from_corr(
    df_corr: pd.DataFrame,
    windows: Dict[str, np.ndarray],
    selected_exps: Optional[List[str]] = None,
    weight_mode: str = "1/√T",
    cfg: FinalTableConfig = FinalTableConfig(),
) -> pd.DataFrame:
    """
    Суммирует по всем выбранным экспирациям на объединённой сетке K окна.
    Весовая схема: равные | 1/T | 1/√T применяется к AG и NetGEX. OI/Vol суммируются без весов.
    PZ пересчитывается на объединённой сетке с учётом весов (через compute_power_zone).
    """
    if df_corr is None or len(df_corr) == 0:
        return pd.DataFrame(columns=["K","S","call_oi","put_oi","call_vol","put_vol","AG_1pct","NetGEX_1pct","PZ"])

    from lib.netgex_ag import compute_netgex_ag_per_expiry, NetGEXAGConfig

    # 0) список экспираций
    exps_all = sorted(pd.Series(df_corr.get("exp", pd.Series(dtype=str))).dropna().unique().tolist())
    exp_list = [e for e in (selected_exps or exps_all) if e in exps_all]
    if not exp_list:
        return pd.DataFrame(columns=["K","S","call_oi","put_oi","call_vol","put_vol","AG_1pct","NetGEX_1pct","PZ"])

    # 1) веса по T
    t_map: Dict[str, float] = {}
    for e in exp_list:
        g = df_corr[df_corr["exp"] == e]
        T_vals = pd.to_numeric(g.get("T", pd.Series(dtype=float)), errors="coerce").dropna().values
        T_med = float(np.nanmedian(T_vals)) if T_vals.size else 1.0/252.0
        if not (np.isfinite(T_med) and T_med > 0):
            T_med = 1.0/252.0
        t_map[e] = T_med

    w_raw: Dict[str, float] = {}
    for e in exp_list:
        if weight_mode == "1/T":
            w_raw[e] = 1.0 / max(t_map[e], 1.0/252.0)
        elif weight_mode in ("1/√T","1/sqrt(T)","1/sqrtT"):
            w_raw[e] = 1.0 / math.sqrt(max(t_map[e], 1.0/252.0))
        else:
            w_raw[e] = 1.0
    w_sum = sum(w_raw.values()) or 1.0
    weights: Dict[str, float] = {e: w_raw[e]/w_sum for e in exp_list}

    # 2) объединённая сетка K по окнам
    Ks_union: List[float] = []
    for e in exp_list:
        arr = windows.get(e, None)
        if arr is None:
            continue
        try:
            Ks_union.extend([float(x) for x in (arr.tolist() if hasattr(arr, "tolist") else list(arr))])
        except Exception:
            pass
    Ks = sorted(set(Ks_union))
    if not Ks:
        return pd.DataFrame(columns=["K","S","call_oi","put_oi","call_vol","put_vol","AG_1pct","NetGEX_1pct","PZ"])

    # 3) посчитать NetGEX/AG отдельно по каждой экспирации на её окне и слить
    per_exp_frames: Dict[str, pd.DataFrame] = {}
    for e in exp_list:
        try:
            per_exp_frames[e] = compute_netgex_ag_per_expiry(df_corr=df_corr, exp=e, windows=windows, cfg=NetGEXAGConfig())
        except Exception:
            per_exp_frames[e] = pd.DataFrame(columns=["exp","K","S","F","call_oi","put_oi","call_vol","put_vol","dg1pct_call","dg1pct_put","AG_1pct","NetGEX_1pct"])

    # 4) собрать объединённую таблицу
    base = pd.DataFrame({"K": Ks})
    # S и F как медиана
    def _median_from_frames(col: str) -> pd.Series:
        vals = []
        for e, df in per_exp_frames.items():
            if col in df.columns:
                vals.append(pd.to_numeric(df[col], errors="coerce"))
        if not vals:
            return pd.Series([np.nan]*len(Ks))
        med = float(np.nanmedian(pd.concat(vals).dropna().values))
        return pd.Series([med]*len(Ks))
    base["S"] = _median_from_frames("S")
    if any("F" in df.columns for df in per_exp_frames.values()):
        base["F"] = _median_from_frames("F")

    # init sum columns
    for col in ["call_oi","put_oi","call_vol","put_vol","AG_1pct","NetGEX_1pct"]:
        base[col] = 0.0

    for e, df in per_exp_frames.items():
        if df is None or df.empty:
            continue
        df_e = df[df["K"].isin(Ks)].copy()
        cols = ["K","call_oi","put_oi","AG_1pct","NetGEX_1pct"] + ([ "call_vol","put_vol"] if set(["call_vol","put_vol"]).issubset(df_e.columns) else [])
        df_e = df_e[cols].copy()
        base = base.merge(df_e, on="K", how="left", suffixes=("","__add"))
        # sum non-weighted
        for col in ["call_oi","put_oi","call_vol","put_vol"]:
            if f"{col}__add" in base.columns:
                base[col] = base[col] + base[f"{col}__add"].fillna(0.0)
        # weighted for gamma metrics
        for col in ["AG_1pct","NetGEX_1pct"]:
            if f"{col}__add" in base.columns:
                base[col] = base[col] + weights.get(e, 0.0) * base[f"{col}__add"].fillna(0.0)
        # clean
        drop_cols = [c for c in base.columns if c.endswith("__add")]
        base.drop(columns=drop_cols, inplace=True)

    # 5) масштаб
    if getattr(cfg, "scale_millions", None):
        scale = float(cfg.scale_millions)
        if scale > 0:
            base["AG_1pct_M"] = base["AG_1pct"] / scale
            base["NetGEX_1pct_M"] = base["NetGEX_1pct"] / scale

    # 6) Power Zone: из df_corr, но шары гаммы взвесим на weights
    all_ctx: List[dict] = []
    try:
        for e in exp_list:
            series_map = _series_ctx_from_corr(df_corr, e)
            if e in series_map:
                ctx = dict(series_map[e])
                if "gamma_abs_share" in ctx:
                    ctx["gamma_abs_share"] = (np.array(ctx["gamma_abs_share"], dtype=float) * weights[e]).tolist()
                if "gamma_net_share" in ctx:
                    ctx["gamma_net_share"] = (np.array(ctx["gamma_net_share"], dtype=float) * weights[e]).tolist()
                all_ctx.append(ctx)
        strikes_eval = base["K"].astype(float).tolist()
        S_med = float(np.nanmedian(pd.to_numeric(base["S"], errors="coerce").dropna().values)) if "S" in base.columns else np.nan
        pz = compute_power_zone(
            S=S_med,
            strikes_eval=strikes_eval,
            all_series_ctx=all_ctx,
            day_high=getattr(cfg, "day_high", None),
            day_low=getattr(cfg, "day_low", None),
        )
        base["PZ"] = pd.Series(pz, index=base.index).astype(float)
    except Exception:
        base["PZ"] = np.nan

    cols = ["K","S"] + (["F"] if "F" in base.columns else []) + ["call_oi","put_oi","call_vol","put_vol","AG_1pct","NetGEX_1pct"]
    if "AG_1pct_M" in base.columns:
        cols += ["AG_1pct_M","NetGEX_1pct_M"]
    cols += ["PZ"]
    base = base[cols].sort_values("K").reset_index(drop=True)
    return base

