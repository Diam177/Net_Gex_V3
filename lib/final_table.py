
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

        results[exp] =

def build_final_sum_from_corr(
    df_corr: pd.DataFrame,
    windows: Dict[str, np.ndarray],
    selected_exps: List[str],
    weight_mode: str = "1/T",
    cfg: FinalTableConfig = FinalTableConfig(),
) -> pd.DataFrame:
    """
    Построение агрегированной финальной таблицы (Multi) по выбранным экспирациям.
    Логика совпадает с резервной веткой в app.py:
      - веса по экспирациям: "1/T" или "equal"
      - AG_1pct и NetGEX_1pct — взвешенная сумма по сериям
      - call_oi / put_oi — простая сумма
      - S и F — медианы по сериям
      - PZ — по compute_power_zone на единой сетке K и контекстам серий, домноженным на веса
    Возвращает DataFrame c колонками: K, S, [F], call_oi, put_oi, AG_1pct, NetGEX_1pct, [AG_1pct_M, NetGEX_1pct_M], PZ
    """
    # 0) список экспираций
    all_exps = sorted([e for e in selected_exps if e in df_corr.get("exp", pd.Series(dtype=str)).unique().tolist()])
    if not all_exps:
        return pd.DataFrame(columns=["K","S","call_oi","put_oi","AG_1pct","NetGEX_1pct","PZ"])
    # 1) веса
    t_map: Dict[str, float] = {}
    for e in all_exps:
        T_vals = df_corr.loc[df_corr["exp"] == e, "T"].dropna().values
        T_med = float(np.nanmedian(T_vals)) if T_vals.size else float("nan")
        if not (np.isfinite(T_med) and T_med > 0):
            T_med = 1.0/252.0
        t_map[e] = T_med
    w_raw: Dict[str, float] = {}
    for e in all_exps:
        if str(weight_mode).lower() in ["1/t","1t","inv_t","inverse_t"]:
            w_raw[e] = 1.0 / max(t_map.get(e, 1.0/252.0), 1e-12)
        elif str(weight_mode).lower() in ["equal","eq","1"]:
            w_raw[e] = 1.0
        else:
            w_raw[e] = 1.0 / max(t_map.get(e, 1.0/252.0), 1e-12)
    w_sum = float(sum(w_raw.values())) or 1.0
    weights = {e: w_raw[e]/w_sum for e in all_exps}

    # 2) финальные таблицы по каждой экспирации (Single-логика на её окне)
    per_exp: Dict[str, pd.DataFrame] = {}
    for e in all_exps:
        nt = compute_netgex_ag_per_expiry(df_corr, e, windows, cfg=NetGEXAGConfig(scale=getattr(cfg, "scale_millions", 1e6)))
        keep_cols = ["K","AG_1pct","NetGEX_1pct"] + [c for c in ["S","F"] if c in nt.columns]
        per_exp[e] = nt[keep_cols].copy()

    # 3) объединённая сетка K
    all_K = sorted(set().union(*[set(df["K"].astype(float).tolist()) for df in per_exp.values() if not df.empty]))
    base = pd.DataFrame({"K": all_K})
    # S и F как медианы по сериям
    S_vals = []
    F_vals = []
    for e, nt in per_exp.items():
        if "S" in nt.columns and nt["S"].notna().any():
            S_vals.append(float(np.nanmedian(nt["S"])))
        if "F" in nt.columns and nt["F"].notna().any():
            F_vals.append(float(np.nanmedian(nt["F"])))
    base["S"] = float(np.nanmedian(S_vals)) if S_vals else np.nan
    if F_vals:
        base["F"] = float(np.nanmedian(F_vals))

    # 4) накопление AG/NetGEX
    base["AG_1pct"] = 0.0
    base["NetGEX_1pct"] = 0.0
    for e, nt in per_exp.items():
        m = nt.groupby("K")[["AG_1pct","NetGEX_1pct"]].sum()
        base = base.merge(m, left_on="K", right_index=True, how="left", suffixes=("","_add")).fillna(0.0)
        base["AG_1pct"] += weights[e] * base.pop("AG_1pct_add")
        base["NetGEX_1pct"] += weights[e] * base.pop("NetGEX_1pct_add")

    # 5) OI суммы
    g = df_corr[df_corr["exp"].isin(all_exps)].copy()
    agg_oi = g.groupby(["K","side"], as_index=False)["oi"].sum()
    piv_oi = agg_oi.pivot_table(index="K", columns="side", values="oi", aggfunc="sum").fillna(0.0)
    base["call_oi"] = piv_oi.get("C", pd.Series(dtype=float)).reindex(base["K"], fill_value=0.0).to_numpy()
    base["put_oi"]  = piv_oi.get("P", pd.Series(dtype=float)).reindex(base["K"], fill_value=0.0).to_numpy()

    # 6) масштаб в млн $
    scale_val = float(getattr(cfg, "scale_millions", 1e6) or 0.0)
    if scale_val > 0.0:
        base["AG_1pct_M"]     = base["AG_1pct"] / scale_val
        base["NetGEX_1pct_M"] = base["NetGEX_1pct"] / scale_val

    # 7) PZ по агрегированному контексту с весами
    all_ctx = []
    for e in all_exps:
        ctx_map = _series_ctx_from_corr(df_corr, e)
        if e in ctx_map:
            ctx = dict(ctx_map[e])
            if "gamma_abs_share" in ctx:
                ctx["gamma_abs_share"] = np.array(ctx["gamma_abs_share"], dtype=float) * weights[e]
            if "gamma_net_share" in ctx:
                ctx["gamma_net_share"] = np.array(ctx["gamma_net_share"], dtype=float) * weights[e]
            all_ctx.append(ctx)
    strikes_eval = base["K"].astype(float).tolist()
    if strikes_eval and all_ctx:
        pz = compute_power_zone(
            S=base["S"].astype(float).median() if "S" in base.columns else float("nan"),
            strikes_eval=strikes_eval,
            all_series_ctx=all_ctx,
            day_high=getattr(cfg, "day_high", None),
            day_low=getattr(cfg, "day_low", None),
        )
        base["PZ"] = pd.Series(pz, index=base.index)
    else:
        base["PZ"] = 0.0

    return base[["K"] + ([c for c in ["S","F"] if c in base.columns]) + ["call_oi","put_oi","AG_1pct","NetGEX_1pct"] + ([c for c in ["AG_1pct_M","NetGEX_1pct_M"] if c in base.columns]) + ["PZ"]]

 net_tbl

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
    res = sanitize_and_window_pipeline(raw_records, S=S, cfg=s_cfg)
    df_corr = res["df_corr"]; windows = res["windows"]

    return build_final_tables_from_corr(df_corr, windows, cfg=final_cfg)
