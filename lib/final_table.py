
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
        "gamma_abs_share": [gamma_abs_share[float(k)] for k in Ks],
        "gamma_net_share": [gamma_net_share[float(k)] for k in Ks],
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
    """
    Multi: то же окно, что в Single (те же p_cover, nmin, nmax, тот же алгоритм),
    но масса для выбора окна = суммарная по выбранным экспирациям из |NetGEX_e(K)|,
    нормированная внутри каждой серии и взвешенная по времени (или равными весами).
    Затем на выбранном Multi-окне агрегируем AG/NetGEX (с весами) и OI/Vol (без весов),
    и считаем PZ на этой же сетке без повторного time-weight (его делает power_zone).
    """
    # 0) Список экспираций
    exps_all = sorted(df_corr.get("exp", pd.Series(dtype=str)).dropna().unique().tolist())
    exp_list = [e for e in (selected_exps or exps_all) if e in exps_all]
    if not exp_list:
        return pd.DataFrame(columns=["K","S","F","call_oi","put_oi","call_vol","put_vol","AG_1pct","NetGEX_1pct","AG_1pct_M","NetGEX_1pct_M","PZ"])

    # 1) Соберём per-exp "оконные" DataFrame и базовые величины
    def _window_strikes_local(df_corr_local: pd.DataFrame, exp: str, windows_map: Dict[str, np.ndarray]) -> np.ndarray:
        Ks = np.array(sorted(df_corr_local.loc[df_corr_local["exp"]==exp, "K"].unique()), dtype=float)
        idx = np.array(windows_map.get(exp, []), dtype=int)
        if Ks.size == 0:
            return Ks
        if idx.size == 0 or idx.max() >= Ks.size or idx.min() < 0:
            # если нет окна — считаем по всей серии, но затем Multi-окно урежет
            return Ks
        return Ks[idx]

    per_exp = {}
    for e in exp_list:
        g = df_corr[df_corr["exp"]==e].copy()
        if g.empty:
            continue
        # Обрезаем до single-окна этой экспирации
        Ks_e = _window_strikes_local(df_corr, e, windows)
        if Ks_e.size:
            g = g[g["K"].isin(Ks_e)].copy()
        # Опорные значения
        S_e = float(np.nanmedian(g["S"].values)) if "S" in g.columns and len(g) else float("nan")
        F_e = float(np.nanmedian(g["F"].values)) if "F" in g.columns and len(g) else float("nan")
        T_e = float(np.nanmedian(g["T"].values)) if "T" in g.columns and len(g) else 0.0
        T_e = max(T_e, 1.0/252.0)

        # Dollar gamma на 1% * OI по строкам
        ok_mask = (
            np.isfinite(g["gamma_corr"]) & (g["gamma_corr"]>0) &
            np.isfinite(g["S"]) & np.isfinite(g["mult"]) &
            np.isfinite(g["oi"]) & (g["oi"]>0)
        )
        dg_row = np.where(ok_mask,
                          g["gamma_corr"].to_numpy() * (g["S"].to_numpy()**2) * 0.01 * g["mult"].to_numpy() * g["oi"].to_numpy(),
                          0.0)
        g = g.assign(dg1pct_row=dg_row)

        # Агрегаты по страйкам/сторонам
        agg = g.groupby(["K","side"], as_index=False)[["dg1pct_row","oi","vol"]].sum()
        p_dg = agg.pivot_table(index="K", columns="side", values="dg1pct_row", aggfunc="sum").fillna(0.0)
        p_oi = agg.pivot_table(index="K", columns="side", values="oi", aggfunc="sum").fillna(0.0)
        p_vol= agg.pivot_table(index="K", columns="side", values="vol", aggfunc="sum").fillna(0.0)

        Ks_sorted = np.array(sorted(agg["K"].unique()), dtype=float)
        dgC = p_dg.get("C", pd.Series(0.0, index=[])).reindex(Ks_sorted, fill_value=0.0).to_numpy(dtype=float)
        dgP = p_dg.get("P", pd.Series(0.0, index=[])).reindex(Ks_sorted, fill_value=0.0).to_numpy(dtype=float)
        AG = dgC + dgP
        Net = dgC - dgP

        # Масса для окна внутри серии: |Net| нормированная
        m = np.abs(Net)
        m_norm = m / m.sum() if m.sum()>0 else np.zeros_like(m)

        per_exp[e] = {
            "Ks": Ks_sorted,
            "S": S_e, "F": F_e, "T": T_e,
            "AG": AG, "Net": Net, "m_norm": m_norm,
            "call_oi": p_oi.get("C", pd.Series(0.0, index=[])).reindex(Ks_sorted, fill_value=0.0).to_numpy(dtype=float),
            "put_oi":  p_oi.get("P", pd.Series(0.0, index=[])).reindex(Ks_sorted, fill_value=0.0).to_numpy(dtype=float),
            "call_vol":p_vol.get("C", pd.Series(0.0, index=[])).reindex(Ks_sorted, fill_value=0.0).to_numpy(dtype=float),
            "put_vol": p_vol.get("P", pd.Series(0.0, index=[])).reindex(Ks_sorted, fill_value=0.0).to_numpy(dtype=float),
        }

    if not per_exp:
        return pd.DataFrame(columns=["K","S","F","call_oi","put_oi","call_vol","put_vol","AG_1pct","NetGEX_1pct","AG_1pct_M","NetGEX_1pct_M","PZ"])

    # 2) Веса серий
    def _w_from_T(T: float, mode: str) -> float:
        if mode == "1/T":
            return 1.0 / max(T, 1.0/252.0)
        if mode == "1/√T" or mode == "1/√T":
            return 1.0 / (max(T, 1.0/252.0) ** 0.5)
        # "равные" или любое иное
        return 1.0

    w_raw = {e: _w_from_T(per_exp[e]["T"], weight_mode) for e in per_exp.keys()}
    w_sum = sum(w_raw.values()) or 1.0
    w = {e: w_raw[e]/w_sum for e in per_exp.keys()}

    # 3) Кандидатная сетка страйков: объединение
    K_union = sorted(set(np.concatenate([per_exp[e]["Ks"] for e in per_exp.keys()])))
    K_union = np.array(K_union, dtype=float)

    # 4) Суммарная масса W(K) = Σ_e w_e * m̂_e(K), нормированная
    K_to_idx = {k:i for i,k in enumerate(K_union)}
    W = np.zeros_like(K_union, dtype=float)
    for e, ctx in per_exp.items():
        # сопоставим m_norm с K_union
        idxs = [K_to_idx.get(float(k)) for k in ctx["Ks"]]
        vals = ctx["m_norm"]
        for j, val in zip(idxs, vals):
            if j is not None:
                W[j] += w[e] * float(val)
    W_sum = W.sum()
    if W_sum > 0:
        W = W / W_sum
    else:
        # fallback: равномерно
        W[:] = 1.0 / max(len(W), 1)

    # 5) Выбор окна по той же процедуре, что в Single
    # Копия _select_window_for_exp из sanitize_window.py, чтобы не делать циклических импортов
    def _select_window_like_single(strikes: np.ndarray, weights: np.ndarray, S_anchor: float, p: float, nmin: int, nmax: int) -> np.ndarray:
        strikes = np.asarray(strikes, float)
        weights = np.asarray(weights, float)
        n = len(strikes)
        if n == 0:
            return np.arange(0, dtype=int)
        i_atm = int(np.argmin(np.abs(strikes - float(S_anchor))))
        L = R = i_atm
        w = np.where(~np.isfinite(weights) | (weights < 0), 0.0, weights)
        total = w.sum()
        need = float(p) * total if total>0 else 0.0
        acc = w[i_atm] if (0 <= i_atm < n) else 0.0
        # гарантируем минимум
        while (R-L+1) < int(max(1, nmin)):
            if (L>0) and (R<n-1):
                if w[L-1] >= w[R+1]: L -= 1
                else: R += 1
            elif L>0: L -= 1
            elif R<n-1: R += 1
            else: break
            acc += w[L] + (w[R] if R!=L else 0.0)
        # расширяем до покрытия p или до nmax
        while ((acc < need) if total>0 else False) and ((R-L+1) < int(max(1, nmax))):
            left_gain = w[L-1] if L>0 else -1.0
            right_gain = w[R+1] if R<n-1 else -1.0
            if left_gain >= right_gain and L>0:
                L -= 1; acc += w[L]
            elif R<n-1:
                R += 1; acc += w[R]
            else:
                break
        # ограничение по nmax
        while (R-L+1) > int(max(1, nmax)):
            # отрежем сторону с меньшей погрешностью
            if w[L] <= w[R]:
                L += 1
            else:
                R -= 1
        return np.arange(L, R+1, dtype=int)

    # Получаем параметры single-окна из sanitize_window.SanitizerConfig по умолчанию
    # Попробуем импортировать для доступа к p_cover/nmin/nmax, иначе используем дефолты
    p_cover = 0.90; nmin = 9; nmax = 33
    try:
        from lib.sanitize_window import SanitizerConfig  # type: ignore
        sc = SanitizerConfig()
        p_cover = getattr(sc, "p_cover", p_cover)
        nmin = getattr(sc, "nmin", nmin)
        nmax = getattr(sc, "nmax", nmax)
    except Exception:
        pass

    S_multi = float(np.nanmedian([ctx["S"] for ctx in per_exp.values()]))
    idx_win = _select_window_like_single(K_union, W, S_multi, p_cover, nmin, nmax)
    K_window = K_union[idx_win]

    # 6) Агрегация рядов на Multi-окне
    # Подготовим словари по K для ускорения
    K_to_pos = {k:i for i,k in enumerate(K_window)}
    AG_sum = np.zeros_like(K_window, dtype=float)
    NET_sum = np.zeros_like(K_window, dtype=float)
    call_oi_sum = np.zeros_like(K_window, dtype=float)
    put_oi_sum  = np.zeros_like(K_window, dtype=float)
    call_vol_sum= np.zeros_like(K_window, dtype=float)
    put_vol_sum = np.zeros_like(K_window, dtype=float)

    for e, ctx in per_exp.items():
        # карты
        map_idx = {float(k): i for i,k in enumerate(ctx["Ks"])}
        for k, pos in K_to_pos.items():
            i_src = map_idx.get(float(k))
            if i_src is None:
                continue
            AG_sum[pos]  += w[e] * float(ctx["AG"][i_src])
            NET_sum[pos] += w[e] * float(ctx["Net"][i_src])
            call_oi_sum[pos] += float(ctx["call_oi"][i_src])
            put_oi_sum[pos]  += float(ctx["put_oi"][i_src])
            call_vol_sum[pos]+= float(ctx["call_vol"][i_src])
            put_vol_sum[pos] += float(ctx["put_vol"][i_src])

    # 7) Сбор финальной таблицы
    base = pd.DataFrame({
        "K": K_window.astype(float),
        "S": S_multi,
    })
    # F_multi если есть
    if "F" in df_corr.columns and pd.notna(df_corr["F"]).any():
        F_multi = float(np.nanmedian([ctx["F"] for ctx in per_exp.values()]))
        base["F"] = F_multi
    base["call_oi"] = call_oi_sum
    base["put_oi"]  = put_oi_sum
    base["call_vol"]= call_vol_sum
    base["put_vol"] = put_vol_sum
    base["AG_1pct"] = AG_sum
    base["NetGEX_1pct"] = NET_sum
    scale = getattr(cfg, "scale_millions", 1_000_000.0)
    if scale and scale>0:
        base["AG_1pct_M"] = base["AG_1pct"] / float(scale)
        base["NetGEX_1pct_M"] = base["NetGEX_1pct"] / float(scale)

    # 8) Power Zone на той же сетке (без наложения w_e внутрь профилей)
    # Готовим all_series_ctx на исходных per-exp оконных данных, но compute_power_zone сам применит time-weight
    all_ctx = []
    for e in exp_list:
        # соберём контекст с долями gamma_abs/net из per_exp, урезанными на K_window
        if e not in per_exp:
            continue
        ctx = per_exp[e]
        Ks = ctx["Ks"]
        # доли
        ag = ctx["AG"]; net = ctx["Net"]
        sum_ag = float(np.nansum(ag)); sum_net_abs = float(np.nansum(np.abs(net)))
        gamma_abs_share = {float(k): (float(v)/sum_ag if sum_ag>0 else 0.0) for k, v in zip(Ks, ag)}
        gamma_net_share = {float(k): (float(v)/sum_net_abs if sum_net_abs>0 else 0.0) for k, v in zip(Ks, net)}
        # словари OI/Vol по K
        call_oi = {float(k): float(v) for k, v in zip(Ks, ctx["call_oi"])}
        put_oi  = {float(k): float(v) for k, v in zip(Ks, ctx["put_oi"])}
        call_vol= {float(k): float(v) for k, v in zip(Ks, ctx["call_vol"])}
        put_vol = {float(k): float(v) for k, v in zip(Ks, ctx["put_vol"])}
        # iv_call/iv_put приготовим как медиану по df_corr на окне
        g = df_corr[(df_corr["exp"]==e) & (df_corr["K"].isin(Ks))].copy()
        iv_call = g[g["side"]=="C"].groupby("K")["iv_corr"].median().to_dict()
        iv_put  = g[g["side"]=="P"].groupby("K")["iv_corr"].median().to_dict()

        all_ctx.append({
            "strikes": [float(k) for k in Ks],
            "gamma_abs_share": [gamma_abs_share[float(k)] for k in Ks],
            "gamma_net_share": [gamma_net_share[float(k)] for k in Ks],
            "call_oi": call_oi, "put_oi": put_oi,
            "call_vol": call_vol, "put_vol": put_vol,
            "iv_call": iv_call, "iv_put": iv_put,
            "T": float(ctx["T"]),
        })

    try:
        from lib.power_zone_er import compute_power_zone  # type: ignore
    except Exception:
        # локальный импорт если без пакета
        from power_zone_er import compute_power_zone  # type: ignore

    pz = compute_power_zone(
        S=S_multi if np.isfinite(S_multi) else float("nan"),
        strikes_eval=K_window.tolist(),
        all_series_ctx=all_ctx,
        day_high=getattr(cfg, "day_high", None),
        day_low=getattr(cfg, "day_low", None),
    )
    base["PZ"] = pd.Series(pz, index=base.index).astype(float)

    # 9) Порядок колонок
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
