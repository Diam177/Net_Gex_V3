
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Tuple

import streamlit as st

# --- Helpers to hide tables from main page ---
def _st_hide_df(*args, **kwargs):
    # no-op: we suppress table rendering on main page per requirements
    return None
def _st_hide_subheader(*args, **kwargs):
    # no-op: suppress section headers for tables
    return None
from lib.netgex_chart import render_netgex_bars

# Project imports
from lib.sanitize_window import sanitize_and_window_pipeline
from lib.tiker_data import (
    list_future_expirations,
    download_snapshot_json,
    get_spot_price,
    PolygonError,
)

st.set_page_config(page_title="GammaStrat — df_raw", layout="wide")

# --- Helpers -----------------------------------------------------------------
def _coerce_results(data: Any) -> List[Dict]:
    """
    Приводит разные варианты JSON к list[dict] записей опционов.
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("results", "options", "data"):
            v = data.get(key)
            if isinstance(v, list):
                return v
    return []


def _infer_spot_from_snapshot(raw: List[Dict]) -> float | None:
    """
    Fallback-оценка S из снимка опционов:
    - Берём страйки и |delta| близкие к 0.5 (окно 0.2..0.8, вес 1/| |delta|-0.5 |)
    """
    num = 0.0
    den = 0.0

    def get_nested(d: Dict, keys: list[str], blocks=("details","greeks","day","underlying_asset")):
        for k in keys:
            if k in d:
                return d[k]
        for b in blocks:
            sub = d.get(b, {})
            if isinstance(sub, dict):
                for k in keys:
                    if k in sub:
                        return sub[k]
        return None

    for r in raw:
        K = get_nested(r, ["strike","k","strike_price","strikePrice"])
        dlt = get_nested(r, ["delta","dlt"])
        if K is None or dlt is None:
            continue
        try:
            K = float(K); dlt = float(dlt)
        except Exception:
            continue
        if 0.2 <= abs(dlt) <= 0.8 and K > 0:
            w = 1.0 / (abs(abs(dlt) - 0.5) + 1e-6)
            num += w * K
            den += w
    if den > 0:
        return num / den
    return None


def _get_api_key() -> str | None:
    key = None
    try:
        key = st.secrets.get("POLYGON_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        key = None
    if not key:
        key = os.getenv("POLYGON_API_KEY")
    return key


# --- UI: Controls -------------------------------------------------------------
st.sidebar.markdown("### Действия")
st.markdown("## Главная · Выбор тикера/экспирации и просмотр df_raw")

api_key = _get_api_key()
if not api_key:
    st.error("POLYGON_API_KEY не задан в Streamlit Secrets или переменных окружения.")
    st.stop()

col1, col2, col3 = st.columns([1.2, 1, 1])

with col1:
    ticker = st.text_input("Тикер", value=st.session_state.get("ticker", "SPY")).strip().upper()
    st.session_state["ticker"] = ticker

with col2:
    # Получаем список будущих экспираций под выбранный тикер
    expirations: list[str] = st.session_state.get(f"expirations:{ticker}", [])
    if not expirations and ticker:
        try:
            expirations = list_future_expirations(ticker, api_key)
            st.session_state[f"expirations:{ticker}"] = expirations
        except Exception as e:
            st.error(f"Не удалось получить даты экспираций: {e}")
            expirations = []

    if expirations:
        # по умолчанию ближайшая дата — первая в списке
        default_idx = 0
        sel = st.selectbox("Дата экспирации", options=expirations, index=default_idx, key=f"exp_sel:{ticker}")
        expiration = sel

        # --- Режим агрегации экспираций ---
        mode_exp = st.radio("Режим экспираций", ["Single","Multi"], index=0, horizontal=True)
        selected_exps = []
        weight_mode = "равные"
        if mode_exp == "Single":
            selected_exps = [expiration]
        else:
            selected_exps = st.multiselect("Выберите экспирации", options=expirations, default=expirations[:2])
            weight_mode = st.selectbox("Взвешивание", ["равные","1/T","1/√T"], index=2)

    else:
        expiration = ""
        st.warning("Нет доступных дат экспираций для тикера.")

st.divider()

# --- Data fetch ---------------------------------------------------------------
raw_records: List[Dict] | None = None
snapshot_js: Dict | None = None

if ticker and expiration:
    try:
        snapshot_js = download_snapshot_json(ticker, expiration, api_key)
        raw_records = _coerce_results(snapshot_js)
        st.session_state["raw_records"] = raw_records
    except PolygonError as e:
        st.error(f"Ошибка Polygon: {e}")
    except Exception as e:
        st.error(f"Ошибка при загрузке snapshot JSON: {e}")

# --- Sidebar: download raw provider JSON -------------------------------------
if snapshot_js:
    try:
        raw_bytes = json.dumps(snapshot_js, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        fname = f"{ticker}_{expiration}.json"
        st.sidebar.download_button(
            label="Скачать сырой JSON (Polygon)",
            data=raw_bytes,
            file_name=fname,
            mime="application/json",
            use_container_width=True,
        )
    except Exception as e:
        st.sidebar.error(f"Не удалось подготовить JSON к скачиванию: {e}")
else:
    st.sidebar.info("Выберите тикер и дату экспирации, чтобы загрузить snapshot и скачать JSON.")

# --- Spot price ---------------------------------------------------------------
S: float | None = None
if ticker:
    try:
        S, ts_ms, src = get_spot_price(ticker, api_key)
        st.caption(f"Spot {S} (источник: {src})")
    except Exception:
        S = None
# 3) из snapshot (fallback)
if S is None and raw_records:
    S = _infer_spot_from_snapshot(raw_records)
    if S:
        st.caption(f"Spot ~{S:.2f} (оценка по ATM-страйкам из snapshot)")

# --- Run sanitize/window + show df_raw ---------------------------------------
if raw_records:
    try:
        res = sanitize_and_window_pipeline(raw_records, S=S)

        # --- TRUE MULTI-EXP PROCESSING: прогон по каждой выбранной экспирации и объединение ---
        try:
            if ("mode_exp" in locals()) and (mode_exp == "Multi") and selected_exps:
                import pandas as pd
                df_corr_multi_list = []
                windows_multi = {}

                for _exp in selected_exps:
                    try:
                        js_e = download_snapshot_json(ticker, _exp, api_key)
                        recs_e = _coerce_results(js_e)
                        res_e = sanitize_and_window_pipeline(recs_e, S=S)
                        dfe = res_e.get("df_corr")
                        if dfe is not None and not getattr(dfe, "empty", True):
                            df_corr_multi_list.append(dfe)
                        win_e = res_e.get("windows") or {}
                        for k, v in win_e.items():
                            windows_multi[k] = v
                    except Exception as _exc:
                        # мягко пропускаем проблемные экспирации, чтобы не ломать UI
                        pass

                if df_corr_multi_list:
                    df_corr = pd.concat(df_corr_multi_list, ignore_index=True)
                if windows_multi:
                    windows = windows_multi
                    # --- Соберём промежуточные таблицы по каждой экспирации для ZIP-скачивания ---
                    try:
                        import io, zipfile
                        import pandas as pd

                        multi_exports = {}  # exp -> {name: DataFrame}
                        for _exp in selected_exps:
                            try:
                                js_e = download_snapshot_json(ticker, _exp, api_key)
                                recs_e = _coerce_results(js_e)
                                res_e = sanitize_and_window_pipeline(recs_e, S=S)

                                df_raw_e    = res_e.get("df_raw")
                                df_marked_e = res_e.get("df_marked")
                                df_corr_e   = res_e.get("df_corr")
                                df_weights_e= res_e.get("df_weights")
                                windows_e   = res_e.get("windows") or {}
                                window_raw_e= res_e.get("window_raw")
                                window_corr_e=res_e.get("window_corr")

                                # windows (табличный вид) для этой экспирации
                                df_windows_e = None
                                try:
                                    if windows_e and isinstance(windows_e, dict) and df_weights_e is not None:
                                        rows_w = []
                                        idxs = windows_e.get(_exp, [])
                                        if hasattr(df_weights_e, "empty") and not df_weights_e.empty:
                                            gW = df_weights_e[df_weights_e["exp"] == _exp].sort_values("K").reset_index(drop=True)
                                            for i in list(idxs):
                                                ii = int(i)
                                                if 0 <= ii < len(gW):
                                                    rows_w.append({
                                                        "exp": _exp,
                                                        "row_index": ii,
                                                        "K": float(gW.loc[ii, "K"]),
                                                        "w_blend": float(gW.loc[ii, "w_blend"]),
                                                    })
                                        if rows_w:
                                            df_windows_e = pd.DataFrame(rows_w, columns=["exp","row_index","K","w_blend"]).sort_values(["exp","K"])
                                except Exception:
                                    df_windows_e = None

                                multi_exports[_exp] = {
                                    "df_raw": df_raw_e,
                                    "df_marked": df_marked_e,
                                    "df_corr": df_corr_e,
                                    "df_weights": df_weights_e,
                                    "windows_table": df_windows_e,
                                    "window_raw": window_raw_e,
                                    "window_corr": window_corr_e,
                                }
                            except Exception:
                                # пропускаем проблемные даты, ZIP соберётся из удачных
                                pass

                        # Кнопка скачивания ZIP (только если есть хоть что-то)
                        
                        def _zip_multi_intermediate(exports: dict, final_sum_df=None) -> io.BytesIO:
                            bio = io.BytesIO()
                            with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                                # write per-exp intermediates
                                for exp_key, tbls in (exports or {}).items():
                                    for name, df in (tbls or {}).items():
                                        if df is None:
                                            continue
                                        try:
                                            if hasattr(df, "empty") and df.empty:
                                                continue
                                        except Exception:
                                            pass
                                        try:
                                            csv_bytes = df.to_csv(index=False).encode("utf-8")
                                        except Exception:
                                            csv_bytes = str(df).encode("utf-8")
                                        zf.writestr(f"{exp_key}/{name}.csv", csv_bytes)
                                # write per-exp finals
                                try:
                                    from lib.final_table import build_final_tables_from_corr, FinalTableConfig
                                    finals = build_final_tables_from_corr(df_corr, windows, cfg=FinalTableConfig())
                                    for exp_key, fin in (finals or {}).items():
                                        if fin is None or getattr(fin, "empty", True):
                                            continue
                                        zf.writestr(f"{exp_key}/final_table.csv", fin.to_csv(index=False).encode("utf-8"))
                                except Exception:
                                    pass
                                # aggregated final table for Multi (used by chart)
                                try:
                                    if 'df_final_multi' in locals() and df_final_multi is not None and not getattr(df_final_multi, 'empty', True):
                                        zf.writestr("FINAL_SUM.csv", df_final_multi.to_csv(index=False).encode("utf-8"))
                                except Exception:
                                    pass
                                # aggregated final table (sum) used by chart
                                try:
                                    if final_sum_df is not None and not getattr(final_sum_df, 'empty', True):
                                        zf.writestr("FINAL_SUM.csv", final_sum_df.to_csv(index=False).encode("utf-8"))
                                except Exception:
                                    pass


                            bio.seek(0)
                            return bio
                        if any(tbl is not None and (not getattr(tbl, "empty", True)) 
                               for tbls in multi_exports.values() for tbl in (tbls or {}).values()):
                            zip_bytes = _zip_multi_intermediate(multi_exports, df_final_multi if 'df_final_multi' in locals() else None)
                            fname = f"{ticker}_intermediate_{len(multi_exports)}exps.zip" if ticker else "intermediate_tables.zip"
                            st.download_button(
                                "Скачать таблицы",
                                data=zip_bytes.getvalue(),
                                file_name=fname,
                                mime="application/zip",
                                type="primary",
                                use_container_width=False,
                            )
                    except Exception as _zip_err:
                        st.warning("Не удалось подготовить ZIP с промежуточными таблицами.")
                        st.exception(_zip_err)

        except Exception as _e_multi:
            st.warning("Multi-exp: не удалось объединить серии.")
            st.exception(_e_multi)
        df_raw = res.get("df_raw")
        if df_raw is None or getattr(df_raw, "empty", True):
            st.warning("df_raw пуст. Проверьте формат данных.")
        else:
            _st_hide_df(df_raw, use_container_width=True)
            # --- Ниже показываем остальные массивы по степени создания ---
            # 1) df_marked (df_raw + флаги аномалий)
            df_marked = res.get("df_marked")
            if df_marked is not None and not getattr(df_marked, "empty", True):
                _st_hide_subheader()
                _st_hide_df(df_marked, use_container_width=True, hide_index=True)

            # 2) df_corr (восстановленные IV/Greeks)
            df_corr = res.get("df_corr")
            if df_corr is not None and not getattr(df_corr, "empty", True):
                _st_hide_subheader()
                _st_hide_df(df_corr, use_container_width=True, hide_index=True)

            # 3) df_weights (веса окна по страйку)
            df_weights = res.get("df_weights")
            if df_weights is not None and not getattr(df_weights, "empty", True):
                _st_hide_subheader()
                _st_hide_df(df_weights, use_container_width=True, hide_index=True)

            # 4) windows (выбранные страйки каждого окна) — преобразуем в таблицу
            windows = res.get("windows")
            if windows and isinstance(windows, dict) and df_weights is not None:
                try:
                    rows = []
                    for exp, idx in windows.items():
                        # df_weights на эту экспирацию, отсортируем по K как в select_windows
                        g = df_weights[df_weights["exp"] == exp].sort_values("K").reset_index(drop=True)
                        for i in list(idx):
                            if 0 <= int(i) < len(g):
                                rows.append({"exp": exp, "row_index": int(i), "K": float(g.loc[int(i), "K"]),
                                             "w_blend": float(g.loc[int(i), "w_blend"])})
                    import pandas as pd  # локальный импорт безопасен
                    df_windows = pd.DataFrame(rows, columns=["exp","row_index","K","w_blend"]).sort_values(["exp","K"])
                    _st_hide_subheader()
                    # Приведём тип w_blend к float64 и зададим формат — уберёт предупреждение 2^53 в браузере
                    try:
                        df_windows = df_windows.astype({'w_blend': 'float64'})
                        _st_hide_df(
                            df_windows,
                            use_container_width=True,
                            hide_index=True,
                            column_config={'w_blend': st.column_config.NumberColumn('w_blend', format='%.6f')},
                        )
                    except Exception:
                        _st_hide_df(df_windows, use_container_width=True, hide_index=True)
                
                except Exception as _e:
                    st.warning("Не удалось отобразить windows в табличном виде.")
                    st.exception(_e)

            # 5) window_raw (строки окна из исходных + флаги)
            window_raw = res.get("window_raw")
            if window_raw is not None and not getattr(window_raw, "empty", True):
                _st_hide_subheader()
                _st_hide_df(window_raw, use_container_width=True, hide_index=True)

            # 6) window_corr (строки окна из исправленных данных)
            window_corr = res.get("window_corr")
            if window_corr is not None and not getattr(window_corr, "empty", True):
                _st_hide_subheader()
                _st_hide_df(window_corr, use_container_width=True, hide_index=True)

            if 'mode_exp' in locals() and mode_exp != 'Multi':
                # Кнопка скачивания ZIP со всеми таблицами (single)
                try:
                    import io, zipfile
                    from lib.final_table import build_final_tables_from_corr, FinalTableConfig
                    def _zip_single_tables(res_dict, df_corr_single, windows_single, exp_str):
                        bio = io.BytesIO()
                        with zipfile.ZipFile(bio, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
                            for name in ['df_raw','df_marked','df_corr','df_weights','window_raw','window_corr']:
                                df = res_dict.get(name)
                                if df is None:
                                    continue
                                try:
                                    if hasattr(df, 'empty') and df.empty:
                                        continue
                                except Exception:
                                    pass
                                try:
                                    csv_bytes = df.to_csv(index=False).encode('utf-8')
                                except Exception:
                                    csv_bytes = str(df).encode('utf-8')
                                zf.writestr(f"{exp_str}/{name}.csv", csv_bytes)
                            # финальная таблица
                            try:
                                finals = build_final_tables_from_corr(df_corr_single, windows_single, cfg=FinalTableConfig())
                                if exp_str in finals and finals[exp_str] is not None and not getattr(finals[exp_str], 'empty', True):
                                    zf.writestr(f"{exp_str}/final_table.csv", finals[exp_str].to_csv(index=False).encode('utf-8'))
                            except Exception:
                                pass
                        bio.seek(0)
                        return bio
                    exp_str = expiration if 'expiration' in locals() else 'exp'
                    zip_bytes = _zip_single_tables(res, df_corr, windows, exp_str)
                    st.download_button('Скачать таблицы', data=zip_bytes.getvalue(), file_name=(f"{ticker}_{exp_str}_tables.zip" if ticker else 'tables.zip'), mime='application/zip', type='primary')
                except Exception as _e_zip_single:
                    st.warning('Не удалось подготовить ZIP с таблицами (single).')
                    st.exception(_e_zip_single)
            # 7) Финальная таблица (по df_corr + windows)
            try:
                from lib.final_table import build_final_tables_from_corr, FinalTableConfig, _series_ctx_from_corr
                from lib.netgex_ag import compute_netgex_ag_per_expiry, NetGEXAGConfig
                from lib.power_zone_er import compute_power_zone_and_er
                import numpy as np
                import pandas as pd
                import math

                final_cfg = FinalTableConfig()
                scale_val = getattr(final_cfg, "scale_millions", 1_000_000)

                if df_corr is not None and windows:

                    # --- MULTI режим: взвешенная сумма по нескольким экспирациям ---
                    if ("mode_exp" in locals()) and (mode_exp == "Multi") and selected_exps:
                        # 1) веса по T
                        exp_list = [e for e in selected_exps if e in df_corr["exp"].unique().tolist()]
                        if not exp_list:
                            raise ValueError("Нет выбранных экспираций для режима Multi.")
                        t_map = {}
                        for e in exp_list:
                            T_vals = df_corr.loc[df_corr["exp"]==e, "T"].dropna().values
                            T_med = float(np.nanmedian(T_vals)) if T_vals.size else float("nan")
                            if not (math.isfinite(T_med) and T_med>0):
                                T_med = 1.0/252.0  # защита от T→0/NaN
                            t_map[e] = T_med
                        w_raw = {}
                        for e in exp_list:
                            if weight_mode == "1/T":
                                w_raw[e] = 1.0 / max(t_map[e], 1.0/252.0)
                            elif weight_mode == "1/√T":
                                w_raw[e] = 1.0 / math.sqrt(max(t_map[e], 1.0/252.0))
                            else:
                                w_raw[e] = 1.0
                        w_sum = sum(w_raw.values()) or 1.0
                        weights = {e: w_raw[e]/w_sum for e in exp_list}

                        # 2) NetGEX/AG профили по каждой экспирации
                        per_exp = {}
                        for e in exp_list:
                            nt = compute_netgex_ag_per_expiry(df_corr, e, windows=windows, cfg=NetGEXAGConfig(scale=scale_val, aggregate="none"))
                            per_exp[e] = nt[["K","AG_1pct","NetGEX_1pct"] + ([c for c in ["S","F"] if c in nt.columns])].copy()

                        # 3) единая сетка K (union)
                        K_union = sorted(set().union(*[set(df["K"].astype(float).tolist()) for df in per_exp.values()]))
                        base = pd.DataFrame({"K": K_union})
                        # медианные S/F по сериям
                        S_med = float(np.nanmedian([float(np.nanmedian(nt["S"])) for nt in per_exp.values() if "S" in nt.columns])) if per_exp else float("nan")
                        base["S"] = S_med
                        if any("F" in nt.columns for nt in per_exp.values()):
                            F_vals = []
                            for nt in per_exp.values():
                                if "F" in nt.columns and nt["F"].notna().any():
                                    F_vals.append(float(np.nanmedian(nt["F"])))
                            base["F"] = float(np.nanmedian(F_vals)) if F_vals else np.nan

                        base["AG_1pct"] = 0.0
                        base["NetGEX_1pct"] = 0.0

                        for e, nt in per_exp.items():
                            m = nt.groupby("K")[["AG_1pct","NetGEX_1pct"]].sum()
                            base = base.merge(m, left_on="K", right_index=True, how="left", suffixes=("","_add")).fillna(0.0)
                            base["AG_1pct"] += weights[e] * base.pop("AG_1pct_add")
                            base["NetGEX_1pct"] += weights[e] * base.pop("NetGEX_1pct_add")

                        # 4) call_oi / put_oi (простая сумма по сериям)
                        g = df_corr[df_corr["exp"].isin(exp_list)].copy()
                        agg_oi = g.groupby(["K","side"], as_index=False)["oi"].sum()
                        piv_oi = agg_oi.pivot_table(index="K", columns="side", values="oi", aggfunc="sum").fillna(0.0)
                        base["call_oi"] = piv_oi.get("C", pd.Series(dtype=float)).reindex(base["K"], fill_value=0.0).to_numpy()
                        base["put_oi"]  = piv_oi.get("P", pd.Series(dtype=float)).reindex(base["K"], fill_value=0.0).to_numpy()

                        # 5) масштаб в млн $
                        if scale_val and scale_val>0:
                            base["AG_1pct_M"] = base["AG_1pct"] / scale_val
                            base["NetGEX_1pct_M"] = base["NetGEX_1pct"] / scale_val

                        # 6) PZ/ER по списку контекстов с весами
                        all_ctx = []
                        for e in exp_list:
                            ctx_map = _series_ctx_from_corr(df_corr, e)
                            if e in ctx_map:
                                ctx = dict(ctx_map[e])  # копия
                                # домножим "формообразующие" ряды
                                if "gamma_abs_share" in ctx and "gamma_net_share" in ctx:
                                    ctx["gamma_abs_share"] = (np.array(ctx["gamma_abs_share"], dtype=float) * weights[e])
                                    ctx["gamma_net_share"] = (np.array(ctx["gamma_net_share"], dtype=float) * weights[e])
                                all_ctx.append(ctx)

                        strikes_eval = base["K"].astype(float).tolist()
                        pz, er_up, er_down = compute_power_zone_and_er(
                            S=S_med,
                            strikes_eval=strikes_eval,
                            all_series_ctx=all_ctx,
                            day_high=getattr(final_cfg, "day_high", None),
                            day_low=getattr(final_cfg, "day_low", None),
                        )
                        # маппинг на таблицу
                        base["PZ"] = pd.Series(pz, index=base.index).astype(float)
                        base["ER_Up"] = pd.Series(er_up, index=base.index).astype(float)
                        base["ER_Down"] = pd.Series(er_down, index=base.index).astype(float)

                        # порядок колонок
                        cols = ["K","S"] + (["F"] if "F" in base.columns else []) + ["call_oi","put_oi","AG_1pct","NetGEX_1pct"]
                        if "AG_1pct_M" in base.columns: cols += ["AG_1pct_M","NetGEX_1pct_M"]
                        cols += ["PZ","ER_Up","ER_Down"]
                        df_final_multi = base[cols].sort_values("K").reset_index(drop=True)

                        _st_hide_subheader()
                        _st_hide_df(df_final_multi, use_container_width=True, hide_index=True)
                        # --- Net GEX chart (Multi: aggregated) ---
                        try:
                            render_netgex_bars(df_final=df_final_multi, ticker=ticker, spot=S if 'S' in locals() else None, toggle_key='netgex_multi')
                        except Exception as _chart_em:
                            st.error('Не удалось отобразить чарт Net GEX (Multi)')
                            st.exception(_chart_em)

                    else:
                        # --- SINGLE режим: как было ---
                        final_tables = build_final_tables_from_corr(df_corr, windows, cfg=final_cfg)
                        exps = list(final_tables.keys())
                        if exps:
                            exp_to_show = expiration if 'expiration' in locals() and expiration in final_tables else exps[0]
                            df_final = final_tables.get(exp_to_show)
                            if df_final is not None and not getattr(df_final, "empty", True):
                                _st_hide_subheader()
                                _st_hide_df(df_final, use_container_width=True, hide_index=True)
                                # --- Net GEX chart (under the final table) ---
                                try:
                                    render_netgex_bars(df_final=df_final, ticker=ticker, spot=S if 'S' in locals() else None, toggle_key='netgex_main')
                                except Exception as _chart_e:
                                    st.error('Не удалось отобразить чарт Net GEX')
                                    st.exception(_chart_e)
                            else:
                                st.info("Финальная таблица пуста для выбранной экспирации.")
            except Exception as _e:
                st.error("Ошибка построения финальной таблицы.")
                st.exception(_e)
    except Exception as e:
            st.error("Ошибка пайплайна sanitize/window.")
            st.exception(e)
