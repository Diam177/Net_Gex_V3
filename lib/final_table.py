
# -*- coding: utf-8 -*-
"""
final_table.py — ЕДИНЫЙ источник данных для UI и графиков.
Рефакторинг: вся математика берётся из sanitize_window.py, netgex_ag.py, power_zone_er.py.
compute.py удалён. Для обратной совместимости здесь реализованы оболочки с прежними именами.
Никаких прямых вызовов низкоуровневых модулей из UI-кода — только через этот фасад.

Публичный API:
- build_final_table(...): основной конвейер → pandas.DataFrame
- get_key_levels_for_chart(final_df): подготовка уровней для чарта
- extract_core_from_chain(...), compute_series_metrics_for_expiry(...), aggregate_series(...): совместимость с прежними импортами

Все вычисления делегируются в sanitize_window / netgex_ag / power_zone_er.
"""

from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Any, Optional, Tuple, Iterable
import numpy as np
import pandas as pd

# Внутренние импорты низкого уровня — только здесь
from . import sanitize_window as _sw
from . import netgex_ag as _nga
from . import power_zone_er as _pzer


# -----------------------------
# Вспомогательные универсальные вызовы
# -----------------------------

def _call_if_exists(mod, *names: str):
    """Возвращает первую найденную в модуле функцию из списка имён, иначе None."""
    for n in names:
        fn = getattr(mod, n, None)
        if callable(fn):
            return fn
    return None


def _safe_numeric(s):
    return pd.to_numeric(s, errors='coerce')


def _ensure_chain_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Минимальная нормализация колонок чейна, НЕ выполняет математику греков.
    Требуемые поля: strike, type (call/put), oi, volume, gamma (если есть), mid/price (если есть)
    Допускается название колонок в разных регистрах.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=['strike','type','oi','volume','gamma'])
    # нормализуем имена
    cols = {c.lower(): c for c in df.columns}
    def pick(*opts):
        for o in opts:
            if o in cols: return cols[o]
        return None
    m = {}
    m['strike'] = pick('strike','k','strk')
    m['type']   = pick('type','option_type','side')
    m['oi']     = pick('oi','open_interest')
    m['volume'] = pick('volume','vol')
    m['gamma']  = pick('gamma','opt_gamma','g')
    # Построим базовый фрейм
    out = pd.DataFrame()
    if m['strike'] is not None: out['strike'] = _safe_numeric(df[m['strike']])
    else: out['strike'] = np.nan
    if m['type'] is not None:
        t = df[m['type']].astype(str).str.lower().str.replace('c','call').str.replace('p','put')
        t = t.replace({'calll':'call','puts':'put'})
        out['type'] = np.where(t.str.startswith('c'), 'call',
                        np.where(t.str.startswith('p'),'put', t))
    else:
        out['type'] = 'call'
    if m['oi'] is not None: out['oi'] = _safe_numeric(df[m['oi']])
    else: out['oi'] = 0.0
    if m['volume'] is not None: out['volume'] = _safe_numeric(df[m['volume']])
    else: out['volume'] = 0.0
    if m['gamma'] is not None: out['gamma'] = _safe_numeric(df[m['gamma']])
    else:
        # gamma может быть рассчитана в sanitize_window при необходимости
        out['gamma'] = np.nan
    # очистка
    out = out.dropna(subset=['strike']).copy()
    out['strike'] = out['strike'].round(2)
    return out


# -----------------------------
# Публичный основной конвейер
# -----------------------------

@lru_cache(maxsize=64)
def build_final_table(
    ticker: str,
    expiry: str,
    chain_df_key: Optional[str] = None,
    spot: Optional[float] = None,
    extra: Optional[Tuple] = None
) -> pd.DataFrame:
    """
    Возвращает итоговую таблицу метрик по ТОЛЬКО одной экспирации.
    Все вычисления делегированы в sanitize_window / netgex_ag / power_zone_er.

    Параметры:
        ticker: тикер
        expiry: экспирация в формате YYYY-MM-DD
        chain_df_key: опциональный ключ для кеша данных чейна (если внешний код кэширует DataFrame)
        spot: спот-цена для ATM/EM
        extra: опциональный кортеж доп. параметров для кэша (например, источник данных)

    Ожидаемые столбцы результата (примерный набор):
        ['strike','net_gex','ag','pz','er_up','er_down','call_oi','put_oi','call_vol','put_vol',
         'gflip','pos_netgex_1','pos_netgex_2','pos_netgex_3',
         'neg_netgex_1','neg_netgex_2','neg_netgex_3',
         'ag_1','ag_2','ag_3','atm_iv','em_d','em_w', ...]
    """
    # 1) Получение и нормализация чейна (внешний код обязан передать или положить в глобальный кеш).
    df_chain = _fetch_chain_df_from_global_cache(chain_df_key)
    df_chain = _ensure_chain_schema(df_chain)

    # 2) Подготовка окон/IV/греков (без собственной математики — делегируем sanitize_window)
    fn_prep = _call_if_exists(_sw,
        'prepare_chain', 'rebuild_iv_and_greeks', 'sanitize_chain', 'prepare_for_metrics'
    )
    if fn_prep is not None:
        df_prepared = fn_prep(df_chain, spot=spot, expiry=expiry, ticker=ticker)
    else:
        # нет явной функции — используем как есть
        df_prepared = df_chain.copy()

    # 3) Профили Net GEX и Absolute Gamma по страйкам (делегируем netgex_ag)
    # Ищем наиболее вероятные названия функций
    fn_ng = _call_if_exists(_nga,
        'compute_netgex_profile', 'compute_net_gex_profile', 'compute_net_gex', 'netgex_profile'
    )
    fn_ag = _call_if_exists(_nga,
        'compute_abs_gamma_profile','compute_abs_gamma','abs_gamma_profile','compute_ag_profile'
    )
    if fn_ng is None or fn_ag is None:
        raise RuntimeError('netgex_ag: не найдены функции расчёта NetGEX/AG')

    df_ng = fn_ng(df_prepared, expiry=expiry, spot=spot)
    df_ag = fn_ag(df_prepared, expiry=expiry, spot=spot)

    # 4) Power Zone / Easy Reach (делегируем power_zone_er)
    fn_pz = _call_if_exists(_pzer,
        'compute_power_zone','compute_pz','power_zone_profile'
    )
    fn_er = _call_if_exists(_pzer,
        'compute_easy_reach','compute_er','easy_reach_profile'
    )
    if fn_pz is None or fn_er is None:
        raise RuntimeError('power_zone_er: не найдены функции расчёта PZ/ER')

    df_pz = fn_pz(df_prepared, expiry=expiry, spot=spot)
    df_er = fn_er(df_prepared, expiry=expiry, spot=spot)

    # 5) Сведение в единую таблицу
    # Требуемая колонка в каждом — 'strike'.
    _cols = ['strike']
    for d in (df_ng, df_ag, df_pz, df_er):
        if 'strike' not in d.columns:
            raise ValueError('Ожидается колонка strike во всех промежуточных таблицах')

    out = (
        df_ng.merge(df_ag, on='strike', how='outer', suffixes=('','_ag'))
             .merge(df_pz, on='strike', how='outer', suffixes=('','_pz'))
             .merge(df_er, on='strike', how='outer', suffixes=('','_er'))
    )

    # 6) Добавляем базовые колонки OI/Volume (если присутствуют)
    if 'call_oi' not in out.columns or 'put_oi' not in out.columns:
        # Попробуем собрать из подготовленного чейна
        if set(['strike','type','oi']).issubset(df_prepared.columns):
            pivot_oi = (df_prepared[['strike','type','oi']]
                        .pivot_table(index='strike', columns='type', values='oi', aggfunc='sum')
                        .rename(columns={'call':'call_oi','put':'put_oi'}))
            pivot_oi.columns.name = None
            out = out.merge(pivot_oi.reset_index(), on='strike', how='left')
    if 'call_vol' not in out.columns or 'put_vol' not in out.columns:
        if set(['strike','type','volume']).issubset(df_prepared.columns):
            pivot_vol = (df_prepared[['strike','type','volume']]
                        .pivot_table(index='strike', columns='type', values='volume', aggfunc='sum')
                        .rename(columns={'call':'call_vol','put':'put_vol'}))
            pivot_vol.columns.name = None
            out = out.merge(pivot_vol.reset_index(), on='strike', how='left')

    # 7) Производные уровни (g-flip, топ-3 и т.п.) — МЕТКИ, не математика NetGEX/AG/PZ/ER
    out = _attach_key_levels(out)

    # 8) Доп. атрибуты ATM IV / Expected Move (делегируем sanitize_window, если есть)
    fn_em = _call_if_exists(_sw, 'compute_expected_move','expected_move_from_iv','calc_em')
    iv_val = None; em_d = None; em_w = None
    if fn_em is not None:
        try:
            em = fn_em(spot=spot, ticker=ticker, expiry=expiry, chain=df_prepared)
            if isinstance(em, dict):
                iv_val = em.get('atm_iv')
                em_d = em.get('em_d') or em.get('daily')
                em_w = em.get('em_w') or em.get('weekly')
        except Exception:
            pass

    # Располагаем итоговые атрибуты в столбцах (одни и те же значения по всей таблице, для удобства UI)
    if iv_val is not None:
        out['atm_iv'] = iv_val
    if em_d is not None:
        out['em_d'] = em_d
    if em_w is not None:
        out['em_w'] = em_w

    # Упорядочим колонки: strike первым
    first_cols = ['strike']
    ordered = first_cols + [c for c in out.columns if c not in first_cols]
    return out[ordered].sort_values('strike').reset_index(drop=True)


# --- совместимость с прежним API compute.py ---

def extract_core_from_chain(chain_df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """Совместимость: теперь просто нормализация + делегирование в sanitize_window при наличии."""
    df = _ensure_chain_schema(chain_df)
    fn = _call_if_exists(_sw, 'extract_core_from_chain','prepare_core','sanitize_chain')
    if fn is not None:
        try:
            df = fn(df, *args, **kwargs)
        except Exception:
            pass
    return df


def compute_series_metrics_for_expiry(*, ticker: str, expiry: str, chain_df_key: Optional[str]=None,
                                      spot: Optional[float]=None, **kwargs) -> pd.DataFrame:
    """Совместимость: возвращаем уже ГОТОВУЮ итоговую таблицу для данной экспирации."""
    return build_final_table(ticker=ticker, expiry=expiry, chain_df_key=chain_df_key, spot=spot)


def aggregate_series(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """Совместимость: агрегирование уже производится внутри build_final_table; просто возвращаем df."""
    return df


# -----------------------------
# Подготовка данных для чарта
# -----------------------------

def get_key_levels_for_chart(final_df: pd.DataFrame) -> Dict[str, Any]:
    """Возвращает словарь с ключевыми уровнями/набором серий для Key Levels chart.
    Всё берём ТОЛЬКО из final_df.
    """
    if final_df is None or final_df.empty:
        return {'levels': {}, 'series': {}}

    levels = {}
    # G-Flip (по нашему DataFrame): берём strike с минимальной |net_gex|, либо явная колонка gflip
    if 'gflip' in final_df.columns and final_df['gflip'].notna().any():
        gflip_val = final_df.loc[final_df['gflip'].notna(), 'gflip'].iloc[0]
        levels['gflip'] = float(np.round(gflip_val))
    elif 'net_gex' in final_df.columns:
        idx = final_df['net_gex'].abs().idxmin()
        if pd.notna(idx):
            levels['gflip'] = float(np.round(final_df.loc[idx, 'strike']))

    # Топ-3 положительных/отрицательных NetGEX
    if 'net_gex' in final_df.columns:
        pos = final_df.nlargest(3, 'net_gex')[['strike','net_gex']]
        neg = final_df.nsmallest(3, 'net_gex')[['strike','net_gex']]
        for i, row in enumerate(pos.itertuples(index=False), start=1):
            levels[f'pos_netgex_{i}'] = float(row.strike)
        for i, row in enumerate(neg.itertuples(index=False), start=1):
            levels[f'neg_netgex_{i}'] = float(row.strike)

    # Топ-3 AG
    if 'ag' in final_df.columns:
        top_ag = final_df.nlargest(3, 'ag')[['strike','ag']]
        for i, row in enumerate(top_ag.itertuples(index=False), start=1):
            levels[f'ag_{i}'] = float(row.strike)

    # PZ / ER
    for col in ['pz','er_up','er_down']:
        if col in final_df.columns and final_df[col].notna().any():
            # для чарта удобно иметь серию {strike: value}
            series = dict(zip(final_df['strike'].astype(float), final_df[col].astype(float)))
            levels[col] = series

    # Доп. сведения
    meta = {}
    for c in ['atm_iv','em_d','em_w']:
        if c in final_df.columns and final_df[c].notna().any():
            meta[c] = float(final_df[c].dropna().iloc[0])
    return {'levels': levels, 'series': {}, 'meta': meta}


# -----------------------------
# Локальные хранилища/кеши
# -----------------------------

_GLOBAL_CHAIN_CACHE: Dict[str, pd.DataFrame] = {}

def put_chain_df_to_global_cache(key: str, df: pd.DataFrame) -> None:
    _GLOBAL_CHAIN_CACHE[key] = df.copy() if df is not None else pd.DataFrame()

def _fetch_chain_df_from_global_cache(key: Optional[str]) -> pd.DataFrame:
    if key and key in _GLOBAL_CHAIN_CACHE:
        return _GLOBAL_CHAIN_CACHE[key].copy()
    # по умолчанию пусто
    return pd.DataFrame()


# -----------------------------
# Вспомогательные маркеры уровней
# -----------------------------

def _attach_key_levels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # G-Flip как strike минимального |net_gex|
    if 'net_gex' in out.columns and 'gflip' not in out.columns and not out['net_gex'].isna().all():
        try:
            k = out.loc[out['net_gex'].abs().idxmin(), 'strike']
            out['gflip'] = float(k)
        except Exception:
            out['gflip'] = np.nan
    return out
