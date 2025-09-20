# -*- coding: utf-8 -*-
"""
sanitize_window.py — Робастная подготовка опционных данных:
1) Принимаем сырые записи от провайдера (Polygon/иные).
2) Строим "сырую" таблицу.
3) Помечаем аномалии (IV/греков/дельт/данных).
4) Восстанавливаем IV-кривую по каждой экспирации и пересчитываем грейки (BS).
5) Формируем "исправленные" данные.
6) Выбираем окно страйков (по бленд-весу: |C-P|, ликвидность, долларовая гамма).
7) Возвращаем "оконные" таблицы (raw/corrected) и служебные артефакты.

Без внешних зависимостей, кроме numpy/pandas.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import pandas as pd

# NEW: Для SABR calibration
from scipy.optimize import least_squares

def _parse_expiration_to_ts(exp_raw):
    """Parse expiration to unix timestamp (seconds, UTC), DST-aware.
    - If number-like UNIX ts (seconds), return as float.
    - If string date without time (YYYY-MM-DD), interpret as end-of-day market close
      at 16:00 America/New_York for that calendar date (DST-aware), then convert to UTC.
    - Otherwise try robust UTC parsing.
    """
    # Numeric epoch seconds (keep as-is)
    if isinstance(exp_raw, (int, float)) and exp_raw > 10_000:
        try:
            return float(exp_raw)
        except Exception:
            return None

    exp_str = str(exp_raw) if exp_raw is not None else None
    if not exp_str:
        return None

    # Try parsing generically first
    try:
        dt_utc = pd.to_datetime(exp_str, utc=True)
    except Exception:
        try:
            dt_utc = pd.to_datetime(exp_str).tz_localize("UTC")
        except Exception:
            dt_utc = None

    # If this is a pure date (YYYY-MM-DD), build 16:00 America/New_York and convert
    is_date_only = isinstance(exp_str, str) and len(exp_str) == 10 and exp_str[4] == "-" and exp_str[7] == "-"
    if is_date_only:
        from datetime import datetime, time
        try:
            from zoneinfo import ZoneInfo  # Python 3.9+
            ny = ZoneInfo("America/New_York")
            y, m, d = map(int, exp_str.split("-"))
            dt_ny = datetime(y, m, d, 16, 0, 0, tzinfo=ny)  # 16:00 ET (DST-aware)
            ts = float(dt_ny.astimezone(ZoneInfo("UTC")).timestamp())
            return ts
        except Exception:
            # Fallback: keep previous UTC parse if available; else assume 20:00 UTC
            if dt_utc is not None:
                if dt_utc.hour == 0 and dt_utc.minute == 0 and dt_utc.second == 0:
                    return float(dt_utc.timestamp() + 20*3600.0)
                return float(dt_utc.timestamp())
            try:
                naive = pd.to_datetime(exp_str)
                return float(naive.tz_localize("UTC").timestamp() + 20*3600.0)
            except Exception:
                return None

    # If not date-only, return the robust UTC parse if present
    if dt_utc is not None:
        return float(dt_utc.timestamp())

    # As a final fallback, try naive -> UTC
    try:
        return float(pd.to_datetime(exp_str).tz_localize("UTC").timestamp())
    except Exception:
        return None

_SEC_PER_YEAR = 365.25 * 24 * 3600.0

def _yearfrac(now_ts: float, exp_ts: float) -> float:
    if exp_ts is None:
        return float('nan')
    T = max(0.0, (exp_ts - now_ts) / _SEC_PER_YEAR)
    return T

def _phi(x: np.ndarray) -> np.ndarray:
    # Стандартная нормальная pdf
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def _ndist(x: np.ndarray) -> np.ndarray:
    """Стандартная нормальная CDF без np.erf (используем math.erf)."""
    x = np.asarray(x, float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))

def _bs_price(side: str, S: float, K: float, T: float, sigma: float, r: float = 0.0, q: float = 0.0) -> float:
    if T <= 0.0 or sigma <= 0.0:
        if side == "C":
            return max(0.0, S - K)
        else:
            return max(0.0, K - S)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if side == "C":
        return S * np.exp(-q * T) * _ndist(d1) - K * np.exp(-r * T) * _ndist(d2)
    else:
        return K * np.exp(-r * T) * _ndist(-d2) - S * np.exp(-q * T) * _ndist(-d1)

def _bs_delta(side: str, S: float, K: float, T: float, sigma: float, r: float = 0.0, q: float = 0.0) -> float:
    if T <= 0.0 or sigma <= 0.0:
        if side == "C":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if side == "C":
        return np.exp(-q * T) * _ndist(d1)
    else:
        return np.exp(-q * T) * (_ndist(d1) - 1.0)

def _bs_gamma(S: float, K: float, T: float, sigma: float, r: float = 0.0, q: float = 0.0) -> float:
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return np.exp(-q * T) * _phi(d1) / (S * sigma * np.sqrt(T))

def _bs_vega(S: float, K: float, T: float, sigma: float, r: float = 0.0, q: float = 0.0) -> float:
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * _phi(d1) * np.sqrt(T)

def _bs_iv_from_price(side: str, price: float, S: float, K: float, T: float, r: float = 0.0, q: float = 0.0, max_iter: int = 100, eps: float = 1e-6) -> float:
    if T <= 0.0:
        return float('nan')
    if price <= 0.0:
        return 0.0
    sigma = 0.5  # initial guess
    for _ in range(max_iter):
        bs_p = _bs_price(side, S, K, T, sigma, r, q)
        vega = _bs_vega(S, K, T, sigma, r, q)
        if vega < 1e-8:
            break
        diff = bs_p - price
        if abs(diff) < eps:
            break
        sigma -= diff / vega
        if sigma < 0.0:
            sigma = 0.001
    return sigma if sigma > 0.0 else float('nan')

# NEW: SABR volatility function
def sabr_vol(F, K, T, alpha, beta, rho, nu):
    if abs(F - K) < 1e-6:
        return alpha / (F**(1 - beta)) * (1 + ((2 - 3*rho**2)/24 * nu**2 * T))
    z = nu / alpha * (F * K)**((1 - beta)/2) * np.log(F / K)
    chi = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))
    return (alpha * z / chi / (F * K)**((1 - beta)/2) * (1 + ((1 - beta)**2 / 24 * np.log(F / K)**2 + (1 - beta)**4 / 1920 * np.log(F / K)**4) + ((2 - 3*rho**2)/24 * nu**2 * T)))

@dataclass
class SanitizerConfig:
    # Флаги аномалий (по умолчанию все включены)
    flag_iv_anom: bool = True
    flag_greeks_anom: bool = True
    flag_delta_anom: bool = True
    flag_data_anom: bool = True

    # Пороги для аномалий
    iv_min: float = 0.0001
    iv_max: float = 10.0
    gamma_min: float = 0.0
    gamma_max: float = 1.0
    delta_min: float = -1.0
    delta_max: float = 1.0
    oi_min: float = 0.0
    vol_min: float = 0.0

    # Для восстановления IV (BS implied)
    r: float = 0.0  # risk-free rate
    q: float = 0.0  # dividend yield
    max_iter_iv: int = 100
    eps_iv: float = 1e-6

    # Для окна страйков
    w_cp_diff: float = 0.4  # вес для |call OI - put OI|
    w_liquidity: float = 0.3  # вес для liquidity (vol + oi)
    w_dollar_gamma: float = 0.3  # вес для dollar gamma
    window_size: int = 20  # целевой размер окна (страйков)

    advanced_mode: bool = False  # NEW: Флаг для SABR и т.д.

def make_raw_df(raw_records: List[dict], now_ts: float = None) -> pd.DataFrame:
    # ... (оригинальный код, без изменений)

def flag_anomalies(df_raw: pd.DataFrame, cfg: SanitizerConfig = SanitizerConfig()) -> pd.DataFrame:
    # ... (оригинальный код, без изменений)

def rebuild_iv_and_greeks(df_marked: pd.DataFrame, cfg: SanitizerConfig = SanitizerConfig()) -> pd.DataFrame:
    df_corr = df_marked.copy()

    # Восстановление IV для строк с аномалиями IV или греков
    mask_anom = df_corr["anom_iv"] | df_corr["anom_greeks"]
    if mask_anom.any():
        for idx in df_corr.index[mask_anom]:
            row = df_corr.loc[idx]
            side = row["side"]
            price = row["last"] if np.isfinite(row["last"]) else (row["bid"] + row["ask"]) / 2.0 if np.isfinite(row["bid"]) and np.isfinite(row["ask"]) else float('nan')
            if not np.isfinite(price):
                continue
            S = row["S"]
            K = row["K"]
            T = row["T"]
            iv_new = _bs_iv_from_price(side, price, S, K, T, cfg.r, cfg.q, cfg.max_iter_iv, cfg.eps_iv)
            if np.isfinite(iv_new):
                df_corr.loc[idx, "iv_corr"] = iv_new

    # Пересчёт греков на основе corrected IV
    for idx in df_corr.index:
        row = df_corr.loc[idx]
        side = row["side"]
        S = row["S"]
        K = row["K"]
        T = row["T"]
        iv_corr = row["iv_corr"]
        if np.isfinite(iv_corr):
            delta_new = _bs_delta(side, S, K, T, iv_corr, cfg.r, cfg.q)
            gamma_new = _bs_gamma(S, K, T, iv_corr, cfg.r, cfg.q)
            df_corr.loc[idx, "delta_corr"] = delta_new
            df_corr.loc[idx, "gamma_corr"] = gamma_new

    # IMPROVED: Если advanced_mode, используем SABR для IV correction
    if cfg.advanced_mode:
        for exp in sorted(df_marked["exp"].unique()):
            g = df_marked[df_marked["exp"] == exp].copy()
            if g.empty:
                continue

            Ks = g["K"].values.astype(float)
            iv_obs = g["iv"].values.astype(float)  # Используем observed IV

            mask = np.isfinite(iv_obs) & (iv_obs > 0)
            Ks_valid = Ks[mask]
            iv_valid = iv_obs[mask]

            if len(Ks_valid) < 4:
                continue  # Fallback to BS

            S_exp = float(np.nanmedian(g["S"].values))
            F_exp = float(np.nanmedian(g.get("F", g["S"]).values)) if "F" in g.columns else S_exp
            T_med = float(np.nanmedian(g["T"].values))

            # SABR calibration (beta=0.5 fixed for equity)
            def sabr_residuals(params):
                alpha, rho, nu = params
                model_iv = [sabr_vol(F_exp, k, T_med, alpha, 0.5, rho, nu) for k in Ks_valid]
                return np.array(model_iv) - iv_valid

            initial_params = [0.1, -0.5, 0.3]
            bounds = ([0, -1, 0], [np.inf, 1, np.inf])
            res = least_squares(sabr_residuals, initial_params, bounds=bounds)

            if res.success:
                alpha, rho, nu = res.x
                sabr_iv = np.array([sabr_vol(F_exp, k, T_med, alpha, 0.5, rho, nu) for k in Ks])
                df_corr.loc[g.index, "iv_corr"] = sabr_iv  # Заменяем corrected IV на SABR

    # Пересчёт греков на основе corrected IV (для всех, включая SABR)
    for idx in df_corr.index:
        row = df_corr.loc[idx]
        side = row["side"]
        S = row["S"]
        K = row["K"]
        T = row["T"]
        iv_corr = row["iv_corr"]
        if np.isfinite(iv_corr):
            delta_new = _bs_delta(side, S, K, T, iv_corr, cfg.r, cfg.q)
            gamma_new = _bs_gamma(S, K, T, iv_corr, cfg.r, cfg.q)
            df_corr.loc[idx, "delta_corr"] = delta_new
            df_corr.loc[idx, "gamma_corr"] = gamma_new

    return df_corr

def compute_window_weights(df_corr: pd.DataFrame, cfg: SanitizerConfig = SanitizerConfig()) -> pd.DataFrame:
    # ... (оригинальный код, без изменений)

def select_windows(df_weights: pd.DataFrame, cfg: SanitizerConfig = SanitizerConfig()) -> Dict[str, np.ndarray]:
    # ... (оригинальный код, без изменений)

def sanitize_and_window_pipeline(raw_records: List[dict], S: float, now_ts: Optional[float] = None, cfg: SanitizerConfig = SanitizerConfig()) -> Dict[str, Any]:
    # ... (оригинальный код, без изменений, но передавать cfg с advanced_mode если нужно)

def build_window_panels(
    df_corr: pd.DataFrame,
    df_weights: pd.DataFrame,
    windows: Dict[str, np.ndarray]
) -> Dict[str, pd.DataFrame]:
    # ... (оригинальный код, без изменений)
