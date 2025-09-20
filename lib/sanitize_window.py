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

    if dt_utc is not None:
        return float(dt_utc.timestamp())

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

# ... (остальной оригинальный код для BS и других функций, я опускаю для краткости, но в реале он остаётся без изменений)

# NEW: SABR volatility function
def sabr_vol(F, K, T, alpha, beta, rho, nu):
    if abs(F - K) < 1e-6:
        return alpha / (F**(1 - beta)) * (1 + ((2 - 3*rho**2)/24 * nu**2 * T))
    z = nu / alpha * (F * K)**((1 - beta)/2) * np.log(F / K)
    chi = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))
    return alpha * z / chi / (F * K)**((1 - beta)/2) * (1 + ((1 - beta)**2 / 24 * np.log(F / K)**2 + (1 - beta)**4 / 1920 * np.log(F / K)**4) + ((2 - 3*rho**2)/24 * nu**2 * T))

def rebuild_iv_and_greeks(df_marked: pd.DataFrame, cfg: SanitizerConfig = SanitizerConfig()) -> pd.DataFrame:
    # ORIGINAL код для BS восстановления ...

    # IMPROVED: Если advanced_mode, используем SABR для IV correction
    if getattr(cfg, 'advanced_mode', False):  # NEW: Проверяем флаг
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
                df_marked.loc[g.index, "iv_corr"] = sabr_iv  # Заменяем corrected IV на SABR

    # Продолжаем с пересчётом греков на основе corrected IV (оригинальный код BS остаётся)
    # ... (оригинальный код для греков)

    return df_corr

# ... (остальной оригинальный код для window selection и т.д.)
