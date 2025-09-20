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
__all__ = ['FinalTableConfig', 'build_final_tables_from_corr', 'process_from_raw', '_series_ctx_from_corr']


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
from lib.power_zone_er import compute_power_zone_and_er


# --------- конфиг ---------

@dataclass
class FinalTableConfig:
    # Масштаб для млн $ в колонках *_M (используется в netgex_ag)
    scale_millions: float = 1e6
    # Параметры для Power Zone / ER (оставляем значения по умолчанию из power_zone_er)
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    advanced_mode: bool = False  # NEW: Флаг для улучшенного режима
    market_cap: float = 654.8e9  # NEW: Для normalization
    adv: float = 70e6  # NEW

# ... (оригинальные helpers)

def build_final_tables_from_corr(
    df_corr: pd.DataFrame,
    windows: Dict[str, np.ndarray],
    cfg: FinalTableConfig = FinalTableConfig(),
) -> Dict[str, pd.DataFrame]:
    results: Dict[str, pd.DataFrame] = {}

    # IMPROVED: Передаём advanced_mode в config для netgex и power_zone
    net_cfg = NetGEXAGConfig(scale=cfg.scale_millions, advanced_mode=cfg.advanced_mode, market_cap=cfg.market_cap, adv=cfg.adv)

    for exp in sorted(df_corr["exp"].dropna().unique()):
        # 1) таблица NetGEX/AG по окну с новым config
        net_tbl = compute_netgex_ag_per_expiry(
            df_corr, exp, windows=windows,
            cfg=net_cfg  # IMPROVED
        )
        # ... (оригинальный код для vol добавления)

        # 3) PZ/ER с advanced_mode
        pz, er_up, er_down = compute_power_zone_and_er(
            S=...,
            strikes_eval=...,
            all_series_ctx=...,
            day_high=cfg.day_high,
            day_low=cfg.day_low,
            advanced_mode=cfg.advanced_mode  # NEW
        )
        # ... (оригинальный mapping)

    return results

# ... (остальной оригинальный код для process_from_raw, передавать cfg)
