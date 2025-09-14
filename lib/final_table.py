
from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class FinalTableScaleCfg:
    scale: float = 1.0

@dataclass
class FinalTableConfig:
    scale_millions: bool = True

def _side_agg(df: pd.DataFrame, side_label: str, available: set) -> pd.DataFrame:
    mapping = {
        "iv_corr":    f"iv_{'call' if side_label=='C' else 'put'}",
        "delta_corr": f"delta_{'call' if side_label=='C' else 'put'}",
        "gamma_corr": f"gamma_{'call' if side_label=='C' else 'put'}",
        "vega_corr":  f"vega_{'call' if side_label=='C' else 'put'}",
        "theta_corr": f"theta_{'call' if side_label=='C' else 'put'}",
    }
    use_cols = {k:v for k,v in mapping.items() if k in available}
    if not use_cols:
        return pd.DataFrame(index=pd.Index([], name="K"))
    agg = df[df["side"]==side_label].groupby("K", as_index=True).agg({k:"median" for k in use_cols.keys()})
    return agg.rename(columns=use_cols)

def _attach_iv_greeks(net_tbl: pd.DataFrame, df_corr: pd.DataFrame, exp: str) -> pd.DataFrame:
    cols_set = set(df_corr.columns)
    if not {'exp','K','side'}.issubset(cols_set):
        return net_tbl
    g = df_corr[(df_corr['exp']==exp) & (df_corr['K'].isin(net_tbl['K']))].copy()
    if g.empty:
        return net_tbl
    calls = _side_agg(g, "C", cols_set)
    puts  = _side_agg(g, "P", cols_set)
    for frag in (calls, puts):
        if not frag.empty:
            net_tbl = net_tbl.merge(frag, left_on="K", right_index=True, how="left")
    return net_tbl

def build_final_tables_from_corr(df_corr: pd.DataFrame,
                                 windows: Dict[str, List[int]],
                                 scale: float,
                                 cfg: Optional[FinalTableConfig] = None) -> Dict[str, pd.DataFrame]:
    from netgex_ag import compute_netgex_ag_per_expiry  # avoid cycles
    from power_zone_er import compute_power_zone_and_er

    result: Dict[str, pd.DataFrame] = {}
    expirations = list(windows.keys())

    for exp in expirations:
        strikes_eval = windows[exp]

        # Important: pass object with attribute .scale to avoid `'dict' object has no attribute 'scale'`
        net_tbl = compute_netgex_ag_per_expiry(df_corr, exp, windows, FinalTableScaleCfg(scale=scale))

        # Try attach PZ/ER if function available
        try:
            S = df_corr.loc[df_corr["exp"]==exp, "S"].dropna().iloc[0]
        except Exception:
            S = None
        try:
            day_high = df_corr.loc[df_corr["exp"]==exp, "day_high"].dropna().iloc[0]
            day_low  = df_corr.loc[df_corr["exp"]==exp, "day_low"].dropna().iloc[0]
        except Exception:
            day_high = day_low = None
        try:
            pz_tbl = compute_power_zone_and_er(S, strikes_eval, None, day_high, day_low)
            if set(["K","PZ","ER_Up","ER_Down"]).issubset(set(pz_tbl.columns)):
                net_tbl = net_tbl.merge(pz_tbl[["K","PZ","ER_Up","ER_Down"]], on="K", how="left")
        except Exception:
            pass

        # Attach IV & Greeks
        try:
            net_tbl = _attach_iv_greeks(net_tbl, df_corr, exp)
        except Exception:
            pass

        desired = ["K","S","F","call_oi","put_oi","call_vol","put_vol",
                   "iv_call","iv_put","delta_call","delta_put","gamma_call","gamma_put",
                   "vega_call","vega_put","theta_call","theta_put",
                   "dg1pct_call","dg1pct_put","AG_1pct","NetGEX_1pct","AG_1pct_M","NetGEX_1pct_M",
                   "PZ","ER_Up","ER_Down"]
        cols = [c for c in desired if c in net_tbl.columns] + [c for c in net_tbl.columns if c not in desired]
        net_tbl = net_tbl[cols]
        result[exp] = net_tbl.reset_index(drop=True)

    return result

def process_from_raw(raw_records: list,
                     S: float,
                     sanitizer_cfg=None,
                     final_cfg: Optional[FinalTableConfig] = None) -> Dict[str, pd.DataFrame]:
    """Full pipeline from raw provider records to final tables by expiration."""
    from sanitize_window import sanitize_and_window_pipeline, SanitizerConfig
    if sanitizer_cfg is None:
        sanitizer_cfg = SanitizerConfig()
    df_corr, windows = sanitize_and_window_pipeline(raw_records, S, now=None, shares_per_contract=100, cfg=sanitizer_cfg)
    scale = 1_000_000.0 if (final_cfg and final_cfg.scale_millions) else 1.0
    return build_final_tables_from_corr(df_corr, windows, scale, final_cfg or FinalTableConfig())
