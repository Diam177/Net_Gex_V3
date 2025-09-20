# ... (оригинальный docstring)

# NEW import
from scipy.interpolate import CubicSpline  # Для cubic G-Flip
from sklearn.utils import resample  # Но env no sklearn; use pd.sample instead

@dataclass
class NetGEXAGConfig:
    # ORIGINAL ...
    advanced_mode: bool = False  # NEW
    market_cap: float = 654.8e9  # From search, SPY example
    adv: float = 70e6  # From search
    decay_theta: float = 0.5  # Fitted default
    n_bootstrap: int = 1000  # For CI

# ... (оригинальный _ensure_required_columns)

def compute_netgex_ag_per_expiry(...):
    if not cfg.advanced_mode:
        # ORIGINAL код ...
        return pivot
    
    # IMPROVED: Advanced mode
    # 1. Bootstrap for CI
    netgex_boot = []
    for _ in range(cfg.n_bootstrap):
        df_boot = df_corr.sample(frac=1, replace=True)  # pd resample
        # Compute as original on boot
        boot_tbl = compute_netgex_ag_per_expiry(df_boot, exp, windows, NetGEXAGConfig())  # Fallback config
        netgex_boot.append(boot_tbl['NetGEX_1pct'].values)
    
    netgex_mean = np.mean(netgex_boot, axis=0)
    netgex_std = np.std(netgex_boot, axis=0)
    
    # 2. Normalization
    adv_factor = cfg.adv / 1e6
    netgex_norm = netgex_mean / (cfg.market_cap * adv_factor)
    
    # 3. G-Flip with cubic
    Ks = pivot['K'].values
    if len(Ks) > 2:
        spline = CubicSpline(Ks, netgex_norm)
        from scipy.optimize import root_scalar
        def f(k): return spline(k)
        try:
            gflip = root_scalar(f, bracket=[min(Ks), max(Ks)]).root
        except:
            gflip = None
    else:
        gflip = None
    
    # Add to table (e.g., new cols: NetGEX_norm, NetGEX_std, GFlip)
    pivot['NetGEX_norm'] = netgex_norm
    pivot['NetGEX_std'] = netgex_std
    # ... integrate gflip if needed
    
    return pivot

# Для multi: Weighted sum with decay
def compute_netgex_ag(...):
    if not cfg.advanced_mode:
        # ORIGINAL
    # IMPROVED
    # ... compute T_exp from df, w = OI * np.exp(-cfg.decay_theta * T_exp)
    # Sum weighted

# ... (остальной код)
