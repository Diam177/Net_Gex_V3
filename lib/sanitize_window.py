# -*- coding: utf-8 -*-
"""
sanitize_window.py — ... (оригинальный docstring)
"""

# ... (оригинальный импорт)
from scipy.optimize import least_squares  # NEW: для SABR calibration

# ... (оригинальный код до rebuild_iv_and_greeks)

def rebuild_iv_and_greeks(df_marked: pd.DataFrame, cfg: SanitizerConfig = SanitizerConfig()) -> pd.DataFrame:
    # ORIGINAL код ...
    
    # NEW: Добавляем опцию для advanced_mode с SABR
    if cfg.advanced_mode:  # IMPROVED: Флаг из config
        # SABR calibration per exp
        for exp in df_marked['exp'].unique():
            g = df_marked[df_marked['exp'] == exp]
            if g.empty: continue
            
            # Собираем observed IV и strikes
            Ks = g['K'].values
            iv_obs = g['iv'].values  # или 'iv_corr' если уже
            T = np.median(g['T'].values)
            F = np.median(g.get('F', g['S']).values)  # forward or S
            
            # Filter valid
            mask = np.isfinite(iv_obs) & (iv_obs > 0)
            Ks_valid, iv_valid = Ks[mask], iv_obs[mask]
            
            if len(Ks_valid) < 4:  # Min for fit
                continue  # Fallback to BS
            
            # Calibrate SABR (beta=0.5 fixed)
            def calibrate(params):
                alpha, rho, nu = params
                model_iv = []
                for k in Ks_valid:
                    if F == k:
                        vol = alpha / (F**(1-0.5)) * (1 + ... )  # Полная SABR formula как в тесте
                    else:
                        # ... (вставить полную sabr_vol из моего теста code)
                        vol = sabr_vol(F, k, T, alpha, 0.5, rho, nu)
                    model_iv.append(vol)
                return np.array(model_iv) - iv_valid
            
            initial = [0.1, -0.5, 0.3]
            res = least_squares(calibrate, initial, bounds=([0,-1,0], [np.inf,1,np.inf]))
            if res.success:
                alpha, rho, nu = res.x
                # Apply to all strikes
                sabr_iv = np.array([sabr_vol(F, k, T, alpha, 0.5, rho, nu) for k in Ks])
                df_marked.loc[g.index, 'iv_corr'] = sabr_iv  # IMPROVED: Use SABR IV
            
            # Затем пересчитать greeks с новым IV (оригинальный BS код)
    
    # ORIGINAL: Продолжить с BS ...
    return df_corr

# ... (остальной оригинальный код)
