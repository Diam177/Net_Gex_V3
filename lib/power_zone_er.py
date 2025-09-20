# ... (оригинальный docstring)

# NEW imports
import torch
import torch.nn as nn
from scipy.stats import multivariate_normal  # Для copula approx
from sklearn.decomposition import PCA  # No sklearn; use np.linalg.svd instead

def compute_power_zone_and_er(... , advanced_mode=False):  # NEW param
    if not advanced_mode:
        # ORIGINAL код ...
    
    # IMPROVED: Fit params with torch (e.g., eta, zeta)
    # Assume historical data stub; in real fetch or default
    # Dummy fit
    optimizer = torch.optim.SGD(...)  # Как в тесте
    # Fit eta, zeta on pinning data (stub: eta=0.6, zeta=0.4)
    
    # Kernel: Epanechnikov instead Gaussian
    def epanechnikov(u):
        return 0.75 * (1 - u**2) * (np.abs(u) <= 1)
    dist_eval = (strikes_eval - S) / h
    Wd_eval = epanechnikov(dist_eval)
    
    # Stab: Gaussian copula approx
    # Compute cov from gamma_net and vol
    cov = np.cov(gamma_net_share, vol)
    copula_stab = multivariate_normal.cdf(...)  # Approx joint prob low vol high stab
    
    # Act: PCA
    X_act = np.vstack([G_norm, V_norm, np.log(1+vol_vec)]).T
    U, S, Vt = np.linalg.svd(X_act, full_matrices=False)
    act_pca = U[:,0]  # First component
    
    # PZ = ... with new hats
    
    # Для ER: Nelder-Mead fit alpha_g etc.
    from scipy.optimize import minimize
    def obj(params):
        alpha_g, alpha_v = params
        # MSE on historical moves (stub data)
        return loss
    res = minimize(obj, [1,1], method='Nelder-Mead')
    
    # Regime HMM: Simple 2-state
    # ... implement with statsmodels or manual
    
    return (pz_norm, er_up_norm, er_down_norm)
