# -*- coding: utf-8 -*-
"""
power_zone_er.py — расчёт Power Zone (PZ) и Easy Reach (ER Up / ER Down)
по формулировке, использованной в проекте (функция compute_power_zone_and_er).
"""

from __future__ import annotations
import math
from typing import Iterable, Tuple, Dict, Any, List, Optional
import numpy as np

# NEW imports
import torch
import torch.optim as optim
from scipy.stats import multivariate_normal  # Для copula
from scipy.optimize import minimize  # Для Nelder-Mead
from hmmlearn.hmm import GaussianHMM  # NEW dep, но если нет — manual impl; assume added

def compute_power_zone_and_er(
    S: float,
    strikes_eval: Iterable[float],
    all_series_ctx: Iterable[Dict[str, Any]],
    day_high: Optional[float] = None,
    day_low: Optional[float] = None,
    advanced_mode: bool = False,  # NEW: Флаг
    * ,  # ORIGINAL params
    beta0: float = 1.0,
    eta: float = 0.60,
    zeta: float = 0.40,
    alpha_g: float = 1.0,
    alpha_v: float = 1.0,
    theta_short_gamma: float = 0.5,
    c_vol_log: float = 4.0,
    d_star: float = 2.0,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not advanced_mode:
        # ORIGINAL код полностью ...
        return (pz_norm, er_up_norm, er_down_norm)

    # IMPROVED: Advanced mode
    # 1. Fit params (eta, zeta etc.) with torch on stub historical (in real: load data)
    # Stub: assume historical pinning data as tensor
    historical_data = torch.tensor([[...]])  # Placeholder
    params = torch.tensor([eta, zeta], requires_grad=True)
    optimizer = optim.SGD([params], lr=0.01)
    for _ in range(100):
        loss = ...  # MSE on pinning
        loss.backward()
        optimizer.step()
    eta, zeta = params.detach().numpy()

    # Kernel: Epanechnikov
    dist_eval = (strikes_eval - S) / h
    Wd_eval = np.where(np.abs(dist_eval) <= 1, 0.75 * (1 - dist_eval**2), 0)

    # h by Silverman
    h = 1.06 * np.std(strikes_eval) * len(strikes_eval)**(-1/5)

    # Stab_hat: Gaussian copula
    gamma_net_share = np.array(...)  # From ctx
    vol = np.array(...)  # From ctx
    cov = np.cov(gamma_net_share, vol)
    copula = multivariate_normal(mean=[0,0], cov=cov)
    stab_hat = copula.cdf(np.vstack([gamma_net_share, vol]).T)  # Joint prob

    # Act_hat: PCA via svd
    X = np.vstack([G_norm, V_norm, np.log(1 + vol_vec)]).T
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    act_hat = U[:, 0]  # First PC

    # PZ computation with new hats
    # ... аналогично original, но с новыми

    # Для ER: Fit alphas with Nelder-Mead
    def obj(params):
        alpha_g, alpha_v = params
        denom = (G_norm**alpha_g * V_norm**alpha_v) + eps
        er_pred = Wd_eval / denom
        # MSE vs historical moves (stub)
        return np.mean((er_pred - historical_er)**2)
    res = minimize(obj, [1.0, 1.0], method='Nelder-Mead')
    alpha_g, alpha_v = res.x

    # Regime: HMM for short/long gamma
    hmm = GaussianHMM(n_components=2)
    hmm.fit(gamma_net_share.reshape(-1,1))
    states = hmm.predict(gamma_net_share.reshape(-1,1))
    theta = 0.5 if np.mean(states) > 0.5 else 0  # Probabilistic

    # ... compute ER with fitted

    return (pz_norm, er_up_norm, er_down_norm)
