import math, numpy as np
from math import log, sqrt
from dateutil import tz

M_CONTRACT = 100.0

def phi(x):
    return math.exp(-0.5*x*x) / math.sqrt(2*math.pi)

def N(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def bs_d1(S, K, sigma, T, r=0.0):
    sigma = max(float(sigma), 1e-8); T = max(float(T), 1e-8)
    return (math.log(S / K) + (r + 0.5*sigma*sigma)*T) / (sigma*math.sqrt(T))

def bs_gamma(S, K, sigma, T, r=0.0):
    d1 = bs_d1(S, K, sigma, T, r)
    return phi(d1) / (S * max(sigma,1e-8) * math.sqrt(max(T,1e-8)))

def robust_fill_iv(iv_dict, strikes, fallback=0.30):
    values = [v for v in iv_dict.values() if v is not None and v>0 and not math.isnan(v)]
    use_fallback = float(np.median(values)) if len(values)>0 else float(fallback)
    for k in strikes:
        v = iv_dict.get(k, None)
        if v is None or v<=0 or (isinstance(v, float) and math.isnan(v)):
            best_val, best_dist = None, 1e18
            for kk, vv in iv_dict.items():
                if vv is not None and vv>0 and (not isinstance(vv, float) or not math.isnan(vv)):
                    d = abs(kk - k)
                    if d < best_dist:
                        best_dist, best_val = d, vv
            iv_dict[k] = best_val if best_val is not None else use_fallback
    return iv_dict

def extract_core_from_chain(chain_json):
    res = chain_json.get("optionChain", {}).get("result", [])
    if not res:
        raise ValueError("Invalid chain JSON: missing optionChain.result[0]")
    root = res[0]
    quote = root.get("quote", {})
    S = float(quote.get("regularMarketPrice", quote.get("postMarketPrice", 0.0)))
    t0 = int(quote.get("regularMarketTime", quote.get("postMarketTime", 0)))
    expirations = root.get("expirationDates", [])
    if not isinstance(expirations, list) or len(expirations)==0:
        expirations = []
        for blk in root.get("options", []):
            e = blk.get("expirationDate", None)
            if e is not None: expirations.append(int(e))
    blocks_by_date = {}
    for blk in root.get("options", []):
        if "expirationDate" in blk:
            blocks_by_date[int(blk["expirationDate"])] = blk
    return quote, t0, S, expirations, blocks_by_date

def aggregate_series(block):
    calls = block.get("calls", [])
    puts  = block.get("puts",  [])
    strikes = sorted(set([float(r.get("strike")) for r in calls + puts if r.get("strike") is not None]))
    call_oi = {float(r["strike"]): int(r.get("openInterest") or 0) for r in calls}
    put_oi  = {float(r["strike"]): int(r.get("openInterest") or 0) for r in puts}
    call_vol= {float(r["strike"]): int(r.get("volume") or 0) if r.get("volume") is not None else 0 for r in calls}
    put_vol = {float(r["strike"]): int(r.get("volume") or 0) if r.get("volume") is not None else 0 for r in puts}
    iv_call = {}
    iv_put  = {}
    for r in calls:
        k = float(r["strike"]); iv = r.get("impliedVolatility", None)
        if iv is not None and (not isinstance(iv, float) or not math.isnan(iv)) and float(iv)>0:
            iv_call[k] = float(iv)
    for r in puts:
        k = float(r["strike"]); iv = r.get("impliedVolatility", None)
        if iv is not None and (not isinstance(iv, float) or not math.isnan(iv)) and float(iv)>0:
            iv_put[k] = float(iv)
    call_oi = {k: int(call_oi.get(k,0)) for k in strikes}
    put_oi  = {k: int(put_oi.get(k,0))  for k in strikes}
    call_vol= {k: int(call_vol.get(k,0)) for k in strikes}
    put_vol = {k: int(put_vol.get(k,0))  for k in strikes}
    iv_call = robust_fill_iv({k: iv_call.get(k, None) for k in strikes}, strikes)
    iv_put  = robust_fill_iv({k: iv_put.get(k, None)  for k in strikes}, strikes)
    return strikes, call_oi, put_oi, call_vol, put_vol, iv_call, iv_put

def compute_k_scale(S, K_atm, sigma_atm, T):
    return bs_gamma(S, K_atm, sigma_atm, T) * S * M_CONTRACT / 1000.0

def pick_sigma_atm(K_atm, iv_call, iv_put, strikes):
    vals = []
    if iv_call.get(K_atm) is not None: vals.append(iv_call[K_atm])
    if iv_put.get(K_atm)  is not None: vals.append(iv_put[K_atm])
    if len(vals)>0: return float(sum(vals)/len(vals))
    cand = [(k, iv_call.get(k)) for k in strikes if iv_call.get(k)]
    cand += [(k, iv_put.get(k))  for k in strikes if iv_put.get(k)]
    if cand:
        return float(sorted(cand, key=lambda kv: abs(kv[0]-K_atm))[0][1])
    return 0.30

def compute_series_metrics_for_expiry(S, t0, expiry_unix, block, day_high=None, day_low=None, all_series=None):
    import numpy as np
    strikes, call_oi, put_oi, call_vol, put_vol, iv_call, iv_put = aggregate_series(block)

    T = max((expiry_unix - t0) / (365*24*3600), 1e-6)
    K_atm = min(strikes, key=lambda k: abs(k - S))
    sigma_atm = pick_sigma_atm(K_atm, iv_call, iv_put, strikes)
    k_scale = compute_k_scale(S, K_atm, sigma_atm, T)

    call_oi_arr = np.array([call_oi[k] for k in strikes], dtype=float)
    put_oi_arr  = np.array([put_oi[k]  for k in strikes], dtype=float)
    call_vol_arr= np.array([call_vol[k] for k in strikes], dtype=float)
    put_vol_arr = np.array([put_vol[k]  for k in strikes], dtype=float)

    delta_oi = call_oi_arr - put_oi_arr
    net_gex = np.round(k_scale * delta_oi, 1)
    ag      = np.round(k_scale * (call_oi_arr + put_oi_arr), 1)

    if all_series is None:
        pz = np.zeros_like(net_gex)
        pz_fp = np.zeros_like(net_gex)
    else:
        pz, pz_fp = compute_pz_and_flow(S, t0, strikes, sigma_atm, day_high, day_low, all_series)

    return {
        "strikes": strikes,
        "put_oi": put_oi_arr.astype(int),
        "call_oi": call_oi_arr.astype(int),
        "put_vol": put_vol_arr.astype(int),
        "call_vol": call_vol_arr.astype(int),
        "net_gex": net_gex,
        "ag": ag,
        "pz": pz,
        "pz_fp": pz_fp
    }

def compute_pz_and_flow(S, t0, strikes_eval, sigma_atm, day_high, day_low, all_series):
    import numpy as np
    tau = 0.5
    if isinstance(day_high, (int,float)) and isinstance(day_low, (int,float)):
        R = float(day_high) - float(day_low)
    else:
        R = 0.01*S

    eta, zeta = 0.60, 0.40
    total_oi = np.array([
        sum(s["call_oi"].values()) + sum(s["put_oi"].values()) for s in all_series
    ], dtype=float)
    T_arr = np.array([s["T"] for s in all_series], dtype=float)
    W_time = total_oi * (T_arr**(-eta)) * ((1.0 - tau)**zeta)
    W_time = (W_time / W_time.sum()) if W_time.sum()>0 else np.ones_like(W_time)/len(W_time)

    def median_step(xs):
        if len(xs) < 2: return 1.0
        diffs = sorted([abs(xs[i+1]-xs[i]) for i in range(len(xs)-1)])
        mid = len(diffs)//2
        return diffs[mid] if len(diffs)%2==1 else 0.5*(diffs[mid-1]+diffs[mid])
    step = median_step(list(strikes_eval))
    beta0 = 1.0
    h_base = beta0 * S * max(sigma_atm, 1e-8) / math.sqrt(252.0)
    beta_eff = beta0 * max(0.8, min(1.3, R / (2.0 * max(h_base, 1e-8)) ))
    h_min = 0.25 * step
    h_max = 0.025 * S
    h = min(max(beta_eff * S * max(sigma_atm,1e-8) / math.sqrt(252.0), h_min), h_max)

    def Wd_kernel(K, S, h):
        if h <= 0: return 0.0
        x = (K - S) / h
        return math.exp(-0.5 * x * x)

    prep = []
    for s in all_series:
        Ks = np.array(s["strikes"], dtype=float)
        iv_mean = {k: 0.5*(s["iv_call"].get(k,0.0) + s["iv_put"].get(k,0.0)) for k in s["strikes"]}
        iv_mean = robust_fill_iv(iv_mean, s["strikes"])
        iv_vec = np.array([iv_mean[k] for k in s["strikes"]], dtype=float)

        call_oi_vec = np.array([s["call_oi"][k] for k in s["strikes"]], dtype=float)
        put_oi_vec  = np.array([s["put_oi"][k]  for k in s["strikes"]], dtype=float)
        vol_vec     = np.array([s["call_vol"][k] + s["put_vol"][k] for k in s["strikes"]], dtype=float)
        dOI_vec     = call_oi_vec - put_oi_vec

        gamma_vec = np.array([bs_gamma(S, float(k), float(sig), s["T"]) for k, sig in zip(s["strikes"], iv_vec)], dtype=float)
        dollar_per_unit = S * M_CONTRACT / 1000.0
        AG_e = gamma_vec * (call_oi_vec + put_oi_vec) * dollar_per_unit
        NG_e = gamma_vec * np.abs(dOI_vec) * dollar_per_unit

        def median_step2(xs):
            if len(xs) < 2: return 1.0
            diffs = sorted([abs(xs[i+1]-xs[i]) for i in range(len(xs)-1)])
            mid = len(diffs)//2
            return diffs[mid] if len(diffs)%2==1 else 0.5*(diffs[mid-1]+diffs[mid])
        step_i = median_step2(list(Ks))
        radius = max(1, int(round(h / max(step_i, 1e-8))))
        AG_loc = np.zeros_like(AG_e)
        NG_loc = np.zeros_like(NG_e)
        for j in range(len(Ks)):
            l = max(0, j-radius); r = min(len(Ks)-1, j+radius)
            idx = np.arange(l, r+1)
            w = 1.0 - np.abs(idx - j) / radius
            w[w<0] = 0.0
            W = w.sum() if w.sum()>0 else 1.0
            AG_loc[j] = float((AG_e[idx] * w).sum() / W)
            NG_loc[j] = float((NG_e[idx] * w).sum() / W)
        Stab_e = AG_loc / (AG_loc + NG_loc + 1e-12)

        c = 4.0
        Vol_eff = np.log1p(c * vol_vec) / np.log1p(c)
        gamma_pow = 0.90
        OI_eff = (call_oi_vec + put_oi_vec) ** gamma_pow
        d_steps = np.abs(Ks - S) / max(step_i, 1e-8)
        d_star = 2.0
        wblend = np.clip(1.0 - d_steps / d_star, 0.0, 1.0)
        Act_raw = wblend * Vol_eff + (1.0 - wblend) * OI_eff

        def norm01(x):
            x = np.array(x, dtype=float)
            xmax = np.nanmax(x) if np.all(np.isfinite(x)) else 0.0
            return np.zeros_like(x) if xmax<=0 else (x / xmax)

        prep.append({
            "Ks": Ks,
            "AG_hat": norm01(AG_e),
            "Stab_hat": norm01(Stab_e),
            "Act_hat": norm01(Act_raw),
            "AG_raw": AG_e,
            "dOI_vec": dOI_vec,
            "adj_vol": (vol_vec/vol_vec.max()) if vol_vec.max()>0 else np.zeros_like(vol_vec),
            "iv_vec": iv_vec,
            "T": s["T"]
        })

    pz_vals = []
    for K in strikes_eval:
        val = 0.0
        WdK = Wd_kernel(K, S, h)
        for w, c in zip(W_time, prep):
            j = int(np.argmin(np.abs(c["Ks"] - K)))
            val += w * WdK * c["AG_hat"][j] * c["Stab_hat"][j] * c["Act_hat"][j]
        pz_vals.append(val)
    pz_vals = np.array(pz_vals, dtype=float)
    pz_norm = (pz_vals / pz_vals.max()) if pz_vals.max()>0 else np.zeros_like(pz_vals)

    kappa, delta_par, eps_small = 0.25, 0.50, 1e-6
    def Stab_at_K(K):
        s_acc = 0.0; w_acc = 0.0
        for w, c in zip(W_time, prep):
            j = int(np.argmin(np.abs(c["Ks"] - K)))
            s_acc += w * c["Stab_hat"][j]; w_acc += w
        return (s_acc/w_acc) if w_acc>0 else 0.0

    def AG_loc(K):
        total = 0.0
        for c in prep:
            mask = np.abs(c["Ks"] - K) <= h
            if np.any(mask):
                total += float(np.sum(c["AG_raw"][mask]))
        return total

    def HF_at_K(K):
        HF = 0.0
        for w, c in zip(W_time, prep):
            Ks = c["Ks"]; iv = c["iv_vec"]; T_i = c["T"]
            d1_S = (np.log(S / Ks) + 0.5*(iv**2)*T_i) / (iv*np.sqrt(T_i))
            d1_K = (np.log(max(K,1e-8) / Ks) + 0.5*(iv**2)*T_i) / (iv*np.sqrt(T_i))
            delta_S = np.array([N(x) for x in d1_S])
            delta_K = np.array([N(x) for x in d1_K])
            dDelta = delta_K - delta_S
            Inv = c["dOI_vec"] + kappa * c["adj_vol"]
            HF_e = M_CONTRACT * float(np.sum(dDelta * Inv))
            HF += w * HF_e
        return HF

    pzfp_vals = []
    for i, K in enumerate(strikes_eval):
        stab = Stab_at_K(K)
        agloc = AG_loc(K)
        hf = abs(HF_at_K(K))
        val = Wd_kernel(K, S, h) * stab * ((agloc**delta_par) if agloc>0 else 0.0) / (eps_small + hf)
        pzfp_vals.append(val)
    pzfp_vals = np.array(pzfp_vals, dtype=float)
    pzfp_norm = (pzfp_vals / pzfp_vals.max()) if pzfp_vals.max()>0 else np.zeros_like(pzfp_vals)

    return pz_norm, pzfp_norm
