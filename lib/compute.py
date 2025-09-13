import math, numpy as np
from math import log, sqrt
from datetime import datetime, time as _dt_time
from dateutil import tz as _tz

def _market_phase_now_et():
    """Return 'pre', 'rth', or 'post' based on America/New_York time (Mon–Fri)."""
    try:
        tz = _tz.gettz('America/New_York')
        now_et = datetime.now(tz=tz)
        if now_et.weekday() >= 5:
            return 'rth'  # treat weekend as regular to avoid misleading pre/post
        t = now_et.time()
        if t < _dt_time(9, 30):
            return 'pre'
        if t >= _dt_time(16, 0):
            return 'post'
        return 'rth'
    except Exception:
        return 'rth'


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

def _first_nonempty_dict(*candidates):
    for c in candidates:
        if isinstance(c, dict) and len(c)>0:
            return c
    return {}

def _extract_root_candidates(chain_json):
    """Return list of possible roots containing (quote, expirationDates, options)."""
    roots = []
    # Yahoo canonical
    try:
        res = chain_json.get("optionChain", {}).get("result", [])
        if isinstance(res, list) and len(res)>0 and isinstance(res[0], dict):
            roots.append(res[0])
    except Exception:
        pass
    # Provider v1: { meta, body: [...] } or { meta, body: {...} }
    body = chain_json.get("body", None)
    if isinstance(body, list) and len(body)>0 and isinstance(body[0], dict):
        roots.append(body[0])
    elif isinstance(body, dict):
        roots.append(body)
    # Some providers use 'data'
    data = chain_json.get("data", None)
    if isinstance(data, list) and len(data)>0 and isinstance(data[0], dict):
        roots.append(data[0])
    elif isinstance(data, dict):
        roots.append(data)
    # Some use 'result' directly
    res2 = chain_json.get("result", None)
    if isinstance(res2, list) and len(res2)>0 and isinstance(res2[0], dict):
        roots.append(res2[0])
    elif isinstance(res2, dict):
        roots.append(res2)
    # As a last resort try the object itself
    if isinstance(chain_json, dict):
        roots.append(chain_json)
    # Deduplicate by id()
    uniq = []
    seen = set()
    for r in roots:
        if id(r) in seen: continue
        seen.add(id(r)); uniq.append(r)
    return uniq

def _coerce_int(x, default=0):
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return default

def extract_core_from_chain(chain_json):
    """Unified extractor for multiple formats.
    Returns: (quote, t0, S, expirations: list[int], blocks_by_date: dict[int -> block]).
    Raises ValueError with a clear message if nothing matches.
    """
    candidates = _extract_root_candidates(chain_json)
    last_err = ""
    for root in candidates:
        try:
            quote = _first_nonempty_dict(root.get("quote", {}), root.get("underlying", {}))
            # robust price/time
            S = None
            phase = _market_phase_now_et()
            if phase == 'pre':
                priority = ('preMarketPrice','regularMarketPrice','postMarketPrice','last','lastPrice','price','close','regularMarketPreviousClose')
            elif phase == 'post':
                priority = ('postMarketPrice','regularMarketPrice','last','lastPrice','price','close','regularMarketPreviousClose')
            else:
                priority = ('regularMarketPrice','last','lastPrice','price','close','regularMarketPreviousClose')
            for key in priority:
                if key in quote and quote[key] is not None:
                    try:
                        S = float(quote[key]); break
                    except Exception:
                        pass
            if S is None:
                # some responses put price in root
                for key in priority:
                    if key in root and root[key] is not None:
                        try:
                            S = float(root[key]); break
                        except Exception:
                            pass
            t0 = 0
            for key in ("regularMarketTime","postMarketTime","time","timestamp","lastTradeDate" ):
                if key in quote and quote[key] is not None:
                    t0 = _coerce_int(quote[key]); break

            expirations = root.get("expirationDates") or root.get("expirations") or []
            # options container may be under 'options' or directly with calls/puts/straddles
            options = root.get("options", None)
            if options is None and ("calls" in root or "puts" in root or "straddles" in root):
                # wrap single block as list
                options = [root]
            if not isinstance(options, list):
                raise KeyError("no options list")
            blocks_by_date = {}
            for blk in options:
                if not isinstance(blk, dict): continue
                e = blk.get("expirationDate") or blk.get("expiry") or blk.get("expiryDate") or blk.get("expiration") or blk.get("expDate")
                if e is None:
                    # some responses put expiry only in meta; fallback to 0-index sentinel
                    continue
                e = _coerce_int(e)
                blocks_by_date[e] = blk
            if blocks_by_date:
                return quote, t0, float(S), list(map(int, expirations)), blocks_by_date
        except Exception as ex:
            last_err = str(ex)
            continue
    raise ValueError("Invalid chain JSON: could not find a compatible root (last error: %s)" % last_err)

def _calls_puts_from_straddles(straddles):
    calls, puts = [], []
    for s in straddles or []:
        c = s.get("call"); p = s.get("put")
        if isinstance(c, dict): calls.append(c)
        if isinstance(p, dict): puts.append(p)
    return calls, puts

def aggregate_series(block):
    calls = block.get("calls"); puts = block.get("puts")
    if calls is None and puts is None and "straddles" in block:
        calls, puts = _calls_puts_from_straddles(block.get("straddles", []))
    if calls is None: calls = []
    if puts is None:  puts  = []

    strikes = sorted(set([float(r.get("strike")) for r in (calls + puts) if r.get("strike") is not None]))
    call_oi = {float(r["strike"]): int(r.get("openInterest") or 0) for r in calls if r.get("strike") is not None}
    put_oi  = {float(r["strike"]): int(r.get("openInterest") or 0) for r in puts  if r.get("strike") is not None} # FIXED (restored condition)t("strike") is not None}
    call_vol= {float(r["strike"]): int(r.get("volume") or 0) if r.get("volume") is not None else 0 for r in calls if r.get("strike") is not None}
    put_vol = {float(r["strike"]): int(r.get("volume") or 0) if r.get("volume") is not None else 0 for r in puts  if r.get("strike") is not None}
    iv_call = {}
    iv_put  = {}
    for r in calls:
        k = r.get("strike"); iv = r.get("impliedVolatility", None)
        if k is None: continue
        k = float(k)
        if iv is not None and (not isinstance(iv, float) or not math.isnan(iv)) and float(iv)>0:
            iv_call[k] = float(iv)
    for r in puts:
        k = r.get("strike"); iv = r.get("impliedVolatility", None)
        if k is None: continue
        k = float(k)
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
    """
    Compute per-strike metrics for a given expiry using real greeks if available.

    This function aggregates open interest and volume as before but replaces the
    approximate Net GEX/AG formulas with sums of per-option gamma exposures.

    The returned dict contains arrays for strikes, put_oi, call_oi, put_vol,
    call_vol, net_gex (dollar gamma with sign), ag (absolute dollar gamma),
    and raw gamma shares (without the dollar conversion) which may be useful for
    downstream computations.
    """
    import numpy as np

    # Aggregate vanilla OI/volume/IV by strike
    strikes, call_oi, put_oi, call_vol, put_vol, iv_call, iv_put = aggregate_series(block)

    # Time to expiry in years (minimum fallback 1/252)
    DT_YEAR = 365.0 * 24 * 3600.0
    T_raw = (expiry_unix - t0) / DT_YEAR
    T = T_raw if T_raw > 0 else (1.0/252.0)

    # Prepare containers for gamma exposures per strike
    gamma_abs = {k: 0.0 for k in strikes}
    gamma_net = {k: 0.0 for k in strikes}

    # Helper to compute Black–Scholes gamma when missing
    def _bs_gamma_local(S_, K_, sigma_, T_, r_=0.0):
        # local copy avoids capturing outer scope bs_gamma due to recursion
        return bs_gamma(S_, K_, sigma_, T_, r_)

    # Iterate over calls and puts to accumulate exposures.  Use real gamma if provided,
    # otherwise compute via Black–Scholes using robust IV. Sign of delta is
    # approximated by contract type (positive for calls, negative for puts).
    for kind, lst in (("call", block.get("calls", [])), ("put", block.get("puts", []))):
        for opt in lst or []:
            try:
                k_val = opt.get("strike")
                if k_val is None:
                    continue
                k = float(k_val)
            except Exception:
                continue
            # Open interest and contract multiplier
            try:
                oi = int(opt.get("openInterest") or 0)
            except Exception:
                oi = 0
            try:
                m = int(opt.get("shares_per_contract") or 0)
                if m <= 0:
                    m = int(M_CONTRACT)
            except Exception:
                m = int(M_CONTRACT)
            # Skip if no OI
            if oi <= 0:
                continue
            # Extract gamma if available
            g_raw = opt.get("gamma", None)
            gamma_j = None
            if g_raw is not None:
                try:
                    gamma_j = float(g_raw)
                except Exception:
                    gamma_j = None
            # If gamma missing or invalid, approximate via BS formula
            if gamma_j is None:
                # Determine sigma from IV dictionaries; fallback to 0.30
                sigma_j = None
                if kind == "call":
                    sigma_j = iv_call.get(k)
                else:
                    sigma_j = iv_put.get(k)
                if sigma_j is None or not isinstance(sigma_j, (int, float)) or sigma_j <= 0 or math.isnan(sigma_j):
                    # fallback to nearest IV if available
                    try:
                        sigma_j = (iv_call.get(k) or iv_put.get(k))
                    except Exception:
                        sigma_j = None
                if sigma_j is None or not isinstance(sigma_j, (int, float)) or sigma_j <= 0 or math.isnan(sigma_j):
                    sigma_j = 0.30
                gamma_j = _bs_gamma_local(float(S), float(k), float(sigma_j), float(T))
            # Use sign by contract type
            sign = 1.0 if kind == "call" else -1.0
            # Accumulate absolute and signed exposures (share gamma)
            # Gamma_j is per 1 underlying; multiply by OI and contract size
            exposure = gamma_j * float(oi) * float(m)
            gamma_abs[k] += exposure
            gamma_net[k] += exposure * sign
    # Convert to arrays in order of strikes
    gamma_abs_arr = np.array([gamma_abs[k] for k in strikes], dtype=float)
    gamma_net_arr = np.array([gamma_net[k] for k in strikes], dtype=float)

    # Dollar conversion: multiply by spot price and divide by 1000
    net_gex = np.round((float(S) * gamma_net_arr) / 1000.0, 1)
    ag      = np.round((float(S) * gamma_abs_arr) / 1000.0, 1)

    # Convert OI/volume arrays
    call_oi_arr = np.array([call_oi[k] for k in strikes], dtype=float)
    put_oi_arr  = np.array([put_oi[k]  for k in strikes], dtype=float)
    call_vol_arr= np.array([call_vol[k] for k in strikes], dtype=float)
    put_vol_arr = np.array([put_vol[k]  for k in strikes], dtype=float)

    # Placeholder for old PZ metrics to maintain compatibility; will be superseded
    if all_series is None:
        pz = np.zeros_like(net_gex)
        pz_fp = np.zeros_like(net_gex)
    else:
        # Compute dummy arrays; new power zone is calculated outside
        pz = np.zeros_like(net_gex)
        pz_fp = np.zeros_like(net_gex)

    return {
        "strikes": strikes,
        "put_oi": put_oi_arr.astype(int),
        "call_oi": call_oi_arr.astype(int),
        "put_vol": put_vol_arr.astype(int),
        "call_vol": call_vol_arr.astype(int),
        "net_gex": net_gex,
        "ag": ag,
        "gamma_abs_share": gamma_abs_arr,  # raw share gamma (without $), useful for weighting
        "gamma_net_share": gamma_net_arr,  # raw net share gamma
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



def compute_power_zone_v2(
    S: float,
    t0,
    strikes_eval,
    sigma_atm: float,
    day_high: float,
    day_low: float,
    all_series,
    *,
    lambda_flow: float = 1.0,
    lambda_unstab: float = 0.5,
    lambda_mass: float = 1.0,
    minutes_to_close: float | None = None,
    bandwidth_dist: float = 15.0,   # в "числе страйков"
    n_candidates_max: int = 50,
):
    """
    Power Zone v2 (путь-зависимый потенциал).
    Возвращает словарь с полями:
      - 'K_star': лучший страйк-магнит (argmin E(K))
      - 'candidates': список (K, E(K)) отсортированный по возрастанию E
      - 'E': np.ndarray энергий вдоль strikes_eval
      - 'mass': np.ndarray масс (0..1) вдоль strikes_eval
      - 'flow_cost': np.ndarray локальной «стоимости потока» вдоль strikes_eval (>=0)
      - 'stab': np.ndarray стабильности (0..1) вдоль strikes_eval
      - 'conf': float, относительная уверенность (0..1)
    Ничего не трогает в существующей логике; использует те же базовые конструкции (AG/NG/HF), что и compute_pz_and_flow.
    """
    import numpy as _np

    strikes_eval = _np.array(list(strikes_eval), dtype=float)
    n_eval = len(strikes_eval)
    if n_eval == 0 or all_series is None or len(all_series) == 0:
        return {
            "K_star": None, "candidates": [], "E": _np.zeros(0),
            "mass": _np.zeros(0), "flow_cost": _np.zeros(0),
            "stab": _np.zeros(0), "conf": 0.0
        }

    # --- Веса по экспирациям (как в compute_pz_and_flow) ---
    tau = 0.5
    eta, zeta = 0.60, 0.40
    total_oi = _np.array([sum(s["call_oi"].values()) + sum(s["put_oi"].values()) for s in all_series], dtype=float)
    T_arr = _np.array([float(s["T"]) for s in all_series], dtype=float)
    W_time = total_oi * (T_arr**(-eta)) * ((1.0 - tau)**zeta)
    W_time = (W_time / W_time.sum()) if W_time.sum() > 0 else _np.ones_like(W_time)/len(W_time)

    # --- Ширина ядра по расстоянию (как в compute_pz_and_flow) ---
    if isinstance(day_high, (int,float)) and isinstance(day_low, (int,float)):
        R = float(day_high) - float(day_low)
    else:
        R = 0.01*float(S)
    # шаг по страйкам в eval-сетке
    def _median_step(xs):
        if len(xs) < 2: return 1.0
        diffs = sorted([abs(xs[i+1]-xs[i]) for i in range(len(xs)-1)])
        mid = len(diffs)//2
        return diffs[mid] if len(diffs)%2==1 else 0.5*(diffs[mid-1]+diffs[mid])

    step_eval = _median_step(list(strikes_eval))
    # h как в существующем коде: связка sigma_atm и дневного хода, c ограничениями
    h_base = max(1e-8, float(S) * max(float(sigma_atm or 0.0), 0.05) / math.sqrt(252.0))
    h = max(0.25*step_eval, min(0.025*float(S), max(h_base, R/6.0)))

    # --- Подготовка по сериям: AG_loc, NG_loc, Stab, Vol, dOI, iv ---
    prep = []
    for s in all_series:
        Ks = _np.array(list(s["strikes"]), dtype=float)
        # IV по узлам (с восстановлением ближайшим валидным, если нужно)
        iv0 = s.get("iv_call", None); iv1 = s.get("iv_put", None)
        iv_dict = {}
        if isinstance(iv0, dict):
            for k,v in iv0.items():
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    iv_dict[float(k)] = float(v)
        if isinstance(iv1, dict):
            for k,v in iv1.items():
                vv = float(v) if (v is not None and not (isinstance(v, float) and math.isnan(v))) else None
                if vv is not None:
                    if float(k) in iv_dict:
                        iv_dict[float(k)] = 0.5*(iv_dict[float(k)] + vv)
                    else:
                        iv_dict[float(k)] = vv
        iv_dict = robust_fill_iv(iv_dict, Ks, fallback=0.30)
        iv_vec = _np.array([iv_dict.get(float(k), 0.30) for k in Ks], dtype=float)

        call_oi_vec = _np.array([s["call_oi"][k] for k in s["strikes"]], dtype=float)
        put_oi_vec  = _np.array([s["put_oi"][k]  for k in s["strikes"]], dtype=float)
        vol_vec     = _np.array([s["call_vol"][k] + s["put_vol"][k] for k in s["strikes"]], dtype=float)
        dOI_vec     = call_oi_vec - put_oi_vec

        # Гамма и денежный масштаб
        gamma_vec = _np.array([bs_gamma(S, float(k), float(sig), float(s["T"])) for k, sig in zip(Ks, iv_vec)], dtype=float)
        dollar_per_unit = float(S) * M_CONTRACT / 1000.0
        AG_e = gamma_vec * (call_oi_vec + put_oi_vec) * dollar_per_unit
        NG_e = gamma_vec * _np.abs(dOI_vec) * dollar_per_unit

        # Локальные сглаживания (треугольное окно ~ h)
        def _median_step2(xarr):
            if len(xarr) < 2: return 1.0
            diffs = sorted([abs(xarr[i+1]-xarr[i]) for i in range(len(xarr)-1)])
            mid = len(diffs)//2
            return diffs[mid] if len(diffs)%2==1 else 0.5*(diffs[mid-1]+diffs[mid])

        step_i = _median_step2(list(Ks))
        radius = max(1, int(round(h / max(step_i, 1e-8))))
        AG_loc = _np.zeros_like(AG_e)
        NG_loc = _np.zeros_like(NG_e)
        for j in range(len(Ks)):
            l = max(0, j-radius); r = min(len(Ks)-1, j+radius)
            idx = _np.arange(l, r+1)
            w = 1.0 - _np.abs(idx - j) / radius
            w[w<0] = 0.0
            W = w.sum() if w.sum()>0 else 1.0
            AG_loc[j] = float((AG_e[idx] * w).sum() / W)
            NG_loc[j] = float((NG_e[idx] * w).sum() / W)
        Stab_e = AG_loc / (AG_loc + NG_loc + 1e-12)

        # Активность: log-сжатие объёма + OI^0.90, смесь по расстоянию
        c = 4.0
        Vol_eff = _np.log1p(c * vol_vec) / _np.log1p(c) if c>0 else vol_vec.copy()
        OI_eff  = _np.power(_np.maximum(call_oi_vec + put_oi_vec, 0.0), 0.90)
        # Бленд ближе/дальше от цены
        # расстояние в «числе шагов» (грубая аппроксимация через индексы)
        atm_idx = int(_np.argmin(_np.abs(Ks - float(S))))
        idxs = _np.arange(len(Ks))
        dist_steps = _np.abs(idxs - atm_idx)
        d_star = 2.0
        w_blend = _np.clip(1.0 - dist_steps / d_star, 0.0, 1.0)
        Act_raw = w_blend * Vol_eff + (1.0 - w_blend) * OI_eff

        # Нормировки [0,1] по серии
        def _norm01(arr):
            arr = _np.asarray(arr, dtype=float)
            mx = float(arr.max()) if arr.size else 0.0
            return (arr / mx) if mx > 0 else _np.zeros_like(arr)

        AG_hat   = _norm01(AG_loc)
        Stab_hat = _norm01(Stab_e)
        Act_hat  = _norm01(Act_raw)

        prep.append({
            "Ks": Ks, "AG_hat": AG_hat, "Stab_hat": Stab_hat, "Act_hat": Act_hat,
            "iv_vec": iv_vec, "T": float(s["T"]), "dOI_vec": dOI_vec, "vol_vec": vol_vec
        })

    # --- Сборка mass(x), stab(x) на strikes_eval ---
    mass_eval = _np.zeros(n_eval, dtype=float)
    stab_eval = _np.zeros(n_eval, dtype=float)
    for w, c in zip(W_time, prep):
        Ks = c["Ks"]
        # индексы ближайших узлов для каждого K_eval
        idx_near = _np.searchsorted(Ks, strikes_eval).clip(1, len(Ks)-1)
        left = idx_near - 1
        right = idx_near
        # возьмем ближайший по расстоянию (без интерполяции, робастно к дыркам сетки)
        pick = _np.where(_np.abs(Ks[left]-strikes_eval) <= _np.abs(Ks[right]-strikes_eval), left, right)
        mass_eval += w * (c["AG_hat"][pick] * c["Stab_hat"][pick] * c["Act_hat"][pick])
        stab_eval += w * (c["Stab_hat"][pick])

    # Нормировка mass в [0,1] (robust)
    mass_max = float(mass_eval.max()) if n_eval>0 else 0.0
    mass_eval = (mass_eval / mass_max) if mass_max > 0 else _np.zeros_like(mass_eval)
    stab_eval = _np.clip(stab_eval, 0.0, 1.0)

    # --- HF(S -> K) для всех K_eval и flow_cost(x) как разностная производная ---
    def HF_S_to_K(K):
        # идентично compute_pz_and_flow.HF_at_K, но более векторно и безопасно
        HF = 0.0
        kappa = 0.25
        for w, c in zip(W_time, prep):
            Ks = c["Ks"]; iv = c["iv_vec"]; T_i = c["T"]
            # Дельта при S и при K для всех узлов Ks
            d1_S = (np.log(float(S) / Ks) + 0.5*(iv**2)*T_i) / (iv*np.sqrt(T_i))
            d1_K = (np.log(max(float(K),1e-8) / Ks) + 0.5*(iv**2)*T_i) / (iv*np.sqrt(T_i))
            delta_S = _np.array([N(x) for x in d1_S])
            delta_K = _np.array([N(x) for x in d1_K])
            dDelta = delta_K - delta_S
            # инвентарь
            # объём нормируем на свой максимум, чтобы не взрывался
            vol = _np.asarray(c["vol_vec"], dtype=float)
            vol_norm = (vol / float(vol.max())) if vol.max() > 0 else _np.zeros_like(vol)
            Inv = c["dOI_vec"] + kappa * vol_norm
            HF_e = M_CONTRACT * float(_np.sum(dDelta * Inv))
            HF += w * abs(HF_e)
        return float(HF)

    HF_vals = _np.array([HF_S_to_K(float(K)) for K in strikes_eval], dtype=float)
    # flow_cost как "локальная трудность двигаться дальше": производная HF по K (в абсолюте)
    # дискретная производная (двусторонняя), в единицах на шаг страйков
    flow_cost = _np.zeros_like(HF_vals)
    if n_eval >= 3:
        dx = _np.maximum(_np.diff(strikes_eval), 1e-8)
        # центральные
        for i in range(1, n_eval-1):
            dxl = max(strikes_eval[i] - strikes_eval[i-1], 1e-8)
            dxr = max(strikes_eval[i+1] - strikes_eval[i], 1e-8)
            flow_cost[i] = 0.5*((HF_vals[i+1]-HF_vals[i])/dxr + (HF_vals[i]-HF_vals[i-1])/dxl)
        # края
        flow_cost[0] = (HF_vals[1]-HF_vals[0]) / max(strikes_eval[1]-strikes_eval[0], 1e-8)
        flow_cost[-1]= (HF_vals[-1]-HF_vals[-2]) / max(strikes_eval[-1]-strikes_eval[-2], 1e-8)
        flow_cost = _np.abs(flow_cost)
    else:
        flow_cost[:] = 0.0

    # --- Веса по времени до закрытия (простая форма) ---
    if minutes_to_close is None:
        w_tod = 1.0
    else:
        m0 = 90.0
        w_tod = ((minutes_to_close + m0) / m0) ** (-0.75)

    # --- Вес по расстоянию (гауссов) ---
    # оценим расстояние в "числе страйков": через индекс ATM в strikes_eval
    atm_idx_eval = int(_np.argmin(_np.abs(strikes_eval - float(S))))
    dist_idx = _np.abs(_np.arange(n_eval) - atm_idx_eval)
    # bandwidth_dist по умолчанию ~15 страйков
    w_dist = _np.exp(-(dist_idx / max(1.0, float(bandwidth_dist)))**2)

    # --- Интегральная энергия пути E(K) влево/вправо ---
    # подготовим «плотность энергии» в точке x:
    density = lambda_flow*flow_cost + lambda_unstab*(1.0 - stab_eval) - lambda_mass*(mass_eval * w_dist)
    density = w_tod * density

    # интегрируем кумулятивно слева и справа от ATM
    E = _np.zeros(n_eval, dtype=float)
    # влево (i < atm)
    for i in range(atm_idx_eval-1, -1, -1):
        dx = max(strikes_eval[i+1] - strikes_eval[i], 1e-8)
        E[i] = E[i+1] + 0.5*(density[i] + density[i+1]) * dx
    # вправо (i > atm)
    for i in range(atm_idx_eval+1, n_eval):
        dx = max(strikes_eval[i] - strikes_eval[i-1], 1e-8)
        E[i] = E[i-1] + 0.5*(density[i] + density[i-1]) * dx

    # --- Кандидаты: локальные максимумы mass или пилоны AG ---
    # Простая робастная детекция локальных максимумов mass
    cand_idx = []
    for i in range(1, n_eval-1):
        if mass_eval[i] >= mass_eval[i-1] and mass_eval[i] >= mass_eval[i+1]:
            cand_idx.append(i)
    if len(cand_idx) == 0:
        # fallback: ближайшие к ATM несколько точек
        around = list(range(max(0, atm_idx_eval-10), min(n_eval, atm_idx_eval+11)))
        cand_idx = around

    # ограничим количество кандидатов
    if len(cand_idx) > n_candidates_max:
        # возьмём топ по величине mass
        cand_idx = sorted(cand_idx, key=lambda i: mass_eval[i], reverse=True)[:n_candidates_max]

    # выбор
    E_cands = [(float(strikes_eval[i]), float(E[i])) for i in cand_idx]
    E_cands.sort(key=lambda x: x[1])
    K_star = E_cands[0][0] if len(E_cands)>0 else None

    # уверенность: разрыв между лучшим и вторым
    if len(E_cands) >= 2:
        e1, e2 = E_cands[0][1], E_cands[1][1]
        denom = abs(e1) + abs(e2) + 1e-8
        conf = max(0.0, min(1.0, (e2 - e1) / denom))
    else:
        conf = 0.0

    return {
        "K_star": K_star,
        "candidates": E_cands,
        "E": E,
        "mass": mass_eval,
        "flow_cost": flow_cost,
        "stab": stab_eval,
        "conf": conf,
    }


# ----------------------------------------------------------------------
# New Power Zone & Easy Reach computation using real gamma exposures
def compute_power_zone_and_er(
    S: float,
    strikes_eval,
    all_series_ctx,
    day_high: float | None = None,
    day_low: float | None = None,
    *,
    beta0: float = 1.0,
    eta: float = 0.60,
    zeta: float = 0.40,
    alpha_g: float = 1.0,
    alpha_v: float = 1.0,
    theta_short_gamma: float = 0.5,
    c_vol_log: float = 4.0,
    d_star: float = 2.0,
    eps: float = 1e-6,
):
    """
    Compute a normalized Power Zone (mass profile) and Easy Reach Level profiles
    across a unified strike grid, leveraging real gamma exposures from the
    option chain. The algorithm is inspired by PZ_local v2.2 methodology
    described in the documentation but adapted for direct gamma data.

    Parameters
    ----------
    S : float
        Spot price of the underlying asset.
    strikes_eval : array-like
        Strikes at which to evaluate the profiles (must be sorted).
    all_series_ctx : list of dict
        Context for each expiry, containing at least:
          - 'strikes': list of strike values (sorted ascending);
          - 'gamma_abs_share': array-like of absolute gamma share exposures (per strike);
          - 'gamma_net_share': array-like of signed gamma share exposures (per strike);
          - 'call_oi', 'put_oi': dicts of open interest by strike;
          - 'call_vol', 'put_vol': dicts of trade volume by strike;
          - 'iv_call', 'iv_put': dicts of IV values (used only for bandwidth estimation);
          - 'T': time to expiry in years.
        Additional fields are ignored.
    day_high, day_low : float or None
        Intraday high and low prices. If both are numbers, they are used to
        adjust the kernel bandwidth; otherwise a default range of 1% of S is used.
    beta0, eta, zeta, alpha_g, alpha_v, theta_short_gamma, c_vol_log, d_star, eps : float
        Hyperparameters controlling weighting, kernel width and functional forms.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A triple (pz, er_up, er_down) where each element is an array of
        length len(strikes_eval) normalized to [0,1], representing the power
        zone profile, ease-to-move-up profile, and ease-to-move-down profile.
    """
    import numpy as _np
    # Convert evaluation grid to numpy array
    strikes_eval = _np.asarray(list(strikes_eval), dtype=float)
    n_eval = len(strikes_eval)
    if n_eval == 0 or not all_series_ctx:
        return (_np.zeros(0, dtype=float), _np.zeros(0, dtype=float), _np.zeros(0, dtype=float))

    # --- Weight across expiries (time weighting) ---
    # Use same form as PZ_local: weight ~ total_OI * T^{-eta} * (1-τ)^ζ
    tau = 0.5  # assume mid-session if real time not provided
    total_oi = _np.array([
        float(sum((s.get("call_oi") or {}).values()) + sum((s.get("put_oi") or {}).values()))
        for s in all_series_ctx
    ], dtype=float)
    T_arr = _np.array([float(s.get("T", 0.0)) for s in all_series_ctx], dtype=float)
    # Compute weight, avoid division by zero
    W_time = total_oi * (T_arr**(-eta)) * ((1.0 - tau)**zeta)
    W_time = (W_time / W_time.sum()) if W_time.sum() > 0 else _np.ones_like(W_time) / len(W_time)

    # --- Kernel bandwidth (h) determination ---
    # Estimate aggregate sigma from available IV data; fallback to 0.25
    iv_vals = []
    for s in all_series_ctx:
        for iv_dict in (s.get("iv_call"), s.get("iv_put")):
            if isinstance(iv_dict, dict):
                for v in iv_dict.values():
                    try:
                        vv = float(v)
                        if vv > 0 and not math.isnan(vv):
                            iv_vals.append(vv)
                    except Exception:
                        pass
    sigma_est = float(_np.median(iv_vals)) if iv_vals else 0.25
    # Intraday range
    if isinstance(day_high, (int, float)) and isinstance(day_low, (int, float)):
        R = float(day_high) - float(day_low)
    else:
        R = 0.01 * float(S)
    # Determine median step among evaluation strikes
    def _median_step(arr):
        if len(arr) < 2:
            return 1.0
        diffs = sorted([abs(arr[i+1] - arr[i]) for i in range(len(arr)-1)])
        mid = len(diffs) // 2
        return diffs[mid] if len(diffs) % 2 == 1 else 0.5 * (diffs[mid-1] + diffs[mid])
    step_eval = _median_step(strikes_eval.tolist())
    # Base width proportional to sigma_est
    h_base = float(S) * max(float(sigma_est), 0.05) / math.sqrt(252.0)
    # Final bandwidth: limit by intraday range and evaluation grid
    h = max(0.25 * step_eval, min(0.025 * float(S), max(h_base, R/6.0)))

    # --- Preprocess each series: compute smoothed AG, stability and activity ---
    prep = []
    for s in all_series_ctx:
        Ks = _np.asarray(s.get("strikes") or [], dtype=float)
        if Ks.size == 0:
            continue
        # Raw gamma exposures per strike (share).  The context may store these
        # either as arrays aligned with Ks or as dicts keyed by strike.
        g_raw_abs = s.get("gamma_abs_share")
        g_raw_net = s.get("gamma_net_share")
        if isinstance(g_raw_abs, dict):
            # Build array by indexing dict with each strike
            g_abs = _np.array([g_raw_abs.get(float(k), 0.0) for k in Ks], dtype=float)
        elif g_raw_abs is None:
            g_abs = _np.zeros_like(Ks, dtype=float)
        else:
            g_abs = _np.asarray(g_raw_abs, dtype=float)
        if isinstance(g_raw_net, dict):
            g_net = _np.array([g_raw_net.get(float(k), 0.0) for k in Ks], dtype=float)
        elif g_raw_net is None:
            g_net = _np.zeros_like(Ks, dtype=float)
        else:
            g_net = _np.asarray(g_raw_net, dtype=float)
        # Convert to dollar exposures
                # Fallback: if both raw gamma profiles are degenerate (all zeros), synthesize share profiles
        # using BS gamma and activity weights (prefer OI, fallback to volume). This does NOT affect
        # dollar Net GEX / AG elsewhere — only the internal mass/stability proxy here.
        if (g_abs.size and _np.all(g_abs == 0.0)) and (g_net.size and _np.all(g_net == 0.0)):
            call_oi_dict = s.get("call_oi", {}) or {}
            put_oi_dict  = s.get("put_oi", {})  or {}
            call_vol_dict= s.get("call_vol", {}) or {}
            put_vol_dict = s.get("put_vol", {})  or {}
            iv_call_dict = s.get("iv_call", {}) or {}
            iv_put_dict  = s.get("iv_put", {})  or {}
            T_e = float(s.get("T") or 0.0)
            def _sigma_for(kv):
                k = float(kv)
                ivc = iv_call_dict.get(k, None); ivp = iv_put_dict.get(k, None)
                if ivc and ivc > 0: return float(ivc)
                if ivp and ivp > 0: return float(ivp)
                return 0.30
            g_abs_syn = _np.zeros_like(g_abs, dtype=float)
            g_net_syn = _np.zeros_like(g_net, dtype=float)
            for j, kk in enumerate(Ks):
                sigma_j = _sigma_for(kk)
                try:
                    g_bs = bs_gamma(float(S), float(kk), float(sigma_j), float(T_e))
                except Exception:
                    g_bs = 0.0
                wc = float(call_oi_dict.get(kk, 0.0)); wp = float(put_oi_dict.get(kk, 0.0))
                if wc == 0.0 and wp == 0.0:
                    wc = float(call_vol_dict.get(kk, 0.0)); wp = float(put_vol_dict.get(kk, 0.0))
                # absolute & net share proxies
                g_abs_syn[j] = g_bs * max(wc + wp, 0.0)
                g_net_syn[j] = g_bs * (wc - wp)
            g_abs = g_abs_syn
            g_net = g_net_syn
        AG_e = g_abs * float(S) / 1000.0
        NG_e = _np.abs(g_net) * float(S) / 1000.0
        # Local smoothing using triangular kernel based on bandwidth h and strike spacing
        def _median_step2(xarr):
            if len(xarr) < 2:
                return 1.0
            diffs = sorted([abs(xarr[i+1] - xarr[i]) for i in range(len(xarr)-1)])
            mid = len(diffs)//2
            return diffs[mid] if len(diffs) % 2 == 1 else 0.5*(diffs[mid-1] + diffs[mid])
        step_i = _median_step2(list(Ks))
        radius = max(1, int(round(h / max(step_i, 1e-8))))
        n_s = len(Ks)
        AG_loc = _np.zeros(n_s, dtype=float)
        NG_loc = _np.zeros(n_s, dtype=float)
        for j in range(n_s):
            l = max(0, j - radius); rj = min(n_s - 1, j + radius)
            idx = _np.arange(l, rj + 1)
            w = 1.0 - _np.abs(idx - j) / radius
            w[w < 0] = 0.0
            W = w.sum() if w.sum() > 0 else 1.0
            AG_loc[j] = float((_np.sum(AG_e[idx] * w)) / W)
            NG_loc[j] = float((_np.sum(NG_e[idx] * w)) / W)
        # Stability ratio
        Stab_e = AG_loc / (AG_loc + NG_loc + eps)
        # Activity metric combining volume and OI
        call_oi_dict = s.get("call_oi", {})
        put_oi_dict  = s.get("put_oi",  {})
        call_vol_dict= s.get("call_vol", {})
        put_vol_dict = s.get("put_vol",  {})
        # Arrays for OI and volume
        call_oi_vec = _np.array([call_oi_dict.get(k, 0) for k in Ks], dtype=float)
        put_oi_vec  = _np.array([put_oi_dict.get(k, 0) for k in Ks], dtype=float)
        vol_vec     = _np.array([call_vol_dict.get(k, 0) + put_vol_dict.get(k, 0) for k in Ks], dtype=float)
        # Log-compressed volume
        if c_vol_log > 0:
            Vol_eff = _np.log1p(c_vol_log * vol_vec) / _np.log1p(c_vol_log)
        else:
            Vol_eff = vol_vec.copy()
        # OI to fractional power
        OI_eff = _np.power(_np.maximum(call_oi_vec + put_oi_vec, 0.0), 0.90)
        # Blend near ATM vs far
        atm_idx = int(_np.argmin(_np.abs(Ks - float(S))))
        idxs = _np.arange(len(Ks), dtype=int)
        dist_steps = _np.abs(idxs - atm_idx)
        w_blend = _np.clip(1.0 - dist_steps / d_star, 0.0, 1.0)
        Act_raw = w_blend * Vol_eff + (1.0 - w_blend) * OI_eff
        # Normalize arrays to [0,1]
        def _norm01(arr):
            arr = _np.asarray(arr, dtype=float)
            mx = float(_np.nanmax(arr)) if arr.size else 0.0
            return (arr / mx) if mx > 0 else _np.zeros_like(arr)
        AG_hat = _norm01(AG_loc)
        Stab_hat = _norm01(Stab_e)
        Act_hat = _norm01(Act_raw)
        prep.append({
            "Ks": Ks,
            "AG_loc": AG_loc,
            "AG_hat": AG_hat,
            "Stab_hat": Stab_hat,
            "Act_hat": Act_hat,
            "vol_vec": vol_vec,
            "call_vol_dict": call_vol_dict,
            "put_vol_dict": put_vol_dict,
            "call_oi_dict": call_oi_dict,
            "put_oi_dict": put_oi_dict
        })
    # If no valid series after preprocessing, return zeros
    if not prep:
        return (_np.zeros(n_eval, dtype=float), _np.zeros(n_eval, dtype=float), _np.zeros(n_eval, dtype=float))

    # --- Mass profile (Power Zone) evaluation ---
    mass_vals = _np.zeros(n_eval, dtype=float)
    # Precompute kernel for evaluation grid
    dist_eval = (strikes_eval - float(S)) / float(h if h > 0 else 1.0)
    Wd_eval = _np.exp(-0.5 * (dist_eval**2))
    for w_e, c in zip(W_time, prep):
        Ks = c["Ks"]
        # For each K_eval pick nearest strike index in this series
        idx_near = _np.searchsorted(Ks, strikes_eval).clip(1, len(Ks)-1)
        left = idx_near - 1
        right = idx_near
        pick = _np.where(_np.abs(Ks[left] - strikes_eval) <= _np.abs(Ks[right] - strikes_eval), left, right)
        mass_vals += w_e * Wd_eval * (c["AG_hat"][pick] * c["Stab_hat"][pick] * c["Act_hat"][pick])
    # Normalize mass
    max_mass = float(_np.nanmax(mass_vals)) if n_eval > 0 else 0.0
    pz_norm = (mass_vals / max_mass) if max_mass > 0 else _np.zeros_like(mass_vals)

    # --- Build aggregated barriers and direction/volume ---
    G_vals = _np.zeros(n_eval, dtype=float)
    V_vals = _np.zeros(n_eval, dtype=float)
    D_vals = _np.zeros(n_eval, dtype=float)
    for w_e, c in zip(W_time, prep):
        Ks = c["Ks"]
        AG_loc = c["AG_loc"]
        # Nearest index for each K_eval
        idx_near = _np.searchsorted(Ks, strikes_eval).clip(1, len(Ks)-1)
        left = idx_near - 1
        right = idx_near
        pick = _np.where(_np.abs(Ks[left] - strikes_eval) <= _np.abs(Ks[right] - strikes_eval), left, right)
        # Sum weighted AG_loc
        G_vals += w_e * AG_loc[pick]
        # Sum weighted volume
        vol_vec = c["vol_vec"]
        V_vals += w_e * vol_vec[pick]
        # Direction: prefer volume diff; fallback to OI diff if volume absent
        diff_vec = []
        call_vol_dict = c["call_vol_dict"]
        put_vol_dict = c["put_vol_dict"]
        call_oi_dict = c["call_oi_dict"]
        put_oi_dict  = c["put_oi_dict"]
        for kk in strikes_eval:
            # nearest strike in this series
            i = int(_np.argmin(_np.abs(Ks - kk)))
            k0 = Ks[i]
            cv = call_vol_dict.get(k0, 0.0)
            pv = put_vol_dict.get(k0, 0.0)
            if cv == 0 and pv == 0:
                dv = call_oi_dict.get(k0, 0.0) - put_oi_dict.get(k0, 0.0)
            else:
                dv = cv - pv
            diff_vec.append(dv)
        D_vals += w_e * _np.asarray(diff_vec, dtype=float)
    # Normalize barriers and direction
    G_norm = _np.zeros_like(G_vals)
    V_norm = _np.zeros_like(V_vals)
    D_norm = _np.zeros_like(D_vals)
    g_max = float(_np.nanmax(G_vals)) if _np.any(_np.isfinite(G_vals)) else 0.0
    v_max = float(_np.nanmax(V_vals)) if _np.any(_np.isfinite(V_vals)) else 0.0
    d_max = float(_np.nanmax(_np.abs(D_vals))) if _np.any(_np.isfinite(D_vals)) else 0.0
    if g_max > 0:
        G_norm = G_vals / g_max
    if v_max > 0:
        V_norm = V_vals / v_max
    if d_max > 0:
        D_norm = D_vals / d_max
    # Determine gamma regime: long vs short gamma
    total_net_gamma = 0.0
    for s in all_series_ctx:
        gnet = s.get("gamma_net_share")
        if gnet is not None:
            try:
                total_net_gamma += float(_np.sum(_np.asarray(gnet, dtype=float)))
            except Exception:
                pass
    theta = theta_short_gamma if total_net_gamma < 0 else 0.0
    # Common denominator for ER
    denom = eps + (G_norm ** alpha_g) * (V_norm ** alpha_v)
    denom_safe = _np.where(denom > eps, denom, eps)
    # Compute up/down ease metrics
    er_up_vals = Wd_eval / denom_safe * (1.0 + theta * D_norm)
    er_down_vals = Wd_eval / denom_safe * (1.0 - theta * D_norm)
    # Normalize to [0,1]
    up_max = float(_np.nanmax(er_up_vals)) if _np.any(_np.isfinite(er_up_vals)) else 0.0
    down_max = float(_np.nanmax(er_down_vals)) if _np.any(_np.isfinite(er_down_vals)) else 0.0
    er_up_norm = (er_up_vals / up_max) if up_max > 0 else _np.zeros_like(er_up_vals)
    er_down_norm = (er_down_vals / down_max) if down_max > 0 else _np.zeros_like(er_down_vals)
    return (pz_norm, er_up_norm, er_down_norm)
