import math, datetime, os

def choose_default_expiration(expirations_unix, now_unix):
    futures = [e for e in expirations_unix if e >= now_unix]
    if futures:
        return sorted(futures)[0]
    return sorted(expirations_unix, key=lambda e: abs(e - now_unix))[0]

def median_step(strikes):
    if len(strikes) < 2:
        return 1.0
    diffs = sorted([abs(strikes[i+1]-strikes[i]) for i in range(len(strikes)-1)])
    mid = len(diffs)//2
    return diffs[mid] if len(diffs)%2==1 else 0.5*(diffs[mid-1]+diffs[mid])

def robust_get(dct, key, default=0):
    try:
        v = dct.get(key, default)
        return default if v is None else v
    except Exception:
        return default

def env_or_secret(st, key, default=None):
    if st is not None:
        try:
            if key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass
    return os.environ.get(key, default)
