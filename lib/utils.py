import math, datetime, os

def choose_default_expiration(expirations, now_unix, tz_name: str = "America/New_York"):
    """
    Choose the nearest expiration by **calendar date** in a given timezone (default ET),
    not by exact timestamp. If today's date equals an expiration date in tz, we select it.
    Fallback: if all expirations are before today, return the latest one.
    """
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo  # stdlib (Py3.9+)
    except Exception:
        ZoneInfo = None

    if not expirations:
        return None

    # Convert 'now' to local date in tz
    if ZoneInfo is not None:
        tz = ZoneInfo(tz_name)
        now_date = datetime.fromtimestamp(now_unix, tz).date()
        exp_pairs = [(e, datetime.fromtimestamp(e, tz).date()) for e in expirations]
    else:
        # Fallback to naive UTC date comparison if zoneinfo not available
        now_date = datetime.utcfromtimestamp(now_unix).date()
        exp_pairs = [(e, datetime.utcfromtimestamp(e).date()) for e in expirations]

    # Keep expirations whose local date is today or later; pick the earliest by date
    future = [e for e, d in exp_pairs if d >= now_date]
    if future:
        # If multiple expirations share the same date, choose the smallest timestamp
        # to preserve original ordering.
        def key(e):
            if ZoneInfo is not None:
                tz = ZoneInfo(tz_name)
                return (datetime.fromtimestamp(e, tz).date(), e)
            else:
                return (datetime.utcfromtimestamp(e).date(), e)
        return min(future, key=key)

    # Fallback: all expirations are before 'today' in tz → return the latest available
    return max(expirations)


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
