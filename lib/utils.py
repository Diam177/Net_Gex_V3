import math, datetime, os

def choose_default_expiration(expirations, now_unix, *, tz_mode: str = "UTC"):
    """
    Choose nearest expiration by **calendar date**, not timestamp.
    Default mode aligns with UI labels which render dates via UTC
    (see app.py fmt_ts -> utcfromtimestamp).

    - tz_mode="UTC": compare by UTC dates (recommended).
    - tz_mode="ET":  compare by America/New_York dates (legacy option).
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    if not expirations:
        return None

    if tz_mode.upper() == "ET":
        tz = ZoneInfo("America/New_York")
        now_date = datetime.fromtimestamp(now_unix, tz).date()
        pairs = [(e, datetime.fromtimestamp(e, tz).date()) for e in expirations]
    else:
        now_date = datetime.utcfromtimestamp(now_unix).date()
        pairs = [(e, datetime.utcfromtimestamp(e).date()) for e in expirations]

    future = [e for e, d in pairs if d >= now_date]
    if future:
        # minimize by (date, timestamp) to keep stable order for same-day expirations
        if tz_mode.upper() == "ET":
            tz = ZoneInfo("America/New_York")
            key = lambda e: (datetime.fromtimestamp(e, tz).date(), e)
        else:
            key = lambda e: (datetime.utcfromtimestamp(e).date(), e)
        return min(future, key=key)

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
