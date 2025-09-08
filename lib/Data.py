
# Data.py — central data store for computed metrics and provider payloads
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd

@dataclass
class ChainData:
    raw: Any = None                 # raw provider JSON (normalized structure)
    raw_bytes: bytes = b""          # original response bytes for download
    quote: Dict[str, Any] = field(default_factory=dict)
    t0: Optional[int] = None
    S: Optional[float] = None
    expirations: List[int] = field(default_factory=list)
    blocks_by_date: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    selected_expiry: Optional[int] = None

@dataclass
class MetricsData:
    strikes: List[float] = field(default_factory=list)
    put_oi: List[int] = field(default_factory=list)
    call_oi: List[int] = field(default_factory=list)
    put_vol: List[int] = field(default_factory=list)
    call_vol: List[int] = field(default_factory=list)
    net_gex: List[float] = field(default_factory=list)
    ag: List[float] = field(default_factory=list)
    pz: List[float] = field(default_factory=list)
    pz_fp: List[float] = field(default_factory=list)
    g_flip: Optional[float] = None

@dataclass
class IntradayData:
    candles_df: Optional[pd.DataFrame] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None

class _Store:
    def __init__(self):
        self.chain = ChainData()
        self.metrics = MetricsData()
        self.intraday = IntradayData()
        self.debug_meta: Dict[str, Any] = {}

    # Chain
    def set_chain(self, raw, raw_bytes, quote, t0, S, expirations, blocks_by_date):
        self.chain = ChainData(raw=raw, raw_bytes=raw_bytes, quote=quote, t0=t0, S=S,
                               expirations=list(expirations or []),
                               blocks_by_date=dict(blocks_by_date or {}))

    def select_expiry(self, expiry_unix: Optional[int]):
        self.chain.selected_expiry = expiry_unix

    # Metrics
    def set_metrics(self, d: Dict[str, Any], g_flip: Optional[float] = None):
        self.metrics = MetricsData(
            strikes=list(map(float, d.get('strikes', []))),
            put_oi=list(map(int, d.get('put_oi', []))),
            call_oi=list(map(int, d.get('call_oi', []))),
            put_vol=list(map(int, d.get('put_vol', []))),
            call_vol=list(map(int, d.get('call_vol', []))),
            net_gex=list(map(float, d.get('net_gex', []))),
            ag=list(map(float, d.get('ag', []))),
            pz=list(map(float, d.get('pz', []))),
            pz_fp=list(map(float, d.get('pz_fp', []))),
            g_flip=g_flip
        )

    # Intraday
    def set_intraday(self, df, day_high=None, day_low=None):
        self.intraday = IntradayData(candles_df=df, day_high=day_high, day_low=day_low)

    # Debug
    def set_debug(self, meta: Dict[str, Any]):
        self.debug_meta = dict(meta or {})

DATA = _Store()
