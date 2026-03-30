import sys
import types


# Unit tests here only exercise strike normalization helpers and do not require
# a live IB client. Provide a lightweight stub so module import succeeds.
if "ib_async" not in sys.modules:
    ib_async_stub = types.ModuleType("ib_async")

    class _Dummy:
        def __init__(self, *args, **kwargs):
            pass

    ib_async_stub.IB = _Dummy
    ib_async_stub.Index = _Dummy
    ib_async_stub.Option = _Dummy
    ib_async_stub.Stock = _Dummy
    sys.modules["ib_async"] = ib_async_stub

if "nest_asyncio" not in sys.modules:
    nest_asyncio_stub = types.ModuleType("nest_asyncio")
    nest_asyncio_stub.apply = lambda *args, **kwargs: None
    sys.modules["nest_asyncio"] = nest_asyncio_stub

from algotrader.data.ibkr_provider import IBKRDataProvider


def test_normalize_option_strikes_snaps_malformed_etf_strikes() -> None:
    raw = [559.78, 564.78, 569.78, 574.78]
    normalized = IBKRDataProvider._normalize_option_strikes("QQQ", raw)
    assert normalized == [560.0, 565.0, 570.0, 575.0]


def test_normalize_option_strikes_keeps_generic_underlying_unchanged() -> None:
    raw = [102.25, 100.0, 101.5]
    normalized = IBKRDataProvider._normalize_option_strikes("AAPL", raw)
    assert normalized == [100.0, 101.5, 102.25]
