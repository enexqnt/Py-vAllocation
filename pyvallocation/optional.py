"""Optional pandas import utilities."""

from importlib import util

_spec = util.find_spec("pandas")
if _spec is not None:
    import pandas as pd  # type: ignore

    HAS_PANDAS = True
else:
    pd = None  # type: ignore
    HAS_PANDAS = False


def require_pandas(msg: str = "pandas is required for this functionality") -> None:
    """Raise :class:`ImportError` if pandas is not installed."""

    if not HAS_PANDAS:
        raise ImportError(msg)


__all__ = ["pd", "HAS_PANDAS", "require_pandas"]
