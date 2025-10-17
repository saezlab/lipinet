from __future__ import annotations

__all__ = ["__version__"]

# Expose package version that matches installed metadata
try:  # Python 3.8+
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version
except Exception:  # pragma: no cover - fallback if needed
    try:
        from importlib_metadata import PackageNotFoundError  # type: ignore
        from importlib_metadata import version as _pkg_version
    except Exception:  # Last resort
        PackageNotFoundError = Exception  # type: ignore

        def _pkg_version(_name):  # type: ignore
            return "0.0.0"


try:
    __version__ = _pkg_version("lipinet")
except PackageNotFoundError:  # when running from source without install
    __version__ = "0.0.0"

__author__ = "Macabe Daley"
