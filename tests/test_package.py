import importlib
import sys

import lipinet


def test_package_version_fallback(monkeypatch):
    # Force importlib.metadata.version to raise PackageNotFoundError on reload
    try:
        import importlib.metadata as im
    except Exception:
        import importlib_metadata as im  # type: ignore

    monkeypatch.setattr(
        im,
        "version",
        lambda name: (_ for _ in ()).throw(im.PackageNotFoundError()),
        raising=False,
    )
    # Remove module to hit import-time branch
    sys.modules.pop("lipinet", None)
    mod = importlib.import_module("lipinet")
    assert mod.__version__ == "0.0.0"


def test_package_import_and_version_attr():
    assert hasattr(lipinet, "__version__")
    assert isinstance(lipinet.__version__, str)
