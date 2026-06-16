from __future__ import annotations


def test_app_version_exported():
    from app import __version__

    assert isinstance(__version__, str)
    parts = __version__.split(".")
    assert len(parts) == 3


def test_app_author_exported():
    from app import __author__

    assert isinstance(__author__, str)
    assert len(__author__) > 0


def test_rag_package_importable():
    import rag

    assert rag.__doc__ is not None


def test_pipelines_package_importable():
    import pipelines

    assert pipelines.__doc__ is not None


def test_version_is_semver():
    from app import __version__

    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_main_app_has_version():
    from app import __version__
    from app.main import APP_VERSION

    assert APP_VERSION == __version__


def test_rag_all_exports_importable():
    import rag

    for name in rag.__all__:
        assert hasattr(rag, name), f"rag.__all__ member {name!r} not importable"


def test_app_all_exports():
    import app

    assert "__version__" in app.__all__
    assert "__author__" in app.__all__


def test_author_is_not_empty():
    from app import __author__

    assert __author__.strip() != ""


def test_exceptions_module_importable():
    import app.exceptions as exc_mod

    assert hasattr(exc_mod, "RagSentinelError")
    assert hasattr(exc_mod, "ModelNotLoadedError")
