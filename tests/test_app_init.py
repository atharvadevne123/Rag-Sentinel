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
    from app.main import APP_VERSION
    from app import __version__
    assert APP_VERSION == __version__
