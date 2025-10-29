from .bootstrap import setup_environment


def run_app() -> None:
    from .ui import run_app as _run_app

    _run_app()


__all__ = ["setup_environment", "run_app"]
