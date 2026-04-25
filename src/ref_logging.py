"""Configure logging for refsproj1 (console + optional rotating file under logs/)."""

from __future__ import annotations

import logging
import logging.config
import os
import sys
from pathlib import Path

_LOGGERS_QUIET = ("urllib3", "urllib3.connectionpool", "charset_normalizer")


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _config_file_has_ini_sections(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#") and s.startswith("[") and s.endswith("]"):
            return True
    return False


def _apply_console_level(level_name: str) -> None:
    lv = getattr(logging, level_name.upper(), None)
    if not isinstance(lv, int):
        return
    log = logging.getLogger("refsproj1")
    log.setLevel(logging.DEBUG)
    for h in log.handlers:
        if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) in (
            sys.stderr,
            sys.stdout,
        ):
            h.setLevel(lv)


def setup_logging(
    level: str = "INFO",
    *,
    enable_file: bool = True,
    log_dir: Path | None = None,
    config_path: Path | None = None,
) -> None:
    """
    Configure the ``refsproj1`` logger.

    If ``REFSPROJ_LOG_CONFIG`` is set, that path is used as the logging config file.
    Else if ``logging.conf`` in the project root contains INI sections, ``fileConfig`` loads it.
    Otherwise a built-in ``dictConfig`` is used (respects ``enable_file`` and ``level``).

    After ``fileConfig``, ``level`` still adjusts the **console** handler threshold.
    """
    root_dir = project_root()
    env_cfg = os.environ.get("REFSPROJ_LOG_CONFIG")
    cfg = (
        Path(env_cfg).expanduser()
        if env_cfg
        else (config_path if config_path is not None else root_dir / "logging.conf")
    )

    (root_dir / "logs").mkdir(parents=True, exist_ok=True)

    use_file = _config_file_has_ini_sections(cfg)
    old_cwd = os.getcwd()
    try:
        if use_file:
            os.chdir(root_dir)
            logging.config.fileConfig(cfg, disable_existing_loggers=False)
        else:
            log_directory = log_dir or root_dir / "logs"
            log_directory.mkdir(parents=True, exist_ok=True)
            log_file = log_directory / "app.log"

            resolved = getattr(logging, level.upper(), None)
            if not isinstance(resolved, int):
                resolved = logging.INFO

            handlers: dict[str, dict] = {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": level.upper(),
                    "formatter": "std",
                    "stream": "ext://sys.stderr",
                },
            }
            logger_handlers = ["console"]
            if enable_file:
                handlers["file"] = {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "DEBUG",
                    "formatter": "std",
                    "filename": str(log_file),
                    "maxBytes": 1_048_576,
                    "backupCount": 5,
                    "encoding": "utf-8",
                }
                logger_handlers.append("file")

            logging.config.dictConfig(
                {
                    "version": 1,
                    "disable_existing_loggers": False,
                    "formatters": {
                        "std": {
                            "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                            "datefmt": "%Y-%m-%d %H:%M:%S",
                        },
                    },
                    "handlers": handlers,
                    "loggers": {
                        "refsproj1": {
                            "level": "DEBUG",
                            "handlers": logger_handlers,
                            "propagate": False,
                        },
                    },
                }
            )
    finally:
        os.chdir(old_cwd)

    if use_file:
        _apply_console_level(level)

    for name in _LOGGERS_QUIET:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return ``refsproj1.<name>`` (e.g. ``get_logger("app")``)."""
    base = "refsproj1" if not name else f"refsproj1.{name}"
    return logging.getLogger(base)
