"""Tests for logging configuration utilities."""

from __future__ import annotations

import logging

import pytest

import cosecha._logging as cosecha_logging
from cosecha._logging import (
    _validate_level,
    configure_logger,
    generate_log_path,
    get_log_file_path,
    logger,
)


@pytest.fixture(autouse=True)
def _restore_logger_state():
    """Save and restore the global logger state after each test."""
    original_handlers = list(logger.handlers)
    original_file_handler = cosecha_logging._file_handler
    original_console_handler = cosecha_logging._console_handler
    original_console_level = (
        original_console_handler.level if original_console_handler is not None else None
    )
    yield
    if (
        cosecha_logging._file_handler is not None
        and cosecha_logging._file_handler is not original_file_handler
    ):
        logger.removeHandler(cosecha_logging._file_handler)
        cosecha_logging._file_handler.close()
    logger.handlers = list(original_handlers)
    cosecha_logging._file_handler = original_file_handler
    cosecha_logging._console_handler = original_console_handler
    if original_console_handler is not None and original_console_level is not None:
        original_console_handler.setLevel(original_console_level)


class TestGenerateLogPath:
    """Tests for generate_log_path."""

    def test_returns_path_with_prefix_and_extension(self, tmp_path):
        """Test returned path has default prefix and .log extension."""
        result = generate_log_path(tmp_path)
        assert result.parent == tmp_path
        assert result.name.startswith("cosecha-")
        assert result.suffix == ".log"

    def test_custom_prefix(self, tmp_path):
        """Test returned path uses custom prefix."""
        result = generate_log_path(tmp_path, prefix="harvest")
        assert result.name.startswith("harvest-")
        assert result.suffix == ".log"

    def test_path_under_work_dir(self, tmp_path):
        """Test that the returned path is under the given work_dir."""
        subdir = tmp_path / "logs"
        subdir.mkdir()
        result = generate_log_path(subdir)
        assert str(result).startswith(str(subdir))


class TestGetLogFilePath:
    """Tests for get_log_file_path."""

    def test_returns_none_when_no_file_handler(self):
        """Test returns None when no file handler is active."""
        if cosecha_logging._file_handler is not None:
            logger.removeHandler(cosecha_logging._file_handler)
            cosecha_logging._file_handler.close()
            cosecha_logging._file_handler = None
        assert get_log_file_path() is None

    def test_returns_path_when_file_logging_enabled(self, tmp_path):
        """Test returns the file path when file logging has been configured."""
        log_file = tmp_path / "test.log"
        configure_logger(file=log_file)
        result = get_log_file_path()
        assert result is not None
        assert result.name == "test.log"


class TestValidateLevel:
    """Tests for _validate_level."""

    @pytest.mark.parametrize(
        ("level_str", "expected"),
        [
            ("DEBUG", logging.DEBUG),
            ("INFO", logging.INFO),
            ("WARNING", logging.WARNING),
            ("ERROR", logging.ERROR),
            ("CRITICAL", logging.CRITICAL),
        ],
    )
    def test_valid_string_levels(self, level_str, expected):
        """Test valid string levels return correct int."""
        assert _validate_level(level_str) == expected

    def test_case_insensitive_lowercase(self):
        """Test that lowercase string levels are accepted."""
        assert _validate_level("debug") == logging.DEBUG

    def test_case_insensitive_mixed(self):
        """Test that mixed-case string levels are accepted."""
        assert _validate_level("Info") == logging.INFO

    def test_invalid_string_raises_value_error(self):
        """Test invalid string level raises ValueError."""
        with pytest.raises(ValueError, match="Invalid log level"):
            _validate_level("TRACE")

    @pytest.mark.parametrize(
        "level_int",
        [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL],
    )
    def test_valid_int_levels(self, level_int):
        """Test valid int levels return the same int."""
        assert _validate_level(level_int) == level_int

    def test_invalid_int_raises_value_error(self):
        """Test invalid int level raises ValueError."""
        with pytest.raises(ValueError, match="Invalid log level"):
            _validate_level(999)


class TestConfigureLogger:
    """Tests for configure_logger."""

    def test_verbose_true_sets_debug(self):
        """Test verbose=True sets console handler to DEBUG."""
        configure_logger(verbose=True)
        assert cosecha_logging._console_handler is not None
        assert cosecha_logging._console_handler.level == logging.DEBUG

    def test_verbose_false_sets_warning(self):
        """Test verbose=False sets console handler to WARNING."""
        configure_logger(verbose=False)
        assert cosecha_logging._console_handler is not None
        assert cosecha_logging._console_handler.level == logging.WARNING

    def test_level_overrides_verbose(self):
        """Test explicit level parameter sets console handler level."""
        configure_logger(level="INFO")
        assert cosecha_logging._console_handler is not None
        assert cosecha_logging._console_handler.level == logging.INFO

    def test_file_enables_file_logging(self, tmp_path):
        """Test file parameter enables file logging and get_log_file_path returns it."""
        log_file = tmp_path / "app.log"
        configure_logger(file=log_file)
        assert cosecha_logging._file_handler is not None
        result = get_log_file_path()
        assert result is not None
        assert result.name == "app.log"

    def test_file_level(self, tmp_path):
        """Test file_level sets the file handler level."""
        log_file = tmp_path / "app.log"
        configure_logger(file=log_file, file_level="ERROR")
        assert cosecha_logging._file_handler is not None
        assert cosecha_logging._file_handler.level == logging.ERROR

    def test_file_none_after_enabling_disables(self, tmp_path):
        """Test that calling configure_logger without file after enabling removes file handler."""
        log_file = tmp_path / "app.log"
        configure_logger(file=log_file)
        assert cosecha_logging._file_handler is not None
        # Now call again without file arg -- the default file=None path removes the handler
        configure_logger()
        assert cosecha_logging._file_handler is None
        assert get_log_file_path() is None

    def test_file_only_removes_console_handler(self, tmp_path):
        """Test file_only=True removes the console handler from logger."""
        log_file = tmp_path / "app.log"
        assert cosecha_logging._console_handler in logger.handlers
        configure_logger(file=log_file, file_only=True)
        assert cosecha_logging._console_handler not in logger.handlers

    def test_file_replaces_previous_file_handler(self, tmp_path):
        """Test that setting a new file replaces the previous file handler."""
        log_file_1 = tmp_path / "first.log"
        log_file_2 = tmp_path / "second.log"
        configure_logger(file=log_file_1)
        first_handler = cosecha_logging._file_handler
        configure_logger(file=log_file_2)
        assert cosecha_logging._file_handler is not first_handler
        result = get_log_file_path()
        assert result is not None
        assert result.name == "second.log"

    def test_file_creates_parent_directories(self, tmp_path):
        """Test that file logging creates parent directories if needed."""
        log_file = tmp_path / "nested" / "dirs" / "app.log"
        configure_logger(file=log_file)
        assert log_file.parent.exists()
        assert cosecha_logging._file_handler is not None
