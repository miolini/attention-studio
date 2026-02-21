from __future__ import annotations

import pytest
from pathlib import Path
import tempfile
import os

from attention_studio.ui.theme import (
    ThemeMode,
    ThemeColors,
    ThemeManager,
)


class TestThemeMode:
    def test_theme_mode_values(self):
        assert ThemeMode.LIGHT.value == "light"
        assert ThemeMode.DARK.value == "dark"
        assert ThemeMode.AUTO.value == "auto"


class TestThemeColors:
    def test_primary_colors(self):
        assert ThemeColors.PRIMARY == "#0078D4"
        assert ThemeColors.SUCCESS == "#107C10"
        assert ThemeColors.ERROR == "#D13438"

    def test_background_colors(self):
        assert ThemeColors.BACKGROUND_PRIMARY == "#FFFFFF"
        assert ThemeColors.BACKGROUND_DARK == "#1E1E1E"


class TestThemeManager:
    def test_singleton(self):
        manager1 = ThemeManager()
        manager2 = ThemeManager()
        assert manager1 is manager2

    def test_default_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            manager = ThemeManager()
            manager._settings_path = path / "theme.json"
            assert manager.mode in [ThemeMode.LIGHT, ThemeMode.DARK]

    def test_set_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            manager = ThemeManager()
            manager._settings_path = path / "theme.json"
            manager.set_mode(ThemeMode.LIGHT)
            assert manager.mode == ThemeMode.LIGHT
            manager.set_mode(ThemeMode.DARK)
            assert manager.mode == ThemeMode.DARK

    def test_toggle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            manager = ThemeManager()
            manager._settings_path = path / "theme.json"
            initial_mode = manager.mode
            manager.toggle()
            assert manager.mode != initial_mode

    def test_stylesheet(self):
        manager = ThemeManager()
        stylesheet = manager.stylesheet
        assert isinstance(stylesheet, str)
        assert len(stylesheet) > 0

    def test_is_dark(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            manager = ThemeManager()
            manager._settings_path = path / "theme.json"
            manager.set_mode(ThemeMode.DARK)
            assert manager.is_dark is True
            manager.set_mode(ThemeMode.LIGHT)
            assert manager.is_dark is False

    def test_get_color(self):
        manager = ThemeManager()
        assert manager.get_color("primary") == "#0078D4"
        assert manager.get_color("success") == "#107C10"
        assert manager.get_color("error") == "#D13438"
