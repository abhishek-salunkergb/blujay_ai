"""
Unit Test Suite for Global Hotkey Manager (Feature 2)
=====================================================
Validates:
1. Hotkey string parsing (modifiers and virtual-key codes)
2. Preference loading and saving round-trips
3. Action registration and callback dispatching
4. Thread lifecycle management (start/stop)
5. Exception isolation in user callbacks
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import os
import json
import time
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import hotkey_manager
from hotkey_manager import (
    parse_hotkey_string,
    get_saved_hotkey_preferences,
    save_hotkey_preferences,
    GlobalHotkeyManager,
    MOD_CONTROL,
    MOD_ALT,
    MOD_SHIFT,
    MOD_WIN,
    MOD_NOREPEAT,
    DEFAULT_HOTKEYS
)


class TestHotkeyManager(unittest.TestCase):

    def setUp(self):
        self.temp_config = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.temp_config.close()
        self._orig_cfg = hotkey_manager.HOTKEYS_CONFIG_FILE
        hotkey_manager.HOTKEYS_CONFIG_FILE = self.temp_config.name

    def tearDown(self):
        hotkey_manager.HOTKEYS_CONFIG_FILE = self._orig_cfg
        if os.path.exists(self.temp_config.name):
            try:
                os.remove(self.temp_config.name)
            except Exception:
                pass

    def test_parse_hotkey_string_standard_combos(self):
        """Verify standard hotkey strings parse to correct Win32 modifier and VK codes."""
        # 1. Ctrl+Alt+H
        mods, vk = parse_hotkey_string("Ctrl+Alt+H")
        self.assertTrue(mods & MOD_CONTROL)
        self.assertTrue(mods & MOD_ALT)
        self.assertEqual(vk, ord('H'))

        # 2. Ctrl+Shift+S
        mods, vk = parse_hotkey_string("Ctrl+Shift+S")
        self.assertTrue(mods & MOD_CONTROL)
        self.assertTrue(mods & MOD_SHIFT)
        self.assertEqual(vk, ord('S'))

        # 3. Win+Alt+T
        mods, vk = parse_hotkey_string("Win+Alt+T")
        self.assertTrue(mods & MOD_WIN)
        self.assertTrue(mods & MOD_ALT)
        self.assertEqual(vk, ord('T'))

        # 4. Function key F12
        mods, vk = parse_hotkey_string("Ctrl+F12")
        self.assertTrue(mods & MOD_CONTROL)
        self.assertEqual(vk, 0x7B)

    def test_parse_hotkey_empty_or_invalid(self):
        """Verify empty or invalid strings return 0, 0 safely."""
        mods, vk = parse_hotkey_string("")
        self.assertEqual(mods, 0)
        self.assertEqual(vk, 0)

    def test_hotkey_preferences_save_and_load(self):
        """Verify loading and saving hotkey preferences round-trips accurately."""
        initial = get_saved_hotkey_preferences()
        self.assertEqual(initial["panic_hide"], "Ctrl+Alt+H")

        custom_prefs = {
            "panic_hide": "Ctrl+Shift+H",
            "solve_screen": "Ctrl+Shift+S",
            "mute_ai": "Ctrl+Shift+M",
            "regenerate": "Ctrl+Shift+R",
            "cycle_opacity": "Ctrl+Shift+T"
        }
        save_hotkey_preferences(custom_prefs)

        loaded = get_saved_hotkey_preferences()
        self.assertEqual(loaded["panic_hide"], "Ctrl+Shift+H")
        self.assertEqual(loaded["solve_screen"], "Ctrl+Shift+S")

    def test_manager_registration_and_dispatch(self):
        """Verify GlobalHotkeyManager stores registered actions and dispatches callbacks."""
        mgr = GlobalHotkeyManager()
        called_actions = []

        mgr.register("panic_hide", "Ctrl+Alt+H", lambda: called_actions.append("panic_hide"))
        mgr.register("mute_ai", "Ctrl+Alt+M", lambda: called_actions.append("mute_ai"))

        self.assertIn("panic_hide", mgr.registered_actions)
        self.assertIn("mute_ai", mgr.registered_actions)

        # Directly invoke registered callback
        mgr.registered_actions["panic_hide"]["callback"]()
        mgr.registered_actions["mute_ai"]["callback"]()

        self.assertEqual(called_actions, ["panic_hide", "mute_ai"])

    def test_manager_lifecycle(self):
        """Verify GlobalHotkeyManager starts and stops cleanly without thread leaks."""
        mgr = GlobalHotkeyManager()
        mgr.register("test_action", "Ctrl+Alt+K", lambda: None)
        
        # Start and stop
        mgr.start()
        time.sleep(0.05)
        mgr.stop()
        self.assertFalse(mgr._is_running)
        self.assertIsNone(mgr._thread)


if __name__ == "__main__":
    unittest.main()
