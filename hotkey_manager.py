"""
Global Hotkey Manager Service
=============================
Provides OS-level global hotkey registration on Windows (via user32 RegisterHotKey)
with a background message pump and cross-platform fallbacks.
Enables instant stealth toggle, screen snip/solve, STT mute/pause, and answer regeneration
regardless of which window currently has OS focus (Zoom, Teams, Meet, VS Code, Browser).
"""

import os
import sys
import json
import ctypes
from ctypes import wintypes
import threading
from typing import Dict, Callable, Optional, Any, Tuple

HOTKEYS_CONFIG_FILE = os.path.expanduser("~/.ai_copilot_hotkeys.json")

# Windows Modifiers
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012

# Virtual-Key Code Mapping
VK_MAPPING = {
    "backspace": 0x08, "tab": 0x09, "enter": 0x0D, "return": 0x0D,
    "pause": 0x13, "caps_lock": 0x14, "escape": 0x1B, "esc": 0x1B,
    "space": 0x20, "page_up": 0x21, "page_down": 0x22, "end": 0x23, "home": 0x24,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    "print_screen": 0x2C, "insert": 0x2D, "delete": 0x2E,
    "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73, "f5": 0x74, "f6": 0x75,
    "f7": 0x76, "f8": 0x77, "f9": 0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B
}
# Alphanumeric keys: 'A'-'Z' -> 0x41-0x5A, '0'-'9' -> 0x30-0x39
for c in range(ord('a'), ord('z') + 1):
    VK_MAPPING[chr(c)] = ord(chr(c).upper())
for c in range(ord('0'), ord('9') + 1):
    VK_MAPPING[chr(c)] = ord(chr(c))


DEFAULT_HOTKEYS = {
    "panic_hide": "Ctrl+Alt+H",
    "solve_screen": "Ctrl+Alt+S",
    "mute_ai": "Ctrl+Alt+M",
    "regenerate": "Ctrl+Alt+R",
    "cycle_opacity": "Ctrl+Alt+T",
    "ghost_mode": "Ctrl+Alt+G",
    "dock_camera": "Ctrl+Alt+C"
}


def parse_hotkey_string(hotkey_str: str) -> Tuple[int, int]:
    """
    Parses a hotkey string like 'Ctrl+Alt+H' or 'Shift+F5' into (modifiers, vk_code).
    """
    if not hotkey_str:
        return 0, 0
        
    parts = [p.strip().lower() for p in hotkey_str.split("+") if p.strip()]
    modifiers = MOD_NOREPEAT
    vk_code = 0
    
    for p in parts:
        if p in ("ctrl", "control"):
            modifiers |= MOD_CONTROL
        elif p in ("alt", "option"):
            modifiers |= MOD_ALT
        elif p in ("shift",):
            modifiers |= MOD_SHIFT
        elif p in ("win", "cmd", "super", "windows"):
            modifiers |= MOD_WIN
        else:
            # Key part
            if p in VK_MAPPING:
                vk_code = VK_MAPPING[p]
            elif len(p) == 1:
                vk_code = ord(p.upper())
                
    return modifiers, vk_code


def get_saved_hotkey_preferences() -> Dict[str, str]:
    """Load hotkey preferences from file."""
    if os.path.exists(HOTKEYS_CONFIG_FILE):
        try:
            with open(HOTKEYS_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                res = DEFAULT_HOTKEYS.copy()
                res.update(data)
                return res
        except Exception:
            pass
    return DEFAULT_HOTKEYS.copy()


def save_hotkey_preferences(prefs: Dict[str, str]):
    """Save hotkey preferences to file."""
    try:
        with open(HOTKEYS_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(prefs, f, indent=2)
    except Exception as e:
        print(f"Notice saving hotkey preferences: {e}")


class GlobalHotkeyManager:
    """
    Registers system-wide hotkeys on Windows via background Win32 message pump.
    Dispatches registered actions safely.
    """

    def __init__(self):
        self.registered_actions: Dict[str, Dict[str, Any]] = {}
        self.id_to_action: Dict[int, str] = {}
        self._next_id = 1001
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self._thread_id: Optional[int] = None
        self._lock = threading.Lock()
        
        # Win32 user32 handle
        self.user32 = ctypes.windll.user32 if sys.platform == "win32" else None

    def register(self, action_name: str, key_combination: str, callback: Callable[[], None]):
        """
        Register or update a hotkey action with a key combo string and a callable.
        """
        with self._lock:
            hotkey_id = self._next_id
            self._next_id += 1
            
            modifiers, vk = parse_hotkey_string(key_combination)
            self.registered_actions[action_name] = {
                "id": hotkey_id,
                "combo": key_combination,
                "modifiers": modifiers,
                "vk": vk,
                "callback": callback
            }
            self.id_to_action[hotkey_id] = action_name

    def start(self):
        """Start the background Win32 message loop."""
        if self._is_running or not self.user32:
            return
            
        self._is_running = True
        self._thread = threading.Thread(target=self._msg_loop, daemon=True, name="GlobalHotkeyMessagePump")
        self._thread.start()

    def stop(self):
        """Stop the background message loop and unregister all hotkeys."""
        self._is_running = False
        if self.user32 and self._thread_id:
            try:
                self.user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
            except Exception:
                pass
                
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._thread = None
        self._thread_id = None

    def _msg_loop(self):
        """Dedicated message pump thread for RegisterHotKey and GetMessage."""
        if not self.user32:
            return

        kernel32 = ctypes.windll.kernel32
        self._thread_id = kernel32.GetCurrentThreadId()

        # 1. Register all hotkeys in this thread
        with self._lock:
            for action_name, data in self.registered_actions.items():
                hotkey_id = data["id"]
                mods = data["modifiers"]
                vk = data["vk"]
                if vk > 0:
                    try:
                        res = self.user32.RegisterHotKey(None, hotkey_id, mods, vk)
                        if not res:
                            # Try without MOD_NOREPEAT for older Windows versions
                            self.user32.RegisterHotKey(None, hotkey_id, mods & ~MOD_NOREPEAT, vk)
                    except Exception as e:
                        print(f"Notice registering hotkey {data['combo']}: {e}")

        # 2. Windows Message Pump Loop
        msg = wintypes.MSG()
        while self._is_running:
            b_ret = self.user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if b_ret <= 0:  # WM_QUIT or error
                break

            if msg.message == WM_HOTKEY:
                hotkey_id = msg.wParam
                with self._lock:
                    action_name = self.id_to_action.get(hotkey_id)
                    if action_name and action_name in self.registered_actions:
                        cb = self.registered_actions[action_name].get("callback")
                        if cb:
                            try:
                                cb()
                            except Exception as e:
                                print(f"Error in hotkey callback ({action_name}): {e}")

            self.user32.TranslateMessage(ctypes.byref(msg))
            self.user32.DispatchMessageW(ctypes.byref(msg))

        # 3. Clean unregister
        with self._lock:
            for action_name, data in self.registered_actions.items():
                try:
                    self.user32.UnregisterHotKey(None, data["id"])
                except Exception:
                    pass
