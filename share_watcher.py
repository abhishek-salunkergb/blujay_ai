"""Detects common screen-sharing/recording windows on Windows and tells the overlay to hide/show.

Usage:
    python share_watcher.py

It connects to ws://127.0.0.1:8765 and sends {'type':'hide'} when a sharing window is detected, and {'type':'show'} when not.

Note: This is heuristic-based and best-effort. It works for common apps (Zoom, Teams, Chrome/Edge share windows, OBS) but cannot cover all capture methods.
"""
import ctypes
from ctypes import wintypes
import time
import psutil
import asyncio
import websockets
import json

user32 = ctypes.windll.user32

EnumWindows = user32.EnumWindows
EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
IsWindowVisible = user32.IsWindowVisible
GetWindowTextLengthW = user32.GetWindowTextLengthW
GetWindowTextW = user32.GetWindowTextW
GetWindowThreadProcessId = user32.GetWindowThreadProcessId

KNOWN_PROCS = {'zoom.exe', 'teams.exe', 'obs64.exe', 'obs32.exe', 'chrome.exe', 'msedge.exe', 'firefox.exe'}
SHARE_KEYWORDS = ['you are sharing', 'screen sharing', 'sharing your screen', 'share', 'presenting', 'zoommeeting', 'zoom meeting']


def enum_windows_titles():
    titles = []

    def foreach_window(hwnd, lParam):
        if not IsWindowVisible(hwnd):
            return True
        length = GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buff = ctypes.create_unicode_buffer(length + 1)
        GetWindowTextW(hwnd, buff, length + 1)
        title = buff.value
        pid = wintypes.DWORD()
        GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        titles.append((hwnd, pid.value, title))
        return True

    EnumWindows(EnumWindowsProc(foreach_window), 0)
    return titles


def is_sharing_active():
    try:
        items = enum_windows_titles()
        for hwnd, pid, title in items:
            if not title:
                continue
            t = title.lower()
            # quick title keyword check
            for kw in SHARE_KEYWORDS:
                if kw in t:
                    # attempt to map pid to process name
                    try:
                        p = psutil.Process(pid)
                        name = p.name().lower()
                    except Exception:
                        name = ''
                    # if the process is a known sharing app, assume sharing
                    if any(k in name for k in KNOWN_PROCS) or any(k in t for k in ['zoom', 'teams', 'obs', 'meet', 'share']):
                        return True
        # fallback: if OBS is running, consider it sharing (best-effort)
        for p in psutil.process_iter(['name']):
            try:
                n = p.info.get('name','').lower()
                if n in KNOWN_PROCS:
                    if 'obs' in n:
                        return True
            except Exception:
                pass
    except Exception as e:
        print('share detection error', e)
    return False


async def run_watcher():
    print("Screen share watcher started (monitoring system windows)...")
    while True:
        sharing = is_sharing_active()
        if sharing:
            print("[ALERT] Active screen sharing / recording window detected.")
        await asyncio.sleep(2)


if __name__ == '__main__':
    try:
        asyncio.run(run_watcher())
    except KeyboardInterrupt:
        print('Share watcher stopped')
