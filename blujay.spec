# -*- mode: python ; coding: utf-8 -*-
# =============================================================================
# BluJay AI - Windows Desktop Client PyInstaller Specification
# =============================================================================

import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

# Project base directory
BASE_DIR = os.path.abspath(SPECPATH)

# Data files to bundle with the binary
added_datas = [
    (os.path.join(BASE_DIR, 'silero_vad.onnx'), '.'),
    (os.path.join(BASE_DIR, 'frontend'), 'frontend'),
    (os.path.join(BASE_DIR, 'knowledge'), 'knowledge'),
]

# Explicit hidden imports to prevent runtime missing module errors
hidden_imports = [
    'soundcard',
    'sounddevice',
    'webrtcvad',
    'webview',
    'pywebview',
    'customtkinter',
    'onnxruntime',
    'numpy',
    'PIL',
    'groq',
    'google.genai',
    'pydantic',
    'pydantic_settings',
    'uvicorn',
    'fastapi',
    'sqlalchemy',
    'sqlalchemy.dialects.sqlite',
    'psutil',
    'pyautogui',
    'noisereduce',
]

a = Analysis(
    ['main.py'],
    pathex=[BASE_DIR],
    binaries=[],
    datas=added_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter.test', 'unittest.test'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='BluJayAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Stealth GUI without console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='BluJayAI',
)
