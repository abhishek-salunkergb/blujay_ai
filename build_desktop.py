"""
=============================================================================
BluJay AI - Desktop Client Packaging & Build Pipeline
=============================================================================
Automates building the standalone Windows desktop client using PyInstaller.
Validates model weights, static assets, and native dependencies prior to compile.

Usage:
    python build_desktop.py [--clean] [--onefile]
=============================================================================
"""

import os
import sys
import shutil
import subprocess
import argparse


def validate_prerequisites(base_dir: str):
    """Ensure all required runtime models and assets exist before building."""
    print("🔍 [1/4] Validating prerequisite assets...")
    
    silero_model = os.path.join(base_dir, "silero_vad.onnx")
    if not os.path.isfile(silero_model):
        raise FileNotFoundError(f"Missing required VAD neural model: {silero_model}")
    print(f"  ✓ Found Silero VAD ONNX model ({os.path.getsize(silero_model):,} bytes)")

    frontend_dir = os.path.join(base_dir, "frontend")
    required_frontend_files = ["home.html", "home.css", "home.js", "index.html", "style.css", "app.js"]
    for f in required_frontend_files:
        path = os.path.join(frontend_dir, f)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing essential frontend asset: {path}")
    print("  ✓ Found all required frontend HTML/CSS/JS assets")

    main_entrypoint = os.path.join(base_dir, "main.py")
    if not os.path.isfile(main_entrypoint):
        raise FileNotFoundError(f"Missing application entrypoint: {main_entrypoint}")
    print("  ✓ Found entrypoint main.py")


def check_pyinstaller():
    """Verify PyInstaller is available in current Python environment."""
    print("🔧 [2/4] Checking build toolchain...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "PyInstaller", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"  ✓ PyInstaller version: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ PyInstaller is not installed in the active environment.")
        print("   Install it using: pip install pyinstaller")
        sys.exit(1)


def clean_build_artifacts(base_dir: str):
    """Remove previous dist and build artifacts."""
    print("🧹 [3/4] Cleaning previous build artifacts...")
    for dir_name in ["build", "dist"]:
        path = os.path.join(base_dir, dir_name)
        if os.path.isdir(path):
            try:
                shutil.rmtree(path)
                print(f"  ✓ Removed {dir_name}/")
            except Exception as e:
                print(f"  ⚠️ Warning removing {dir_name}/: {e}")


def execute_build(base_dir: str, clean: bool, onefile: bool):
    """Execute PyInstaller build."""
    print("🚀 [4/4] Compiling Windows Desktop Client...")
    spec_file = os.path.join(base_dir, "blujay.spec")
    
    cmd = [sys.executable, "-m", "PyInstaller"]
    if clean:
        cmd.append("--clean")
    
    cmd.append(spec_file)

    print(f"  Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        dist_dir = os.path.join(base_dir, "dist", "BluJayAI")
        print("\n" + "=" * 70)
        print("✅ SUCCESS: BluJay AI Desktop Client built successfully!")
        print(f"📁 Output Directory: {dist_dir}")
        print("🚀 Executable: BluJayAI.exe")
        print("=" * 70)
    else:
        print(f"\n❌ Build failed with return code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Build BluJay AI Windows Desktop Executable")
    parser.add_argument("--clean", action="store_true", help="Clean cache before building")
    parser.add_argument("--onefile", action="store_true", help="Package into a single standalone exe")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        validate_prerequisites(base_dir)
        check_pyinstaller()
        if args.clean:
            clean_build_artifacts(base_dir)
        execute_build(base_dir, clean=args.clean, onefile=args.onefile)
    except Exception as e:
        print(f"\n❌ Pre-flight check failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
