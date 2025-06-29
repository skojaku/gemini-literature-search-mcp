#!/usr/bin/env python3
"""
Fallback runner for gemini_literature_search.py that handles different execution environments.
This script tries uv first, then falls back to regular Python with pip-installed dependencies.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_uv_available():
    """Check if uv is available in PATH."""
    return shutil.which("uv") is not None

def check_dependencies():
    """Check if required dependencies are available."""
    required_packages = [
        "fastmcp",
        "google.generativeai", 
        "requests",
        "pydantic",
        "dotenv"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace(".", "_") if "." in package else package)
        except ImportError:
            missing.append(package)
    
    return missing

def main():
    script_dir = Path(__file__).parent
    main_script = script_dir / "gemini_literature_search.py"
    
    if not main_script.exists():
        print(f"Error: {main_script} not found", file=sys.stderr)
        sys.exit(1)
    
    # Try uv first if available
    if check_uv_available():
        try:
            # Change to script directory
            os.chdir(script_dir)
            result = subprocess.run(
                ["uv", "run", "python", str(main_script)],
                check=False
            )
            sys.exit(result.returncode)
        except Exception as e:
            print(f"uv execution failed: {e}", file=sys.stderr)
            print("Falling back to regular Python...", file=sys.stderr)
    
    # Fallback to regular Python
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"Error: Missing dependencies: {', '.join(missing_deps)}", file=sys.stderr)
        print("Please install dependencies with:", file=sys.stderr)
        print("  pip install fastmcp google-generativeai requests pydantic python-dotenv", file=sys.stderr)
        print("Or use uv: uv sync", file=sys.stderr)
        sys.exit(1)
    
    # Run with regular Python
    try:
        os.chdir(script_dir)
        result = subprocess.run([sys.executable, str(main_script)], check=False)
        sys.exit(result.returncode)
    except Exception as e:
        print(f"Python execution failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()