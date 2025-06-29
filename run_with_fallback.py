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

def install_dependencies():
    """Install dependencies using pip if they're missing."""
    required_packages = [
        "fastmcp>=0.4.1",
        "google-generativeai>=0.8.0",
        "requests>=2.31.0", 
        "pydantic>=2.10.0",
        "python-dotenv>=1.0.0"
    ]
    
    # Check if we need to install anything
    missing = []
    package_names = {
        "fastmcp": "fastmcp",
        "google-generativeai": "google.generativeai",
        "requests": "requests",
        "pydantic": "pydantic", 
        "python-dotenv": "dotenv"
    }
    
    for pkg_name, import_name in package_names.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg_name)
    
    if missing:
        print(f"Installing missing dependencies: {', '.join(missing)}", file=sys.stderr)
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "--user"
            ] + [pkg for pkg in required_packages if any(m in pkg for m in missing)], 
            check=True, capture_output=True, text=True)
            print("Dependencies installed successfully", file=sys.stderr)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to install dependencies: {e}", file=sys.stderr)
            print(f"pip stderr: {e.stderr}", file=sys.stderr)
            return False
    
    return True

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
    
    # Fallback to regular Python - install dependencies if needed
    if not install_dependencies():
        print("Failed to install required dependencies", file=sys.stderr)
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