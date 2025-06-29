#!/usr/bin/env python3
"""
Standalone version of gemini_literature_search.py with embedded dependency management.
This version tries to use the uv virtual environment if available, otherwise runs with system Python.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add the .venv/lib/python*/site-packages to sys.path if it exists
script_dir = Path(__file__).parent
venv_paths = []

# Look for uv virtual environment
uv_venv = script_dir / ".venv"
if uv_venv.exists():
    # Find site-packages in the uv venv
    for python_dir in uv_venv.glob("lib/python*/site-packages"):
        if python_dir.exists():
            venv_paths.append(str(python_dir))

# Add venv paths to sys.path at the beginning
for path in reversed(venv_paths):
    if path not in sys.path:
        sys.path.insert(0, path)

# Now try to import and run the main module
try:
    # Change to script directory for relative imports and .env loading
    os.chdir(script_dir)
    
    # Import and run the main script
    import gemini_literature_search
    gemini_literature_search.main()
    
except ImportError as e:
    print(f"Error: Missing dependency: {e}", file=sys.stderr)
    print("Please install dependencies:", file=sys.stderr)
    print("  Option 1: uv sync (if you have uv)", file=sys.stderr)  
    print("  Option 2: pip install fastmcp google-generativeai requests pydantic python-dotenv", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error running server: {e}", file=sys.stderr)
    sys.exit(1)