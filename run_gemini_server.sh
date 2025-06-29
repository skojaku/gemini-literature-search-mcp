#!/bin/bash

# Gemini Literature Search Server Runner
# This script runs the Gemini-powered literature search MCP server using uv

set -e

echo "Starting Gemini Literature Search Server..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed."
    echo "Please install uv: https://github.com/astral-sh/uv"
    echo "Quick install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found."
    echo "Please create a .env file with your GEMINI_API_KEY."
    echo "You can copy .env.template to .env and fill in your API key."
    echo ""
    echo "Example .env content:"
    echo "GEMINI_API_KEY=your_api_key_here"
    echo ""
    read -p "Do you want to continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Sync dependencies with uv
echo "Syncing dependencies with uv..."
uv sync

# Run the server
echo "Starting server on http://localhost:8000..."
uv run gemini-literature-search

echo "Server stopped."