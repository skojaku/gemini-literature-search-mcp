#!/bin/bash

# Gemini Literature Search Server Runner
# This script runs the Gemini-powered literature search MCP server

set -e

echo "Starting Gemini Literature Search Server..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run setup_venv.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

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

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the server
echo "Starting server on http://localhost:8000..."
python gemini_literature_search.py

echo "Server stopped."