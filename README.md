# Gemini Literature Search MCP Server

This is a Model Context Protocol (MCP) server that provides AI-powered academic literature search and validation capabilities using Google Gemini. It helps researchers manage literature databases, validate research relevance, and ensure proper citation practices.

## Features

The Gemini Literature Search MCP Server provides AI-powered tools for academic research:

- **Literature Management**:
  - Add new literature entries with metadata (title, authors, abstract, keywords, DOI)
  - Search literature database with semantic matching
  - List and retrieve literature details
- **AI-Powered Validation**:
  - Validate literature relevance to research topics
  - Check citation appropriateness for specific sentences
  - Generate comprehensive literature summaries
- **Research Support**:
  - Semantic search using Google Gemini AI
  - Contextual analysis of research papers
  - Citation validation and recommendations
- **General AI Assistant**:
  - Direct delegation of any task to Gemini
  - Optional Google Search grounding for real-time information
  - Flexible tool for any research or analysis needs
  - Configurable model selection for different capabilities

## Installation

### Prerequisites

- Python 3.10+ (recommended: Python 3.11+)
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver
- Claude Desktop app (to use the MCP server with Claude)
- Google Gemini API key

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/skojaku/gemini-literature-search-mcp.git
   cd gemini-literature-search-mcp
   ```

2. Install dependencies with uv:
   ```bash
   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Sync dependencies
   uv sync
   ```

### Gemini Literature Search Server Setup

1. Get a Google Gemini API key:
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key

2. Configure your API key:
   ```bash
   cp .env.template .env
   # Edit .env and add your API key: GEMINI_API_KEY=your_api_key_here
   ```

3. Run the Gemini literature search server:
   ```bash
   # Using the run script
   chmod +x run_gemini_server.sh
   ./run_gemini_server.sh
   
   # Or directly with uv
   uv run python gemini_literature_search.py
   ```

## Integration with Claude Desktop

To use this MCP server with Claude Desktop, you need to manually configure it in your Claude Desktop configuration file.

### Manual Configuration Steps

1. **Install uv** if you haven't already ([Installation Guide](https://github.com/astral-sh/uv))

2. **Find your uv path**:
   ```bash
   which uv
   # Example output: /Users/yourusername/.local/bin/uv
   ```

3. **Locate your Claude Desktop config file**:
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

4. **Edit the config file** to add the Literature Search server:
   ```json
   {
     "mcpServers": {
       "Literature Search": {
         "command": "/Users/yourusername/.local/bin/uv",
         "args": [
           "run",
           "--with",
           "fastmcp",
           "--with",
           "google-generativeai",
           "--with",
           "google-genai",
           "--with",
           "requests",
           "--with",
           "python-dotenv",
           "fastmcp",
           "run",
           "/path/to/your/gemini-literature-search-mcp/gemini_literature_search.py"
         ]
       }
     }
   }
   ```

5. **Important replacements**:
   - Replace `/Users/yourusername/.local/bin/uv` with your actual uv path from step 2
   - Replace `/path/to/your/gemini-literature-search-mcp/gemini_literature_search.py` with the full path to your cloned repository

6. **Restart Claude Desktop** for the changes to take effect

### Key Points
- **Use absolute paths**: Both for the `uv` command and the Python script
- **Include all dependencies**: Use `--with` flags for each required package
- **Check paths**: Ensure all file paths exist and are accessible


## Integration with Claude Code

1. Make sure to install the MCP server in Claude Desktop

2. Run `claude mcp add-from-claude-desktop`

3. Select "FastMCP"

## Usage Examples

Here are examples of how to use the literature search functionality:

#### Adding Literature
```
Add this paper to the literature database:
Title: "Deep Learning Applications in Medical Diagnosis"
Authors: ["Smith, J.", "Johnson, A.", "Brown, K."]
Year: 2023
Abstract: "This paper explores the application of deep learning techniques in medical diagnosis, showing significant improvements in accuracy and efficiency."
Keywords: ["deep learning", "medical diagnosis", "healthcare", "AI"]
```

#### Searching Literature
```
Search for papers related to "machine learning in healthcare applications"
```

#### Validating Literature Relevance
```
Check if literature entry #1 is relevant to my research on "neural networks for medical image analysis"
```

#### Validating Citations
```
Is this citation appropriate: "Deep learning has revolutionized medical diagnosis with accuracy rates exceeding 95% (Smith et al., 2023)." - citing literature entry #1
```

#### Generating Literature Summary
```
Generate a comprehensive summary of literature entries #1, #2, and #3 focused on "AI applications in healthcare"
```

#### General Gemini Assistant
```
Ask Gemini: "What are the latest breakthroughs in CRISPR gene editing technology?"
```

```
Ask Gemini without web search: "Explain the concept of machine learning overfitting in simple terms"
```

```
Ask Gemini with specific model: "Analyze recent AI trends" using model "gemini-1.5-pro"
```

### Available Models

The server supports various Gemini models. You can specify the model parameter in `search_literature` and `ask_gemini` tools:

- **`gemini-2.0-flash-exp`** (default) - Latest experimental model with grounding
- **`gemini-1.5-pro`** - High-capability model for complex tasks
- **`gemini-1.5-flash`** - Fast model for quick responses
- **`gemini-1.0-pro`** - Stable production model

Example usage:
```
search_literature("AI in medicine", model="gemini-1.5-pro")
ask_gemini("Explain quantum physics", model="gemini-1.5-flash", use_search=False)
```

## Development

### Interactive Development Mode

For development and debugging, you can use the FastMCP development mode:
```bash
fastmcp dev gemini_literature_search.py
```

This will start a local web interface where you can test all tools interactively.

### Running Tests

```bash
# Run tests with uv
uv run pytest

# Or run with specific options
uv run pytest -v --tb=short
```

### Code Quality

```bash
# Format code with black
uv run black .

# Lint with ruff
uv run ruff check .

# Type checking with mypy
uv run mypy gemini_literature_search.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [FastMCP](https://github.com/jlowin/fastmcp) for the Pythonic MCP server framework
- [Google Gemini AI](https://ai.google.dev/) for AI-powered literature analysis
- The Model Context Protocol community for establishing the MCP standard
