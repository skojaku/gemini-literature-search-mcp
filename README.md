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

## Installation

### Prerequisites

- Python 3.10+ (recommended: Python 3.11+)
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Claude Desktop app (to use the MCP server with Claude)
- Google Gemini API key (for literature search server)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/skojaku/gemini-literature-search-mcp.git
   cd gemini-literature-search-mcp
   ```

2. (Option 1) Setup with the provided script:
   ```bash
   chmod +x setup_venv.sh
   ./setup_venv.sh
   ```

   (Option 2) Or manually set up the virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
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
   chmod +x run_gemini_server.sh
   ./run_gemini_server.sh
   ```

## Integration with Claude Desktop

To use this MCP server with Claude Desktop:

1. Make sure you have uv installed ([Installation Guide](https://github.com/astral-sh/uv))

2. Install the MCP server in Claude Desktop:
   ```bash
   fastmcp install gemini_literature_search.py
   ```

   Or with a custom name:
   ```bash
   fastmcp install gemini_literature_search.py --name "Literature Search"
   ```

3. Once installed, Claude will automatically have access to all the literature search tools and functions.


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

## Development

### Interactive Development Mode

For development and debugging, you can use the FastMCP development mode:
```bash
fastmcp dev gemini_literature_search.py
```

This will start a local web interface where you can test all tools interactively.

### Running Tests

You can test individual functions by running the server and using Claude Desktop or the development interface.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [FastMCP](https://github.com/jlowin/fastmcp) for the Pythonic MCP server framework
- [Google Gemini AI](https://ai.google.dev/) for AI-powered literature analysis
- The Model Context Protocol community for establishing the MCP standard
