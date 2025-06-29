# FastMCP Servers Collection

This repository contains multiple Model Context Protocol (MCP) servers built with FastMCP:

1. **Mathematical Calculator MCP Server** - Advanced mathematical calculations, symbolic math, and statistical analysis
2. **Gemini Literature Search MCP Server** - AI-powered academic literature search and validation using Google Gemini

## Features

### Mathematical Calculator MCP Server

The Mathematical Calculator MCP Server provides the following tools:

- **Basic Calculations**: Evaluate mathematical expressions safely
- **Symbolic Mathematics**:
  - Solve equations (linear, quadratic, polynomial, etc.)
  - Calculate derivatives of expressions
  - Compute integrals of expressions
- **Statistical Analysis**:
  - Mean, median, mode
  - Variance, standard deviation
  - Correlation coefficient
  - Linear regression
  - Confidence intervals
- **Matrix Operations**:
  - Matrix addition
  - Matrix multiplication
  - Matrix transposition

### Gemini Literature Search MCP Server

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
   git clone https://github.com/huhabla/calculator-mcp-server.git
   cd calculator-mcp-server
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

3. Run doc-tests to verify everything works:
   ```bash
   bash run_doctests.sh
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
   fastmcp install calculator_server.py
   ```

   Or with a custom name:
   ```bash
   fastmcp install calculator_server.py --name "Math Calculator"
   ```

3. Once installed, Claude will automatically have access to all the mathematical tools and functions.


## Integration with Claude Code

1. Make sure to install the MCP server in Claude Desktop

2. Run `claude mcp add-from-claude-desktop`

3. Select "FastMCP"

## Usage Examples

### Mathematical Calculator Examples

After integrating with Claude Desktop, you can ask Claude to perform various mathematical operations. Here are some examples:

### Basic Calculations
```
Can you calculate 3.5^2 * sin(pi/4)?
```

### Solving Equations
```
Solve the following equation: x^2 - 5x + 6 = 0
```

### Calculating Derivatives
```
What's the derivative of sin(x^2) with respect to x?
```

### Computing Integrals
```
Calculate the integral of x^2 * e^x
```

### Statistical Analysis
```
Find the mean, median, mode, and standard deviation of this dataset: [23, 45, 12, 67, 34, 23, 18, 95, 41, 23]
```

### Linear Regression
```
Perform a linear regression on these points: (1,2), (2,3.5), (3,5.1), (4,6.5), (5,8.2)
```

### Matrix Operations
```
Multiply these two matrices:
[1, 2, 3]
[4, 5, 6]

and

[7, 8]
[9, 10]
[11, 12]
```

### Gemini Literature Search Examples

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

### Testing

Run the comprehensive doctest suite:
```bash
bash run_doctests.sh
```

### Interactive Development Mode

For development and debugging, you can use the FastMCP development mode:
```bash
fastmcp dev calculator_server.py
```

This will start a local web interface where you can test all tools interactively.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [FastMCP](https://github.com/jlowin/fastmcp) for the Pythonic MCP server framework
- [SymPy](https://sympy.org/) for symbolic mathematics
- [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/) for numerical and statistical computations
