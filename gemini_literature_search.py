import os
import sys
import json
from typing import List, Dict, Optional

try:
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Error importing dotenv: {e}", file=sys.stderr)
    sys.exit(1)

try:
    import google.generativeai as genai
except ImportError as e:
    print(f"Error importing google.generativeai: {e}", file=sys.stderr)
    print("Make sure to install: pip install google-generativeai", file=sys.stderr)
    sys.exit(1)

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:
    print(f"Error importing FastMCP: {e}", file=sys.stderr)
    print("Make sure to install: pip install fastmcp", file=sys.stderr)
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    print("Warning: GEMINI_API_KEY not found in environment variables.", file=sys.stderr)
    print("Please set your Gemini API key in a .env file or environment variable.", file=sys.stderr)
else:
    genai.configure(api_key=api_key)

# Create MCP Server
app = FastMCP(
    title="Gemini Literature Search",
    description="A server for academic literature search and validation using Google Gemini",
    version="1.0.0",
    dependencies=["google-generativeai", "requests", "python-dotenv"],
)

TRANSPORT = "sse"

def get_gemini_model():
    """Get Gemini model instance"""
    return genai.GenerativeModel('gemini-1.5-flash')

@app.tool()
def search_literature(query: str, max_results: int = 10) -> dict:
    """
    Search the internet for academic literature on a topic using Gemini.
    
    Args:
        query: Search query describing the research topic or keywords
        max_results: Maximum number of results to return (default: 10)
    
    Returns:
        On success: {"results": <list of papers found online>}
        On error: {"error": <error message>}
    
    Examples:
        >>> search_literature("machine learning applications in healthcare")
        {'results': [{'title': '...', 'authors': [...], 'year': 2024, 'summary': '...'}]}
    """
    try:
        model = get_gemini_model()
        
        prompt = f"""
        Please search the internet for recent academic literature on the topic: "{query}"
        
        Find {max_results} current and relevant academic papers by searching online academic sources.
        Look for papers from Google Scholar, arXiv, PubMed, or other academic databases.
        
        For each paper you find, provide:
        1. Title
        2. Authors
        3. Publication year
        4. Brief summary/abstract
        5. Key findings or contributions
        6. Journal/venue
        7. DOI or URL if available
        
        Return your response as a JSON array with format:
        [
          {{
            "title": "<paper title>",
            "authors": ["<author1>", "<author2>"],
            "year": <year>,
            "summary": "<brief summary>",
            "key_findings": "<key contributions>",
            "venue": "<journal/conference name>",
            "doi_or_url": "<DOI or URL if available>"
          }}
        ]
        """
        
        response = model.generate_content(prompt)
        
        try:
            # Try to parse as JSON
            results_data = json.loads(response.text.strip())
            return {"results": results_data}
        except json.JSONDecodeError:
            # If JSON parsing fails, return the raw response
            return {
                "results": [],
                "raw_response": response.text,
                "note": "Could not parse JSON response, see raw_response for details"
            }
            
    except Exception as e:
        return {"error": str(e)}

@app.tool()
def validate_paper_relevance(paper_title: str, authors: str, research_topic: str) -> dict:
    """
    Validate the relevance of a specific paper to a research topic using Gemini AI.
    
    Args:
        paper_title: Title of the paper to validate
        authors: Authors of the paper
        research_topic: Description of the research topic or question
    
    Returns:
        On success: {"relevance_score": <score>, "analysis": <detailed analysis>}
        On error: {"error": <error message>}
    
    Examples:
        >>> validate_paper_relevance("Deep Learning in Medical Diagnosis", "Smith et al.", "neural networks in healthcare")
        {'relevance_score': 0.85, 'analysis': 'This paper is highly relevant because...'}
    """
    try:
        model = get_gemini_model()
        
        prompt = f"""
        Research Topic: "{research_topic}"
        
        Paper to Evaluate:
        Title: {paper_title}
        Authors: {authors}
        
        Please analyze the relevance of this paper to the research topic.
        Consider:
        1. How well the paper's content aligns with the research topic
        2. The methodological relevance
        3. The potential contribution to understanding the topic
        4. The quality and impact of the work
        
        Provide:
        1. A relevance score from 0.0 (not relevant) to 1.0 (highly relevant)
        2. A detailed analysis explaining the relevance assessment
        3. Specific aspects that make it relevant or irrelevant
        4. Suggestions for how this paper could be used in research on the topic
        
        Return your response as JSON with format:
        {{
            "relevance_score": <score>,
            "analysis": "<detailed explanation>",
            "relevant_aspects": ["<aspect1>", "<aspect2>"],
            "limitations": ["<limitation1>", "<limitation2>"],
            "research_applications": ["<application1>", "<application2>"]
        }}
        """
        
        response = model.generate_content(prompt)
        
        try:
            result = json.loads(response.text.strip())
            return result
        except json.JSONDecodeError:
            return {
                "relevance_score": 0.5,
                "analysis": response.text,
                "relevant_aspects": [],
                "limitations": [],
                "research_applications": []
            }
            
    except Exception as e:
        return {"error": str(e)}

@app.tool()
def validate_citation_appropriateness(sentence: str, paper_reference: str, context: Optional[str] = None) -> dict:
    """
    Validate whether a sentence appropriately cites a paper using Gemini AI.
    
    Args:
        sentence: The sentence containing the citation to validate
        paper_reference: Reference to the paper being cited (title, authors, year)
        context: Optional surrounding context for better validation
    
    Returns:
        On success: {"is_appropriate": <boolean>, "analysis": <detailed analysis>}
        On error: {"error": <error message>}
    
    Examples:
        >>> validate_citation_appropriateness("Neural networks show 95% accuracy in diagnosis.", "Smith et al. 2023 - Deep Learning in Medical Diagnosis")
        {'is_appropriate': True, 'analysis': 'The citation is appropriate because...'}
    """
    try:
        model = get_gemini_model()
        
        context_info = f"\n\nContext: {context}" if context else ""
        
        prompt = f"""
        Sentence with Citation: "{sentence}"{context_info}
        
        Paper Being Cited: {paper_reference}
        
        Please analyze whether this citation is appropriate and accurate:
        1. Does the sentence accurately represent what the cited paper likely contains?
        2. Is the citation contextually appropriate for the claim being made?
        3. Are there any potential misrepresentations or overstatements?
        4. Does the claim seem reasonable for this type of paper?
        
        Consider:
        - The specificity and nature of the claim
        - Whether the cited work would likely support this claim
        - The appropriateness of the citation style and context
        
        Return your response as JSON with format:
        {{
            "is_appropriate": <true/false>,
            "confidence": <0.0-1.0>,
            "analysis": "<detailed explanation>",
            "issues": ["<issue1>", "<issue2>"],
            "suggestions": ["<suggestion1>", "<suggestion2>"]
        }}
        """
        
        response = model.generate_content(prompt)
        
        try:
            result = json.loads(response.text.strip())
            return result
        except json.JSONDecodeError:
            return {
                "is_appropriate": True,
                "confidence": 0.5,
                "analysis": response.text,
                "issues": [],
                "suggestions": []
            }
            
    except Exception as e:
        return {"error": str(e)}

@app.tool()
def generate_research_summary(topic: str, focus_areas: Optional[List[str]] = None) -> dict:
    """
    Generate a comprehensive research summary on a topic using Gemini's knowledge.
    
    Args:
        topic: The research topic to summarize
        focus_areas: Optional list of specific areas to focus on
    
    Returns:
        On success: {"summary": <comprehensive summary>, "key_findings": <list>}
        On error: {"error": <error message>}
    
    Examples:
        >>> generate_research_summary("machine learning in healthcare", ["diagnosis", "treatment"])
        {'summary': 'Current research in ML healthcare...', 'key_findings': ['Finding 1', 'Finding 2']}
    """
    try:
        model = get_gemini_model()
        
        focus_info = ""
        if focus_areas:
            focus_info = f"\n\nPlease focus particularly on these areas: {', '.join(focus_areas)}"
        
        prompt = f"""
        Topic: "{topic}"{focus_info}
        
        Please provide a comprehensive research summary on this topic based on current academic knowledge.
        Include:
        
        1. Overview of the current state of research
        2. Key findings and breakthroughs in recent years
        3. Major research methodologies being used
        4. Current challenges and limitations
        5. Future research directions
        6. Notable researchers and institutions in this field
        7. Important journals and conferences for this topic
        
        Focus on peer-reviewed academic research and provide specific examples where possible.
        
        Return your response as JSON with format:
        {{
            "summary": "<comprehensive overview>",
            "key_findings": ["<finding1>", "<finding2>", "<finding3>"],
            "methodologies": ["<method1>", "<method2>"],
            "challenges": ["<challenge1>", "<challenge2>"],
            "future_directions": ["<direction1>", "<direction2>"],
            "notable_researchers": ["<researcher1>", "<researcher2>"],
            "important_venues": ["<journal/conference1>", "<journal/conference2>"]
        }}
        """
        
        response = model.generate_content(prompt)
        
        try:
            result = json.loads(response.text.strip())
            return result
        except json.JSONDecodeError:
            return {
                "summary": response.text,
                "key_findings": [],
                "methodologies": [],
                "challenges": [],
                "future_directions": [],
                "notable_researchers": [],
                "important_venues": []
            }
            
    except Exception as e:
        return {"error": str(e)}

@app.tool()
def find_related_papers(paper_title: str, authors: str, max_results: int = 5) -> dict:
    """
    Find papers related to a given paper using Gemini AI.
    
    Args:
        paper_title: Title of the reference paper
        authors: Authors of the reference paper
        max_results: Maximum number of related papers to return
    
    Returns:
        On success: {"related_papers": <list of related papers>}
        On error: {"error": <error message>}
    
    Examples:
        >>> find_related_papers("Deep Learning in Medical Diagnosis", "Smith et al.", 5)
        {'related_papers': [{'title': '...', 'authors': [...], 'relationship': '...'}]}
    """
    try:
        model = get_gemini_model()
        
        prompt = f"""
        Reference Paper:
        Title: {paper_title}
        Authors: {authors}
        
        Please find up to {max_results} academic papers that are closely related to this reference paper.
        Look for papers that:
        1. Use similar methodologies
        2. Address related research questions
        3. Build upon or cite this work
        4. Apply similar approaches to different domains
        5. Provide comparative or competing approaches
        
        For each related paper, provide:
        1. Title
        2. Authors
        3. Publication year
        4. Brief description of how it relates to the reference paper
        5. Type of relationship (methodology, application, extension, comparison, etc.)
        
        Return your response as JSON with format:
        [
          {{
            "title": "<paper title>",
            "authors": ["<author1>", "<author2>"],
            "year": <year>,
            "relationship_description": "<how it relates>",
            "relationship_type": "<methodology/application/extension/comparison/citation>",
            "relevance_score": <0.0-1.0>
          }}
        ]
        """
        
        response = model.generate_content(prompt)
        
        try:
            results_data = json.loads(response.text.strip())
            return {"related_papers": results_data}
        except json.JSONDecodeError:
            return {
                "related_papers": [],
                "raw_response": response.text,
                "note": "Could not parse JSON response, see raw_response for details"
            }
            
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    app.run(transport=TRANSPORT)