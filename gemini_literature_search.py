import os
import json
from typing import List, Dict, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from mcp.server.fastmcp import FastMCP

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Create MCP Server
app = FastMCP(
    title="Gemini Literature Search",
    description="A server for academic literature search and validation using Google Gemini",
    version="1.0.0",
    dependencies=["google-generativeai", "requests", "python-dotenv"],
)

TRANSPORT = "sse"

# In-memory literature database (in production, this would be a proper database)
literature_database = []

class LiteratureEntry:
    def __init__(self, title: str, authors: List[str], year: int, abstract: str, 
                 keywords: List[str], doi: Optional[str] = None, url: Optional[str] = None):
        self.title = title
        self.authors = authors
        self.year = year
        self.abstract = abstract
        self.keywords = keywords
        self.doi = doi
        self.url = url
        self.id = len(literature_database) + 1

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "abstract": self.abstract,
            "keywords": self.keywords,
            "doi": self.doi,
            "url": self.url
        }

def get_gemini_model():
    """Get Gemini model instance"""
    return genai.GenerativeModel('gemini-1.5-flash')

@app.tool()
def add_literature(
    title: str,
    authors: List[str],
    year: int,
    abstract: str,
    keywords: List[str],
    doi: Optional[str] = None,
    url: Optional[str] = None
) -> dict:
    """
    Add a new literature entry to the database.
    
    Args:
        title: The title of the paper
        authors: List of author names
        year: Publication year
        abstract: Paper abstract
        keywords: List of keywords
        doi: DOI if available
        url: URL if available
    
    Returns:
        On success: {"result": "Literature added successfully", "id": <entry_id>}
        On error: {"error": <error message>}
    
    Examples:
        >>> add_literature("Deep Learning Applications", ["Smith, J.", "Doe, A."], 2023, "This paper explores...", ["deep learning", "AI"])
        {'result': 'Literature added successfully', 'id': 1}
    """
    try:
        entry = LiteratureEntry(title, authors, year, abstract, keywords, doi, url)
        literature_database.append(entry)
        
        return {
            "result": "Literature added successfully",
            "id": entry.id,
            "entry": entry.to_dict()
        }
    except Exception as e:
        return {"error": str(e)}

@app.tool()
def search_literature(query: str, max_results: int = 10) -> dict:
    """
    Search through the literature database using Gemini AI for semantic matching.
    
    Args:
        query: Search query describing the topic or keywords
        max_results: Maximum number of results to return
    
    Returns:
        On success: {"results": <list of matching literature entries>}
        On error: {"error": <error message>}
    
    Examples:
        >>> search_literature("machine learning applications in healthcare")
        {'results': [{'id': 1, 'title': '...', 'relevance_score': 0.95}]}
    """
    try:
        if not literature_database:
            return {"results": [], "message": "No literature entries in database"}
        
        model = get_gemini_model()
        
        # Create a prompt for semantic search
        literature_summaries = []
        for entry in literature_database:
            summary = f"ID: {entry.id}, Title: {entry.title}, Abstract: {entry.abstract[:200]}..., Keywords: {', '.join(entry.keywords)}"
            literature_summaries.append(summary)
        
        prompt = f"""
        Given the search query: "{query}"
        
        Please analyze the following literature entries and rank them by relevance to the query.
        Return the top {max_results} most relevant entries with their IDs and a relevance score (0-1).
        
        Literature entries:
        {chr(10).join(literature_summaries)}
        
        Return your response as a JSON array with format:
        [{"id": <entry_id>, "relevance_score": <score>, "reason": "<brief explanation>"}]
        """
        
        response = model.generate_content(prompt)
        
        # Parse Gemini response
        try:
            results_data = json.loads(response.text.strip())
        except json.JSONDecodeError:
            # If JSON parsing fails, extract IDs manually
            results_data = []
        
        # Get full entry details for results
        results = []
        for result in results_data:
            entry_id = result.get("id")
            if entry_id and entry_id <= len(literature_database):
                entry = literature_database[entry_id - 1]
                entry_dict = entry.to_dict()
                entry_dict["relevance_score"] = result.get("relevance_score", 0)
                entry_dict["relevance_reason"] = result.get("reason", "")
                results.append(entry_dict)
        
        return {"results": results}
        
    except Exception as e:
        return {"error": str(e)}

@app.tool()
def validate_literature_relevance(literature_id: int, research_topic: str) -> dict:
    """
    Validate the relevance of a specific literature entry to a research topic using Gemini AI.
    
    Args:
        literature_id: ID of the literature entry to validate
        research_topic: Description of the research topic or question
    
    Returns:
        On success: {"relevance_score": <score>, "analysis": <detailed analysis>}
        On error: {"error": <error message>}
    
    Examples:
        >>> validate_literature_relevance(1, "application of neural networks in medical diagnosis")
        {'relevance_score': 0.85, 'analysis': 'This paper is highly relevant because...'}
    """
    try:
        if literature_id < 1 or literature_id > len(literature_database):
            return {"error": "Literature entry not found"}
        
        entry = literature_database[literature_id - 1]
        model = get_gemini_model()
        
        prompt = f"""
        Research Topic: "{research_topic}"
        
        Literature Entry:
        Title: {entry.title}
        Authors: {', '.join(entry.authors)}
        Year: {entry.year}
        Abstract: {entry.abstract}
        Keywords: {', '.join(entry.keywords)}
        
        Please analyze the relevance of this literature entry to the research topic.
        Provide:
        1. A relevance score from 0.0 (not relevant) to 1.0 (highly relevant)
        2. A detailed analysis explaining why this literature is or isn't relevant
        3. Specific aspects that make it relevant or irrelevant
        
        Return your response as JSON with format:
        {
            "relevance_score": <score>,
            "analysis": "<detailed explanation>",
            "relevant_aspects": ["<aspect1>", "<aspect2>"],
            "concerns": ["<concern1>", "<concern2>"]
        }
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
                "concerns": []
            }
            
    except Exception as e:
        return {"error": str(e)}

@app.tool()
def validate_citation_relevance(sentence: str, literature_id: int, context: Optional[str] = None) -> dict:
    """
    Validate whether a specific sentence appropriately cites a literature entry using Gemini AI.
    
    Args:
        sentence: The sentence containing the citation to validate
        literature_id: ID of the literature entry being cited
        context: Optional surrounding context for better validation
    
    Returns:
        On success: {"is_appropriate": <boolean>, "analysis": <detailed analysis>}
        On error: {"error": <error message>}
    
    Examples:
        >>> validate_citation_relevance("Neural networks have shown promising results in medical diagnosis (Smith et al., 2023).", 1)
        {'is_appropriate': True, 'analysis': 'The citation is appropriate because...'}
    """
    try:
        if literature_id < 1 or literature_id > len(literature_database):
            return {"error": "Literature entry not found"}
        
        entry = literature_database[literature_id - 1]
        model = get_gemini_model()
        
        context_info = f"\n\nContext: {context}" if context else ""
        
        prompt = f"""
        Sentence with Citation: "{sentence}"{context_info}
        
        Literature Entry Being Cited:
        Title: {entry.title}
        Authors: {', '.join(entry.authors)}
        Year: {entry.year}
        Abstract: {entry.abstract}
        Keywords: {', '.join(entry.keywords)}
        
        Please analyze whether this citation is appropriate and accurate:
        1. Does the sentence accurately represent the content/findings of the cited literature?
        2. Is the citation contextually appropriate?
        3. Are there any misrepresentations or overstatements?
        4. Does the cited work actually support the claim being made?
        
        Return your response as JSON with format:
        {
            "is_appropriate": <true/false>,
            "confidence": <0.0-1.0>,
            "analysis": "<detailed explanation>",
            "issues": ["<issue1>", "<issue2>"],
            "suggestions": ["<suggestion1>", "<suggestion2>"]
        }
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
def generate_literature_summary(literature_ids: List[int], topic: str) -> dict:
    """
    Generate a comprehensive summary of multiple literature entries for a specific topic using Gemini AI.
    
    Args:
        literature_ids: List of literature entry IDs to summarize
        topic: The research topic to focus the summary on
    
    Returns:
        On success: {"summary": <comprehensive summary>, "key_findings": <list>}
        On error: {"error": <error message>}
    
    Examples:
        >>> generate_literature_summary([1, 2, 3], "machine learning in healthcare")
        {'summary': 'Based on the reviewed literature...', 'key_findings': ['Finding 1', 'Finding 2']}
    """
    try:
        if not literature_ids:
            return {"error": "No literature IDs provided"}
        
        entries = []
        for lit_id in literature_ids:
            if lit_id < 1 or lit_id > len(literature_database):
                continue
            entries.append(literature_database[lit_id - 1])
        
        if not entries:
            return {"error": "No valid literature entries found"}
        
        model = get_gemini_model()
        
        literature_text = ""
        for entry in entries:
            literature_text += f"""
            Title: {entry.title}
            Authors: {', '.join(entry.authors)} ({entry.year})
            Abstract: {entry.abstract}
            Keywords: {', '.join(entry.keywords)}
            ---
            """
        
        prompt = f"""
        Topic: "{topic}"
        
        Literature Entries to Summarize:
        {literature_text}
        
        Please provide a comprehensive literature summary focused on the topic "{topic}".
        Include:
        1. An overview of the current state of research
        2. Key findings and contributions from each paper
        3. Common themes and patterns
        4. Gaps in the literature
        5. Future research directions
        
        Return your response as JSON with format:
        {
            "summary": "<comprehensive summary>",
            "key_findings": ["<finding1>", "<finding2>"],
            "common_themes": ["<theme1>", "<theme2>"],
            "research_gaps": ["<gap1>", "<gap2>"],
            "future_directions": ["<direction1>", "<direction2>"]
        }
        """
        
        response = model.generate_content(prompt)
        
        try:
            result = json.loads(response.text.strip())
            return result
        except json.JSONDecodeError:
            return {
                "summary": response.text,
                "key_findings": [],
                "common_themes": [],
                "research_gaps": [],
                "future_directions": []
            }
            
    except Exception as e:
        return {"error": str(e)}

@app.tool()
def list_literature() -> dict:
    """
    List all literature entries in the database.
    
    Returns:
        On success: {"entries": <list of all literature entries>}
        On error: {"error": <error message>}
    """
    try:
        entries = [entry.to_dict() for entry in literature_database]
        return {
            "entries": entries,
            "total_count": len(entries)
        }
    except Exception as e:
        return {"error": str(e)}

@app.tool()
def get_literature_details(literature_id: int) -> dict:
    """
    Get detailed information about a specific literature entry.
    
    Args:
        literature_id: ID of the literature entry
    
    Returns:
        On success: {"entry": <literature entry details>}
        On error: {"error": <error message>}
    """
    try:
        if literature_id < 1 or literature_id > len(literature_database):
            return {"error": "Literature entry not found"}
        
        entry = literature_database[literature_id - 1]
        return {"entry": entry.to_dict()}
        
    except Exception as e:
        return {"error": str(e)}

def main():
    """Main entry point for the application."""
    # Check if API key is set
    if not os.getenv('GEMINI_API_KEY'):
        print("Warning: GEMINI_API_KEY not found in environment variables.")
        print("Please set your Gemini API key in a .env file or environment variable.")
    
    app.run(transport=TRANSPORT)

if __name__ == "__main__":
    main()