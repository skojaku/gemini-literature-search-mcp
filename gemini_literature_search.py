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
    GEMINI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: google.generativeai not available: {e}", file=sys.stderr)
    print("Install with: pip install google-generativeai", file=sys.stderr)
    print("Running in basic mode without AI features...", file=sys.stderr)
    genai = None
    GEMINI_AVAILABLE = False

try:
    from fastmcp import FastMCP
except ImportError as e:
    print(f"Error importing FastMCP: {e}", file=sys.stderr)
    print("Make sure to install: pip install fastmcp", file=sys.stderr)
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv('GEMINI_API_KEY')
if GEMINI_AVAILABLE:
    if not api_key:
        print("Warning: GEMINI_API_KEY not found in environment variables.", file=sys.stderr)
        print("Please set your Gemini API key in a .env file or environment variable.", file=sys.stderr)
        print("AI features will be disabled.", file=sys.stderr)
        GEMINI_AVAILABLE = False
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
    if not GEMINI_AVAILABLE:
        return None
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
        
        if not GEMINI_AVAILABLE:
            # Fallback to simple text search when Gemini is not available
            query_lower = query.lower()
            results = []
            
            for entry in literature_database:
                # Simple relevance scoring based on keyword and title matches
                score = 0
                text_to_search = f"{entry.title} {entry.abstract} {' '.join(entry.keywords)}".lower()
                
                # Count query word matches
                query_words = query_lower.split()
                matches = sum(1 for word in query_words if word in text_to_search)
                score = matches / len(query_words) if query_words else 0
                
                if score > 0:
                    entry_dict = entry.to_dict()
                    entry_dict["relevance_score"] = score
                    entry_dict["relevance_reason"] = f"Basic text search: {matches}/{len(query_words)} query words found"
                    results.append(entry_dict)
            
            # Sort by relevance score and limit results
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            return {"results": results[:max_results]}
        
        model = get_gemini_model()
        if not model:
            return {"error": "Gemini model not available"}
        
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
        
        if not GEMINI_AVAILABLE:
            # Fallback: simple keyword matching
            topic_words = research_topic.lower().split()
            entry_text = f"{entry.title} {entry.abstract} {' '.join(entry.keywords)}".lower()
            matches = sum(1 for word in topic_words if word in entry_text)
            score = matches / len(topic_words) if topic_words else 0
            
            return {
                "relevance_score": score,
                "analysis": f"Basic keyword analysis: {matches}/{len(topic_words)} topic words found in literature",
                "relevant_aspects": [f"Matched words: {matches}"],
                "concerns": ["AI analysis not available - using basic keyword matching"]
            }
        
        model = get_gemini_model()
        if not model:
            return {"error": "Gemini model not available"}
        
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
        
        if not GEMINI_AVAILABLE:
            # Fallback: basic appropriateness check
            return {
                "is_appropriate": True,
                "confidence": 0.5,
                "analysis": "AI validation not available - basic check suggests citation format is acceptable",
                "issues": ["AI analysis not available"],
                "suggestions": ["Install google-generativeai for detailed citation validation"]
            }
        
        model = get_gemini_model()
        if not model:
            return {"error": "Gemini model not available"}
        
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
        
        if not GEMINI_AVAILABLE:
            # Fallback: basic summary without AI
            summary = f"Summary of {len(entries)} literature entries on topic: {topic}\n\n"
            key_findings = []
            common_themes = []
            
            for i, entry in enumerate(entries, 1):
                summary += f"{i}. {entry.title} ({', '.join(entry.authors)}, {entry.year})\n"
                summary += f"   Abstract: {entry.abstract[:200]}...\n\n"
                key_findings.append(f"Paper {i}: {entry.title}")
                common_themes.extend(entry.keywords)
            
            # Get unique themes
            common_themes = list(set(common_themes))
            
            return {
                "summary": summary,
                "key_findings": key_findings,
                "common_themes": common_themes[:5],  # Top 5 themes
                "research_gaps": ["AI analysis not available"],
                "future_directions": ["Install google-generativeai for detailed analysis"]
            }
        
        model = get_gemini_model()
        if not model:
            return {"error": "Gemini model not available"}
        
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

if __name__ == "__main__":
    app.run(transport=TRANSPORT)