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
    from google import genai as genai_client
    from google.genai import types
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
client = None

if not api_key:
    print("Warning: GEMINI_API_KEY not found in environment variables.", file=sys.stderr)
    print("Please set your Gemini API key in a .env file or environment variable.", file=sys.stderr)
else:
    genai.configure(api_key=api_key)
    # Also configure the new genai client for grounding
    try:
        client = genai_client.Client(api_key=api_key)
    except Exception as e:
        print(f"Warning: Could not initialize grounding client: {e}", file=sys.stderr)
        client = None

# Create MCP Server
app = FastMCP(
    title="Gemini Literature Search",
    description="A server for academic literature search and validation using Google Gemini",
    version="1.0.0",
    dependencies=["google-generativeai", "google-genai", "requests", "python-dotenv"],
)

TRANSPORT = "sse"

def get_gemini_model(model_name="gemini-1.5-flash"):
    """Get Gemini model instance"""
    return genai.GenerativeModel(model_name)

def search_with_grounding(query, model_name="gemini-2.0-flash-exp"):
    """Search using Gemini with Google Search grounding"""
    if not client:
        raise Exception("Grounding client not available. Check API key configuration.")
    
    # Define the grounding tool
    grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )
    
    # Configure generation settings
    config = types.GenerateContentConfig(
        tools=[grounding_tool]
    )
    
    # Make the request
    response = client.models.generate_content(
        model=model_name,
        contents=query,
        config=config,
    )
    
    return response


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

@app.tool()
def find_relevant_literature(paragraph: str, model: str = "gemini-2.0-flash-exp") -> dict:
    """
    Find relevant literature for each sentence in a given paragraph without bias toward supporting or contradicting.
    
    Args:
        paragraph: The paragraph text to analyze sentence by sentence
        model: Gemini model to use (default: "gemini-2.0-flash-exp")
    
    Returns:
        On success: {"sentence_analysis": <list of sentences with relevant literature>}
        On error: {"error": <error message>}
    
    Examples:
        >>> find_relevant_literature("Machine learning is used in healthcare. Neural networks process medical images.")
        {'sentence_analysis': [{'sentence': 'Machine learning is used...', 'relevant_literature': [...]}]}
    """
    try:
        search_query = f"""
        For the following paragraph, analyze each sentence and find academic literature that is RELEVANT to each topic or concept mentioned:

        Paragraph: "{paragraph}"

        For each sentence:
        1. Break down the paragraph into individual sentences
        2. Identify the main topics, concepts, or research areas mentioned
        3. Search for academic papers that are relevant to these topics
        4. Include foundational papers, recent research, review articles, and key studies
        5. Focus on relevance rather than supporting or contradicting the claims
        6. Include diverse perspectives and comprehensive coverage of the topic

        Return your response as JSON with format:
        {{
            "sentence_analysis": [
                {{
                    "sentence_number": 1,
                    "sentence_text": "<sentence>",
                    "key_topics": ["<topic1>", "<topic2>", "<topic3>"],
                    "relevant_literature": [
                        {{
                            "title": "<paper title>",
                            "authors": ["<author1>", "<author2>"],
                            "year": <year>,
                            "relevance_type": "<foundational/recent_research/review/methodology/application>",
                            "topic_coverage": "<which topics from the sentence this paper covers>",
                            "key_contribution": "<what this paper contributes to understanding the topic>",
                            "relevance_score": <0.0-1.0>,
                            "doi_or_url": "<DOI or URL if available>"
                        }}
                    ]
                }}
            ]
        }}
        """
        
        response = search_with_grounding(search_query, model)
        
        # Try to extract JSON from response
        try:
            import re
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                result_data = json.loads(json_match.group())
                return result_data
            else:
                return {
                    "sentence_analysis": [],
                    "raw_response": response.text,
                    "note": "Could not parse JSON response, see raw_response"
                }
        except json.JSONDecodeError:
            return {
                "sentence_analysis": [],
                "raw_response": response.text,
                "note": "Could not parse JSON response, see raw_response"
            }
            
    except Exception as e:
        return {"error": str(e)}

@app.tool()
def find_supporting_literature(paragraph: str, model: str = "gemini-2.0-flash-exp") -> dict:
    """
    Find supporting literature for each sentence in a given paragraph.
    
    Args:
        paragraph: The paragraph text to analyze sentence by sentence
        model: Gemini model to use (default: "gemini-2.0-flash-exp")
    
    Returns:
        On success: {"sentence_analysis": <list of sentences with supporting literature>}
        On error: {"error": <error message>}
    
    Examples:
        >>> find_supporting_literature("AI improves medical diagnosis. Machine learning reduces errors.")
        {'sentence_analysis': [{'sentence': 'AI improves medical diagnosis', 'supporting_literature': [...]}]}
    """
    try:
        search_query = f"""
        For the following paragraph, analyze each sentence and find academic literature that SUPPORTS each claim:

        Paragraph: "{paragraph}"

        For each sentence:
        1. Break down the paragraph into individual sentences
        2. For each sentence, search for academic papers that support the claim
        3. Find peer-reviewed studies, research papers, or authoritative sources
        4. Include specific findings or data that validate each statement

        Return your response as JSON with format:
        {{
            "sentence_analysis": [
                {{
                    "sentence_number": 1,
                    "sentence_text": "<sentence>",
                    "supporting_literature": [
                        {{
                            "title": "<paper title>",
                            "authors": ["<author1>", "<author2>"],
                            "year": <year>,
                            "key_finding": "<specific finding that supports the sentence>",
                            "relevance_score": <0.0-1.0>,
                            "doi_or_url": "<DOI or URL if available>"
                        }}
                    ]
                }}
            ]
        }}
        """
        
        response = search_with_grounding(search_query, model)
        
        # Try to extract JSON from response
        try:
            # Look for JSON in the response
            import re
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                result_data = json.loads(json_match.group())
                return result_data
            else:
                # If no JSON found, return structured response
                return {
                    "sentence_analysis": [],
                    "raw_response": response.text,
                    "note": "Could not parse JSON response, see raw_response"
                }
        except json.JSONDecodeError:
            return {
                "sentence_analysis": [],
                "raw_response": response.text,
                "note": "Could not parse JSON response, see raw_response"
            }
            
    except Exception as e:
        return {"error": str(e)}

@app.tool()
def find_unsupporting_literature(paragraph: str, model: str = "gemini-2.0-flash-exp") -> dict:
    """
    Find literature that contradicts or challenges each sentence in a given paragraph.
    
    Args:
        paragraph: The paragraph text to analyze sentence by sentence
        model: Gemini model to use (default: "gemini-2.0-flash-exp")
    
    Returns:
        On success: {"sentence_analysis": <list of sentences with contradicting literature>}
        On error: {"error": <error message>}
    
    Examples:
        >>> find_unsupporting_literature("AI is 100% accurate in diagnosis. Automation eliminates all human error.")
        {'sentence_analysis': [{'sentence': 'AI is 100% accurate...', 'contradicting_literature': [...]}]}
    """
    try:
        search_query = f"""
        For the following paragraph, analyze each sentence and find academic literature that CONTRADICTS, CHALLENGES, or shows LIMITATIONS of each claim:

        Paragraph: "{paragraph}"

        For each sentence:
        1. Break down the paragraph into individual sentences
        2. For each sentence, search for academic papers that contradict or challenge the claim
        3. Look for studies showing limitations, failures, or alternative findings
        4. Include research that presents counter-evidence or different conclusions
        5. Find papers highlighting potential problems or exceptions

        Return your response as JSON with format:
        {{
            "sentence_analysis": [
                {{
                    "sentence_number": 1,
                    "sentence_text": "<sentence>",
                    "contradicting_literature": [
                        {{
                            "title": "<paper title>",
                            "authors": ["<author1>", "<author2>"],
                            "year": <year>,
                            "counter_finding": "<specific finding that contradicts the sentence>",
                            "limitation_highlighted": "<what limitation or problem is shown>",
                            "relevance_score": <0.0-1.0>,
                            "doi_or_url": "<DOI or URL if available>"
                        }}
                    ]
                }}
            ]
        }}
        """
        
        response = search_with_grounding(search_query, model)
        
        # Try to extract JSON from response
        try:
            import re
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                result_data = json.loads(json_match.group())
                return result_data
            else:
                return {
                    "sentence_analysis": [],
                    "raw_response": response.text,
                    "note": "Could not parse JSON response, see raw_response"
                }
        except json.JSONDecodeError:
            return {
                "sentence_analysis": [],
                "raw_response": response.text,
                "note": "Could not parse JSON response, see raw_response"
            }
            
    except Exception as e:
        return {"error": str(e)}

@app.tool()
def comprehensive_fact_check(paragraph: str, model: str = "gemini-2.0-flash-exp") -> dict:
    """
    Perform comprehensive fact-checking by finding both supporting and contradicting literature for each sentence.
    
    Args:
        paragraph: The paragraph text to fact-check sentence by sentence
        model: Gemini model to use (default: "gemini-2.0-flash-exp")
    
    Returns:
        On success: {"fact_check_analysis": <comprehensive analysis with both sides>}
        On error: {"error": <error message>}
    
    Examples:
        >>> comprehensive_fact_check("AI improves diagnosis accuracy by 50%. It completely replaces human doctors.")
        {'fact_check_analysis': [{'sentence': '...', 'supporting': [...], 'contradicting': [...], 'verdict': '...'}]}
    """
    try:
        search_query = f"""
        Perform a comprehensive fact-check of the following paragraph by analyzing each sentence for both supporting and contradicting evidence:

        Paragraph: "{paragraph}"

        For each sentence:
        1. Break down the paragraph into individual sentences
        2. Search for academic literature that SUPPORTS the claim
        3. Search for academic literature that CONTRADICTS or CHALLENGES the claim
        4. Provide a balanced assessment with evidence from both sides
        5. Give a fact-check verdict (Supported/Partially Supported/Contradicted/Insufficient Evidence)

        Return your response as JSON with format:
        {{
            "fact_check_analysis": [
                {{
                    "sentence_number": 1,
                    "sentence_text": "<sentence>",
                    "supporting_evidence": [
                        {{
                            "title": "<paper title>",
                            "authors": ["<author1>", "<author2>"],
                            "year": <year>,
                            "supporting_finding": "<finding that supports>",
                            "doi_or_url": "<DOI or URL if available>"
                        }}
                    ],
                    "contradicting_evidence": [
                        {{
                            "title": "<paper title>",
                            "authors": ["<author1>", "<author2>"],
                            "year": <year>,
                            "contradicting_finding": "<finding that contradicts>",
                            "doi_or_url": "<DOI or URL if available>"
                        }}
                    ],
                    "verdict": "<Supported/Partially Supported/Contradicted/Insufficient Evidence>",
                    "confidence": <0.0-1.0>,
                    "analysis": "<balanced assessment of the evidence>"
                }}
            ]
        }}
        """
        
        response = search_with_grounding(search_query, model)
        
        # Try to extract JSON from response
        try:
            import re
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                result_data = json.loads(json_match.group())
                return result_data
            else:
                return {
                    "fact_check_analysis": [],
                    "raw_response": response.text,
                    "note": "Could not parse JSON response, see raw_response"
                }
        except json.JSONDecodeError:
            return {
                "fact_check_analysis": [],
                "raw_response": response.text,
                "note": "Could not parse JSON response, see raw_response"
            }
            
    except Exception as e:
        return {"error": str(e)}

@app.tool()
def ask_gemini(query: str, use_search: bool = True, model: str = "gemini-2.0-flash-exp") -> dict:
    """
    Delegate any task directly to Gemini with optional Google Search grounding.
    
    Args:
        query: Any question or task you want Gemini to handle
        use_search: Whether to use Google Search grounding (default: True)
        model: Gemini model to use (default: "gemini-2.0-flash-exp")
    
    Returns:
        On success: {"response": <Gemini's response>, "sources": <list of sources if grounded>}
        On error: {"error": <error message>}
    
    Examples:
        >>> ask_gemini("What are the latest developments in quantum computing?")
        {'response': 'Recent developments in quantum computing include...', 'sources': [...]}
        >>> ask_gemini("Explain machine learning in simple terms", use_search=False)
        {'response': 'Machine learning is...', 'sources': []}
        >>> ask_gemini("Analyze this research trend", model="gemini-1.5-pro")
        {'response': 'Based on current research...', 'sources': [...]}
    """
    try:
        if use_search:
            # Use grounded search for real-time information
            response = search_with_grounding(query, model)
            
            # Extract information from grounded response
            result = {
                "response": response.text,
                "sources": [],
                "grounded": True
            }
            
            # Add grounding metadata if available
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                    grounding = candidate.grounding_metadata
                    
                    # Add search queries used
                    if hasattr(grounding, 'web_search_queries'):
                        result["web_search_queries"] = grounding.web_search_queries
                    
                    # Add source chunks
                    if hasattr(grounding, 'grounding_chunks'):
                        for chunk in grounding.grounding_chunks:
                            if hasattr(chunk, 'web'):
                                result["sources"].append({
                                    "title": chunk.web.title if hasattr(chunk.web, 'title') else "Unknown",
                                    "url": chunk.web.uri if hasattr(chunk.web, 'uri') else "Unknown"
                                })
            
            return result
        else:
            # Use regular Gemini without grounding
            gemini_model = get_gemini_model(model)
            response = gemini_model.generate_content(query)
            
            return {
                "response": response.text,
                "sources": [],
                "grounded": False
            }
            
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    app.run(transport=TRANSPORT)