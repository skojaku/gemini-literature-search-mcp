#!/usr/bin/env python3
"""
Simple natural language parser for testing
"""

import re

def parse_natural_language_response(response_text):
    """Parse the natural language response from Gemini"""
    
    sentence_analysis = []
    
    # Split by sentence headers
    sentence_sections = re.split(r'\*\*Sentence \d+:\*\*', response_text)
    
    # Skip the first section (usually introduction)
    for i, section in enumerate(sentence_sections[1:], 1):
        
        sentence_obj = {
            "sentence_number": i,
            "sentence_text": "",
            "supporting_literature": []
        }
        
        # Extract sentence text (usually in quotes)
        sentence_match = re.search(r'"([^"]+)"', section)
        if sentence_match:
            sentence_obj["sentence_text"] = sentence_match.group(1)
        
        # Split by paper markers (* **Paper N:**)
        paper_sections = re.split(r'\*\s+\*\*Paper \d+:\*\*', section)
        
        for paper_section in paper_sections[1:]:  # Skip first empty section
            
            paper = {
                "title": "",
                "authors": [],
                "year": None,
                "abstract": "",
                "gemini_assessment": "",
                "relevance_score": 0.8,
                "doi_or_url": ""
            }
            
            # Extract title
            title_match = re.search(r'\*\*Title:\*\*\s*([^\n]+)', paper_section)
            if title_match:
                paper["title"] = title_match.group(1).strip()
            
            # Extract authors
            authors_match = re.search(r'\*\*Authors:\*\*\s*([^\n]+)', paper_section)
            if authors_match:
                authors_text = authors_match.group(1).strip()
                # Split by common separators
                paper["authors"] = [a.strip() for a in re.split(r'[,&]|and\s', authors_text) if a.strip()]
            
            # Extract year
            year_match = re.search(r'\*\*Publication Year:\*\*\s*(\d{4})', paper_section)
            if year_match:
                paper["year"] = int(year_match.group(1))
            
            # Extract abstract
            abstract_match = re.search(r'\*\*Abstract:\*\*\s*(.*?)(?=\*\*|$)', paper_section, re.DOTALL)
            if abstract_match:
                paper["abstract"] = abstract_match.group(1).strip()
            
            # Extract assessment
            assessment_match = re.search(r'\*\*Assessment:\*\*\s*(.*?)(?=\*\*|$)', paper_section, re.DOTALL)
            if assessment_match:
                paper["gemini_assessment"] = assessment_match.group(1).strip()
            
            # Extract relevance score
            relevance_match = re.search(r'\*\*Relevance:\*\*\s*([0-9.]+)', paper_section)
            if relevance_match:
                try:
                    paper["relevance_score"] = float(relevance_match.group(1))
                except ValueError:
                    pass
            
            # Extract DOI
            doi_match = re.search(r'\*\*DOI:\*\*\s*([^\n]+)', paper_section)
            if doi_match:
                paper["doi_or_url"] = doi_match.group(1).strip()
            
            if paper["title"]:  # Only add if we found a title
                sentence_obj["supporting_literature"].append(paper)
        
        if sentence_obj["supporting_literature"]:  # Only add if we found papers
            sentence_analysis.append(sentence_obj)
    
    return {
        "success": True if sentence_analysis else False,
        "sentence_analysis": sentence_analysis,
        "note": "Parsed from natural language response"
    }

if __name__ == "__main__":
    # Test with sample response
    sample = """
    **Sentence 1:** "When biologist Jennifer Doudna explored CRISPR applications..."

    *   **Paper 1:**
        *   **Title:** A Programmable Dual-RNA–Guided DNA Endonuclease
        *   **Authors:** Jinek, M., Chylinski, K., Doudna, J.A.
        *   **Publication Year:** 2012
        *   **Abstract:** This foundational paper describes CRISPR-Cas9
        *   **Assessment:** Shows Doudna's involvement in CRISPR research
        *   **Relevance:** 1.0
        *   **DOI:** 10.1126/science.1429977

    **Sentence 2:** "Her breakthrough required traversing..."
    
    *   **Paper 1:**
        *   **Title:** Structural Mechanism of RNA-Guided DNA Interrogation
        *   **Authors:** Sternberg, S.H., Doudna, J.A.
        *   **Publication Year:** 2014
        *   **Abstract:** Crystal structure of Cas9 complex
        *   **Assessment:** Links RNA structure to genome editing
        *   **Relevance:** 1.0
        *   **DOI:** 10.1038/nature13844
    """
    
    result = parse_natural_language_response(sample)
    print("Success:", result["success"])
    print("Sentences found:", len(result["sentence_analysis"]))
    for sentence in result["sentence_analysis"]:
        print(f"Sentence {sentence['sentence_number']}: {len(sentence['supporting_literature'])} papers")