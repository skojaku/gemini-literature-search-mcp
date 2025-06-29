#!/usr/bin/env python3
"""
Test natural language parsing
"""

import re

def test_natural_language_parsing():
    """Test parsing of natural language response"""
    
    # Sample response like what Gemini might return
    sample_response = """
    Okay, here's a comprehensive analysis of the paragraph with supporting literature.

    **Paragraph:** "When biologist Jennifer Doudna explored CRISPR applications, her breakthrough required traversing from RNA structural biology to genome editing."

    **Sentence 1:** "When biologist Jennifer Doudna explored CRISPR applications, her breakthrough required traversing from RNA structural biology to genome editing."

    Supporting Literature:

    • **CRISPR-Cas9 Structures and Mechanisms** (2014)
      Authors: Martin Jinek, Krzysztof Chylinski, Ines Fonfara, Michael Hauer, Jennifer A. Doudna, Emmanuelle Charpentier
      Abstract: This foundational paper describes the structural and mechanistic details of the CRISPR-Cas9 system
      Assessment: This paper directly supports the sentence by showing Doudna's transition from RNA structural work to genome editing applications
      Relevance: 0.95

    • **A Programmable Dual-RNA–Guided DNA Endonuclease in Adaptive Bacterial Immunity** (2012) 
      Authors: Jinek et al.
      Abstract: Describes the programmable nature of CRISPR systems for genome editing
      Assessment: Shows the trajectory from understanding RNA-guided systems to genome editing
      Relevance: 0.90

    • **RNA-guided human genome engineering via Cas9** (2013)
      Authors: Mali et al.
      Abstract: Demonstrates application of CRISPR to human genome editing
      Assessment: Illustrates the progression from basic RNA biology to therapeutic applications
      Relevance: 0.85
    """
    
    print("Testing natural language parsing...")
    print(f"Sample response length: {len(sample_response)}")
    
    # Test sentence pattern matching
    sentence_patterns = re.findall(
        r'(?:\*\*)?(?:Sentence|SENTENCE)\s*(\d+)(?:\*\*)?[:\.]?\s*[*]*\s*(.*?)(?=(?:\*\*)?(?:Sentence|SENTENCE)\s*\d+|$)', 
        sample_response, 
        re.DOTALL | re.IGNORECASE
    )
    
    print(f"Found {len(sentence_patterns)} sentences")
    for i, (num, content) in enumerate(sentence_patterns):
        print(f"Sentence {num}: {content[:100]}...")
    
    # Test paper extraction
    if sentence_patterns:
        sentence_content = sentence_patterns[0][1]
        
        # Look for bullet points
        paper_sections = re.split(r'\n(?=[-•*]\s)', sentence_content)
        print(f"\nFound {len(paper_sections)} paper sections")
        
        for i, section in enumerate(paper_sections):
            if i == 0:  # Skip first section which might not be a paper
                continue
                
            lines = section.strip().split('\n')
            if lines:
                first_line = lines[0].strip()
                title = re.sub(r'^[-•*]\s*\*\*?', '', first_line).strip()
                title = re.sub(r'\*\*?.*$', '', title).strip()
                
                print(f"Paper {i}: {title}")
                
                # Look for authors
                content = '\n'.join(lines)
                author_match = re.search(r'[Aa]uthors?[:\s]*([^\n]+)', content)
                if author_match:
                    print(f"  Authors: {author_match.group(1).strip()}")
                
                # Look for year
                year_match = re.search(r'\((\d{4})\)', content)
                if year_match:
                    print(f"  Year: {year_match.group(1)}")

if __name__ == "__main__":
    test_natural_language_parsing()