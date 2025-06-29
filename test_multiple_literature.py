#!/usr/bin/env python3
"""
Test script to verify that literature functions return multiple works
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from dotenv import load_dotenv
    from gemini_literature_search import find_supporting_literature, find_unsupporting_literature
    load_dotenv()
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure to install dependencies: uv sync")
    sys.exit(1)

def test_multiple_supporting_literature():
    """Test that supporting literature returns multiple papers"""
    print("Testing supporting literature for multiple papers...")
    
    paragraph = "Machine learning significantly improves medical diagnosis accuracy. Deep learning models can detect cancer with high precision."
    result = find_supporting_literature(paragraph)
    
    if result.get('success'):
        print("✓ Supporting literature function succeeded")
        
        for i, sentence_analysis in enumerate(result.get('sentence_analysis', []), 1):
            sentence = sentence_analysis.get('sentence_text', 'Unknown')
            papers = sentence_analysis.get('supporting_literature', [])
            paper_count = len(papers)
            
            print(f"\nSentence {i}: {sentence}")
            print(f"Supporting papers found: {paper_count}")
            
            if paper_count >= 5:
                print("✓ Good coverage: 5+ papers found")
            elif paper_count >= 3:
                print("○ Moderate coverage: 3-4 papers found")
            else:
                print("⚠ Limited coverage: <3 papers found")
            
            # Show first few papers
            for j, paper in enumerate(papers[:3], 1):
                title = paper.get('title', 'No title')[:80]
                year = paper.get('year', 'No year')
                print(f"  {j}. {title}... ({year})")
                
        total_papers = sum(len(s.get('supporting_literature', [])) for s in result.get('sentence_analysis', []))
        print(f"\nTotal supporting papers found: {total_papers}")
        
    else:
        print("✗ Supporting literature function failed")
        print("Error:", result.get('message', 'Unknown error'))
    
    return result

def test_multiple_unsupporting_literature():
    """Test that unsupporting literature returns multiple papers"""
    print("\n" + "="*60)
    print("Testing unsupporting literature for multiple papers...")
    
    paragraph = "AI is 100% accurate in medical diagnosis. Machine learning completely eliminates all human error in healthcare."
    result = find_unsupporting_literature(paragraph)
    
    if result.get('success'):
        print("✓ Unsupporting literature function succeeded")
        
        for i, sentence_analysis in enumerate(result.get('sentence_analysis', []), 1):
            sentence = sentence_analysis.get('sentence_text', 'Unknown')
            papers = sentence_analysis.get('contradicting_literature', [])
            paper_count = len(papers)
            
            print(f"\nSentence {i}: {sentence}")
            print(f"Contradicting papers found: {paper_count}")
            
            if paper_count >= 5:
                print("✓ Good coverage: 5+ papers found")
            elif paper_count >= 3:
                print("○ Moderate coverage: 3-4 papers found")
            else:
                print("⚠ Limited coverage: <3 papers found")
            
            # Show first few papers
            for j, paper in enumerate(papers[:3], 1):
                title = paper.get('title', 'No title')[:80]
                year = paper.get('year', 'No year')
                print(f"  {j}. {title}... ({year})")
                
        total_papers = sum(len(s.get('contradicting_literature', [])) for s in result.get('sentence_analysis', []))
        print(f"\nTotal contradicting papers found: {total_papers}")
        
    else:
        print("✗ Unsupporting literature function failed")
        print("Error:", result.get('message', 'Unknown error'))
    
    return result

def main():
    """Run tests for multiple literature sources"""
    # Check if API key is available
    if not os.getenv('GEMINI_API_KEY'):
        print("ERROR: GEMINI_API_KEY not found in environment variables.")
        print("Please set your API key in .env file or environment.")
        sys.exit(1)
    
    print("Testing literature functions for multiple paper returns...")
    print(f"API Key found: {'Yes' if os.getenv('GEMINI_API_KEY') else 'No'}")
    print("="*60)
    
    # Test supporting literature
    supporting_result = test_multiple_supporting_literature()
    
    # Test unsupporting literature
    unsupporting_result = test_multiple_unsupporting_literature()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Calculate totals
    supporting_success = supporting_result.get('success', False)
    unsupporting_success = unsupporting_result.get('success', False)
    
    if supporting_success:
        total_supporting = sum(len(s.get('supporting_literature', [])) for s in supporting_result.get('sentence_analysis', []))
        sentences_supporting = len(supporting_result.get('sentence_analysis', []))
        avg_supporting = total_supporting / sentences_supporting if sentences_supporting > 0 else 0
        print(f"Supporting Literature: {total_supporting} papers total, avg {avg_supporting:.1f} per sentence")
    else:
        print("Supporting Literature: FAILED")
    
    if unsupporting_success:
        total_unsupporting = sum(len(s.get('contradicting_literature', [])) for s in unsupporting_result.get('sentence_analysis', []))
        sentences_unsupporting = len(unsupporting_result.get('sentence_analysis', []))
        avg_unsupporting = total_unsupporting / sentences_unsupporting if sentences_unsupporting > 0 else 0
        print(f"Unsupporting Literature: {total_unsupporting} papers total, avg {avg_unsupporting:.1f} per sentence")
    else:
        print("Unsupporting Literature: FAILED")
    
    # Recommendations
    print("\nRecommendations:")
    if supporting_success and unsupporting_success:
        if avg_supporting >= 5 and avg_unsupporting >= 5:
            print("✓ Excellent: Both functions returning 5+ papers per sentence")
        elif avg_supporting >= 3 and avg_unsupporting >= 3:
            print("○ Good: Both functions returning 3+ papers per sentence")
        else:
            print("⚠ Consider optimizing prompts for more comprehensive literature coverage")
    else:
        print("⚠ Fix function failures before optimizing for multiple papers")

if __name__ == "__main__":
    main()