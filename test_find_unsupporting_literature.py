#!/usr/bin/env python3
"""
Test script for find_unsupporting_literature function
"""

import os
import sys
import json
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from dotenv import load_dotenv
    from gemini_literature_search import find_unsupporting_literature
    load_dotenv()
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure to install dependencies: uv sync")
    sys.exit(1)

def print_result(test_name, result):
    """Pretty print test results"""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")
    
    # Print in a structured way
    if isinstance(result, dict):
        print(f"Success: {result.get('success', 'Unknown')}")
        
        if result.get('success'):
            print(f"Sentence Analysis Count: {len(result.get('sentence_analysis', []))}")
            for i, analysis in enumerate(result.get('sentence_analysis', []), 1):
                print(f"\nSentence {i}: {analysis.get('sentence_text', 'N/A')}")
                print(f"Contradicting Literature Count: {len(analysis.get('contradicting_literature', []))}")
                for j, lit in enumerate(analysis.get('contradicting_literature', [])[:2], 1):  # Show first 2
                    print(f"  {j}. {lit.get('title', 'No title')} ({lit.get('year', 'No year')})")
                    if 'counter_finding' in lit:
                        print(f"     Counter-finding: {lit['counter_finding'][:100]}...")
        else:
            print(f"Error Details: {result.get('error_details', 'No details')}")
            print(f"Message: {result.get('message', 'No message')}")
            
        if 'debug_info' in result:
            print(f"\nDebug Info:")
            for key, value in result['debug_info'].items():
                print(f"  {key}: {value}")
                
        # Show raw response length if available
        if 'raw_response' in result:
            raw_len = len(str(result['raw_response']))
            print(f"Raw Response Length: {raw_len} chars")
            if raw_len < 300:
                print(f"Raw Response Preview: {str(result['raw_response'])[:200]}...")
    else:
        print(f"Unexpected result type: {type(result)}")
        print(f"Result: {result}")

def test_overstated_claims():
    """Test with overstated claims that should have contradicting evidence"""
    paragraph = "AI is 100% accurate in medical diagnosis. Machine learning completely eliminates all human error in healthcare."
    
    print("Testing overstated claims...")
    result = find_unsupporting_literature(paragraph)
    print_result("Overstated Claims", result)
    return result

def test_controversial_statements():
    """Test with controversial statements"""
    paragraph = "Traditional doctors will become completely obsolete within 2 years. AI can diagnose any disease better than any human specialist."
    
    print("Testing controversial statements...")
    result = find_unsupporting_literature(paragraph)
    print_result("Controversial Statements", result)
    return result

def test_absolute_claims():
    """Test with absolute claims that likely have exceptions"""
    paragraph = "Deep learning models never make mistakes. Automated diagnosis systems are always more reliable than human judgment."
    
    print("Testing absolute claims...")
    result = find_unsupporting_literature(paragraph)
    print_result("Absolute Claims", result)
    return result

def test_single_overstated_claim():
    """Test with single overstated claim"""
    paragraph = "AI has solved all problems in medical imaging and no further research is needed."
    
    print("Testing single overstated claim...")
    result = find_unsupporting_literature(paragraph)
    print_result("Single Overstated Claim", result)
    return result

def test_reasonable_claims():
    """Test with reasonable claims (should find fewer contradictions)"""
    paragraph = "Machine learning can assist doctors in diagnosis. AI shows promise in medical imaging analysis."
    
    print("Testing reasonable claims...")
    result = find_unsupporting_literature(paragraph)
    print_result("Reasonable Claims", result)
    return result

def test_different_model():
    """Test with different model"""
    paragraph = "Artificial intelligence will replace all radiologists by next year."
    
    print("Testing with different model (gemini-1.5-pro)...")
    result = find_unsupporting_literature(paragraph, model="gemini-1.5-pro")
    print_result("Different Model", result)
    return result

def test_empty_paragraph():
    """Test with empty paragraph"""
    paragraph = ""
    
    print("Testing empty paragraph...")
    result = find_unsupporting_literature(paragraph)
    print_result("Empty Paragraph", result)
    return result

def main():
    """Run all tests"""
    # Check if API key is available
    if not os.getenv('GEMINI_API_KEY'):
        print("ERROR: GEMINI_API_KEY not found in environment variables.")
        print("Please set your API key in .env file or environment.")
        sys.exit(1)
    
    print("Starting find_unsupporting_literature tests...")
    print(f"API Key found: {'Yes' if os.getenv('GEMINI_API_KEY') else 'No'}")
    
    tests = [
        test_overstated_claims,
        test_controversial_statements,
        test_absolute_claims,
        test_single_overstated_claim,
        test_reasonable_claims,
        test_different_model,
        test_empty_paragraph,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"\nERROR in {test_func.__name__}: {e}")
            results.append((test_func.__name__, {"error": str(e)}))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    successful_tests = 0
    for test_name, result in results:
        if isinstance(result, dict):
            success = result.get('success', False)
            status = "✓ SUCCESS" if success else "✗ FAILED"
            successful_tests += 1 if success else 0
        else:
            status = "✗ ERROR"
        
        print(f"{status:<12} {test_name}")
    
    print(f"\nOverall: {successful_tests}/{len(tests)} tests successful")
    
    # Save detailed results to file
    results_file = "test_unsupporting_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Detailed results saved to: {results_file}")

if __name__ == "__main__":
    main()