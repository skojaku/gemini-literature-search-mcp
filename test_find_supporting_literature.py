#!/usr/bin/env python3
"""
Test script for find_supporting_literature function
"""

import os
import sys
import json
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from dotenv import load_dotenv
    from gemini_literature_search import find_supporting_literature
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
                print(f"Supporting Literature Count: {len(analysis.get('supporting_literature', []))}")
                for j, lit in enumerate(analysis.get('supporting_literature', [])[:2], 1):  # Show first 2
                    print(f"  {j}. {lit.get('title', 'No title')} ({lit.get('year', 'No year')})")
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
            if raw_len < 200:
                print(f"Raw Response Preview: {str(result['raw_response'])[:200]}...")
    else:
        print(f"Unexpected result type: {type(result)}")
        print(f"Result: {result}")

def test_basic_functionality():
    """Test basic functionality with a simple paragraph"""
    paragraph = "Machine learning improves medical diagnosis accuracy. Deep learning models can detect cancer with high precision."
    
    print("Testing basic functionality...")
    result = find_supporting_literature(paragraph)
    print_result("Basic Functionality", result)
    return result

def test_single_sentence():
    """Test with a single sentence"""
    paragraph = "Artificial intelligence reduces diagnostic errors by 50%."
    
    print("Testing single sentence...")
    result = find_supporting_literature(paragraph)
    print_result("Single Sentence", result)
    return result

def test_complex_paragraph():
    """Test with a more complex paragraph"""
    paragraph = """
    Neural networks have revolutionized image recognition in healthcare. 
    Convolutional neural networks achieve 95% accuracy in mammography screening.
    Radiologists using AI assistance show 23% improvement in diagnostic confidence.
    However, deep learning models require large datasets for optimal performance.
    """
    
    print("Testing complex paragraph...")
    result = find_supporting_literature(paragraph)
    print_result("Complex Paragraph", result)
    return result

def test_controversial_claim():
    """Test with controversial claims that might have limited supporting evidence"""
    paragraph = "AI will completely replace human doctors within 5 years. Traditional medical education will become obsolete."
    
    print("Testing controversial claims...")
    result = find_supporting_literature(paragraph)
    print_result("Controversial Claims", result)
    return result

def test_different_model():
    """Test with a different model"""
    paragraph = "Machine learning algorithms can predict patient outcomes with high accuracy."
    
    print("Testing with different model (gemini-1.5-pro)...")
    result = find_supporting_literature(paragraph, model="gemini-1.5-pro")
    print_result("Different Model", result)
    return result

def test_empty_paragraph():
    """Test with empty paragraph"""
    paragraph = ""
    
    print("Testing empty paragraph...")
    result = find_supporting_literature(paragraph)
    print_result("Empty Paragraph", result)
    return result

def main():
    """Run all tests"""
    # Check if API key is available
    if not os.getenv('GEMINI_API_KEY'):
        print("ERROR: GEMINI_API_KEY not found in environment variables.")
        print("Please set your API key in .env file or environment.")
        sys.exit(1)
    
    print("Starting find_supporting_literature tests...")
    print(f"API Key found: {'Yes' if os.getenv('GEMINI_API_KEY') else 'No'}")
    
    tests = [
        test_basic_functionality,
        test_single_sentence,
        test_complex_paragraph,
        test_controversial_claim,
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
    results_file = "test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Detailed results saved to: {results_file}")

if __name__ == "__main__":
    main()