#!/usr/bin/env python3
"""
Test script to verify incomplete JSON handling
"""

import json
import re

def test_incomplete_json_handling():
    """Test the incomplete JSON handling logic"""
    
    # Simulate the incomplete JSON from the error
    incomplete_json = '''{
    "sentence_analysis": [
        {
            "sentence_number": 1,
            "sentence_text": "When biologist Jennifer Doudna explored CRISPR applications, her breakthrough required traversing from RNA structural biology to genome editing—a trajectory unlikely to emerge from similarity‑based recommendation systems, which would have directed her toward RNA structure literature rather than therapeutic applications.",
            "supporting_literature": [
                {
                    "title": "Jennifer Doudna - Innovative Genomics Institute (IGI)",
                    "authors": [],
                    "year": 2020,
                    "abstract":"'''
    
    print("Original incomplete JSON:")
    print(f"Length: {len(incomplete_json)}")
    print(f"Open braces: {incomplete_json.count('{')}")
    print(f"Close braces: {incomplete_json.count('}')}")
    print(f"Open brackets: {incomplete_json.count('[')}")  
    print(f"Close brackets: {incomplete_json.count(']')}")
    
    # Apply the same logic as in the function
    json_text = incomplete_json
    
    # Handle incomplete JSON (common with cut-off responses)
    if json_text.count('{') > json_text.count('}'):
        print("\nDetected incomplete JSON, attempting to complete...")
        
        # Try to complete the JSON structure
        missing_braces = json_text.count('{') - json_text.count('}')
        print(f"Missing closing braces: {missing_braces}")
        
        # Find the last complete sentence_analysis entry
        if '"sentence_analysis":' in json_text:
            # Truncate at the last complete literature entry or sentence
            last_complete_pos = max(
                json_text.rfind('}]'),  # End of literature array
                json_text.rfind('}}'),  # End of sentence object
            )
            print(f"Last complete position: {last_complete_pos}")
            
            if last_complete_pos > 0:
                # Find the appropriate closing position
                truncated = json_text[:last_complete_pos + 2]
                print(f"Truncated at position {last_complete_pos + 2}")
                
                # Complete the structure
                if truncated.count('[') > truncated.count(']'):
                    missing_brackets = truncated.count('[') - truncated.count(']')
                    truncated += ']' * missing_brackets
                    print(f"Added {missing_brackets} closing brackets")
                    
                if truncated.count('{') > truncated.count('}'):
                    missing_braces = truncated.count('{') - truncated.count('}')
                    truncated += '}' * missing_braces
                    print(f"Added {missing_braces} closing braces")
                
                json_text = truncated
    
    print("\nCompleted JSON:")
    print(f"Length: {len(json_text)}")
    print(f"Open braces: {json_text.count('{')}")
    print(f"Close braces: {json_text.count('}')}")
    print(f"Open brackets: {json_text.count('[')}")  
    print(f"Close brackets: {json_text.count(']')}")
    
    # Try to parse
    try:
        result = json.loads(json_text)
        print("\n✓ Successfully parsed JSON!")
        print(f"Sentences found: {len(result.get('sentence_analysis', []))}")
        if result.get('sentence_analysis'):
            sentence = result['sentence_analysis'][0]
            papers = len(sentence.get('supporting_literature', []))
            print(f"Papers in first sentence: {papers}")
        return True
    except json.JSONDecodeError as e:
        print(f"\n✗ JSON parsing failed: {e}")
        print("Completed JSON preview:")
        print(json_text[:500] + "..." if len(json_text) > 500 else json_text)
        return False

if __name__ == "__main__":
    test_incomplete_json_handling()