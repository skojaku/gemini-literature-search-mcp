#!/usr/bin/env python3
"""
Test script to verify token limit handling
"""

import os
import sys
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

def test_long_input():
    """Test handling of very long input that would exceed token limits"""
    print("Testing token limit handling with very long input...")
    
    # Create a very long paragraph that would exceed token limits
    long_paragraph = """
    Machine learning has revolutionized medical diagnosis by providing unprecedented accuracy in pattern recognition and data analysis. The integration of artificial intelligence systems in healthcare has led to significant improvements in diagnostic speed and precision, particularly in medical imaging applications such as radiology, pathology, and dermatology. Deep learning models, specifically convolutional neural networks, have demonstrated remarkable capabilities in identifying subtle patterns in medical images that might be missed by human observers. These systems can process vast amounts of data in seconds, providing real-time analysis that supports clinical decision-making processes.
    
    The development of AI-powered diagnostic tools has been particularly transformative in cancer detection and staging. Studies have shown that machine learning algorithms can achieve diagnostic accuracy rates comparable to or exceeding those of experienced specialists in various medical fields. For instance, in dermatology, AI systems have been trained to identify melanoma and other skin cancers with accuracy rates exceeding 90%, while in radiology, deep learning models can detect lung nodules, breast cancer, and neurological abnormalities with high precision.
    
    However, the implementation of AI in medical diagnosis also presents significant challenges and limitations that must be carefully considered. Issues such as data bias, algorithm transparency, regulatory compliance, and integration with existing healthcare systems remain major obstacles to widespread adoption. The quality and diversity of training data significantly impact the performance and generalizability of AI models, and there are concerns about the potential for these systems to perpetuate or amplify existing healthcare disparities.
    
    Furthermore, the interpretability of AI decision-making processes remains a critical concern for healthcare providers and patients alike. Many advanced machine learning models operate as "black boxes," making it difficult for clinicians to understand how specific diagnoses or recommendations are generated. This lack of transparency can undermine trust and acceptance among healthcare professionals and may pose challenges for regulatory approval and clinical adoption.
    
    The economic implications of AI adoption in healthcare are also substantial and multifaceted. While AI systems have the potential to reduce costs through improved efficiency and earlier detection of diseases, the initial investment required for implementation, training, and maintenance can be significant. Healthcare organizations must carefully evaluate the cost-benefit ratio of AI implementation, considering factors such as return on investment, staff training requirements, and ongoing technical support needs.
    
    Patient privacy and data security represent additional critical considerations in the deployment of AI diagnostic systems. The use of large datasets containing sensitive medical information raises important questions about data protection, consent, and the potential for unauthorized access or misuse. Healthcare organizations must implement robust cybersecurity measures and comply with relevant regulations such as HIPAA to ensure patient data remains secure and confidential.
    
    The regulatory landscape for AI in healthcare continues to evolve as authorities work to establish appropriate frameworks for evaluating and approving AI-powered medical devices and diagnostic tools. The FDA and other regulatory bodies are developing new guidelines and approval processes specifically designed to address the unique characteristics and challenges of AI technologies in healthcare settings.
    
    Training and education of healthcare professionals represent another crucial aspect of successful AI implementation. Clinicians must develop new skills and competencies to effectively use AI tools and interpret their outputs in the context of patient care. This requires comprehensive training programs and ongoing professional development to ensure that healthcare providers can leverage AI capabilities while maintaining their clinical judgment and expertise.
    """ * 10  # Multiply to make it really long
    
    print(f"Input length: {len(long_paragraph)} characters")
    
    result = find_supporting_literature(long_paragraph)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print(f"Success: {result.get('success', False)}")
    
    if 'debug_info' in result:
        debug = result['debug_info']
        print(f"Input truncated: {debug.get('input_truncated', 'Unknown')}")
        print(f"Original length: {debug.get('original_input_length', 'Unknown')} characters")
        print(f"Processed length: {debug.get('processed_input_length', 'Unknown')} characters")
        print(f"Query length: {debug.get('query_length', 'Unknown')} characters")
    
    if result.get('success'):
        sentences = len(result.get('sentence_analysis', []))
        print(f"Sentences analyzed: {sentences}")
        
        total_papers = sum(len(s.get('supporting_literature', [])) for s in result.get('sentence_analysis', []))
        print(f"Total papers found: {total_papers}")
    else:
        print(f"Error: {result.get('error_details', 'Unknown error')}")
        print(f"Message: {result.get('message', 'No message')}")

def main():
    """Run token limit test"""
    # Check if API key is available
    if not os.getenv('GEMINI_API_KEY'):
        print("ERROR: GEMINI_API_KEY not found in environment variables.")
        print("Please set your API key in .env file or environment.")
        sys.exit(1)
    
    test_long_input()

if __name__ == "__main__":
    main()