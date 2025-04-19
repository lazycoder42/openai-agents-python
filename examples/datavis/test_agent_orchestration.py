"""
Test script for full agentic workflow: query refinement, search, ranking, and scraping.

Instructions:
1. Ensure you have run:
   pip install requests python-dotenv nltk pandas beautifulsoup4 playwright openai
   playwright install
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
2. Add your OpenAI API key to .env as OPENAI_API_KEY
3. Run this script:
   python test_agent_orchestration.py
"""

from datavis_tools import DataVisTools
import pandas as pd

if __name__ == "__main__":
    user_query = "GDP of UK in last 10 years"
    print(f"User query: {user_query}\nGenerating enhanced search queries...\n")
    query_variants = DataVisTools.generate_query_variants(user_query)
    print("Query variants:")
    for i, q in enumerate(query_variants, 1):
        print(f"  {i}. {q}")
    print("\n---\n")
    found = False
    for variant in query_variants:
        print(f"Searching for: {variant}")
        results = DataVisTools.search_data_sources(variant, provider='google')
        ranked = DataVisTools.classify_and_rank_sources(results['items'], variant)
        print(f"  Top 3 ranked sources:")
        for i, item in enumerate(ranked[:3], 1):
            print(f"    {i}. {item.get('title', 'N/A')} ({item.get('link', 'N/A')}) [Score: {item.get('composite_score', 'N/A')}]")
        print("  Attempting to scrape structured data from top sources...")
        scrape_result = DataVisTools.scrape_structured_data_from_sources(ranked, max_sources=5, table_hint="GDP")
        if scrape_result['status'] == 'success':
            print(f"\nSUCCESS! Data found for query: '{variant}'")
            print(f"Source: {scrape_result['source_url']}")
            print(f"Method: {scrape_result['method']}")
            print(f"Message: {scrape_result['message']}")
            print("\nPreview of extracted data:")
            print(scrape_result['data'].head())
            found = True
            break
        else:
            print(f"  No data found for this query. Trying next variant...\n---\n")
    if not found:
        print("No structured data could be found for any query variant.") 