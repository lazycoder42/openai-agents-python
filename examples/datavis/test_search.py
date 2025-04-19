"""
Test script for DataVisTools.search_data_sources and classify_and_rank_sources

Instructions:
1. Create a .env file in the same directory with the following content:
   RAPIDAPI_KEY=your_rapidapi_key_here
   RAPIDAPI_BING_HOST=bing-search-apis.p.rapidapi.com
   RAPIDAPI_GOOGLE_HOST=googlesearch-api.p.rapidapi.com
2. Install dependencies:
   pip install requests python-dotenv nltk
3. Run this script:
   python test_search.py [provider]
      where provider is 'bing' (default) or 'google'
4. The first time, run:
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
"""

from datavis_tools import DataVisTools
import json
import sys

if __name__ == "__main__":
    query = "GDP of UK in last 10 years"
    provider = 'google'
    if len(sys.argv) > 1:
        provider = sys.argv[1].lower()
    print(f"Searching for: {query}\nUsing provider: {provider}\n")
    try:
        results = DataVisTools.search_data_sources(query, provider=provider)
        print(f"Found {len(results['items'])} search results, {len(results['related_keywords'])} related keywords, and {len(results['questions'])} questions.\n")
        print("=== TOP 5 RANKED RESULTS ===")
        ranked = DataVisTools.classify_and_rank_sources(results['items'], query)
        for i, item in enumerate(ranked[:5], 1):
            print(f"Rank {i}:")
            print(f"  Title: {item.get('title', 'N/A')}")
            print(f"  Link: {item.get('link', 'N/A')}")
            print(f"  Composite Score: {item.get('composite_score', 'N/A')}")
            print(f"    Trust: {item.get('trust_score', 'N/A')}, Relevance: {item.get('relevance_score', 'N/A')}, Data: {item.get('data_signal_score', 'N/A')}, Recency: {item.get('recency_score', 'N/A')}")
            print(f"  Source Type: {item.get('source_type', 'N/A')}")
            print(f"  Description: {item.get('description', 'N/A')[:100]}..." if len(item.get('description', '')) > 100 else item.get('description', 'N/A'))
            print("-" * 80)
        print("\n=== METADATA ===")
        print(f"Query: {results['query']}")
        print(f"Provider: {results['metadata'].get('provider', 'N/A')}")
        if 'response_time' in results['metadata']:
            print(f"Response time: {results['metadata']['response_time']} seconds")
    except Exception as e:
        print(f"Error: {e}") 