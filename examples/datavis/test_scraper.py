"""
Test script for DataVisTools.scrape_structured_data_from_sources

Instructions:
1. Ensure you have run:
   pip install requests python-dotenv nltk pandas beautifulsoup4 playwright
   playwright install
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
2. Run this script:
   python test_scraper.py
"""

from datavis_tools import DataVisTools
import pandas as pd
import openai

if __name__ == "__main__":
    query = "GDP of UK in last 10 years"
    print(f"Searching and scraping for: {query}\n")
    # Step 1: Search
    results = DataVisTools.search_data_sources(query, provider='google')
    # Step 2: Rank
    ranked = DataVisTools.classify_and_rank_sources(results['items'], query)
    print(f"Top 5 ranked sources:")
    for i, item in enumerate(ranked[:5], 1):
        print(f"  {i}. {item.get('title', 'N/A')} ({item.get('link', 'N/A')}) [Score: {item.get('composite_score', 'N/A')}]")
    print("\nAttempting to scrape structured data from top sources...\n")
    # Step 3: Scrape
    scrape_result = DataVisTools.scrape_structured_data_from_sources(ranked, max_sources=5, table_hint="GDP")
    print(f"Status: {scrape_result['status']}")
    print(f"Method: {scrape_result['method']}")
    print(f"Source URL: {scrape_result['source_url']}")
    print(f"Message: {scrape_result['message']}")
    if scrape_result['data'] is not None:
        print("\nPreview of extracted data:")
        print(scrape_result['data'].head())
    else:
        print("No data extracted.")

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[...],
        max_tokens=256,
        temperature=0.2,
    )

    text = response.choices[0].message.content 