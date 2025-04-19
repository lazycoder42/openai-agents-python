"""
Test script for scraping and LLM validation from a user-provided URL.

Usage:
  python test_scrape_validate.py <URL> [user_query]

- The script will scrape the URL (including ZIP support),
  and validate the data with the LLM using the user query (or a default prompt).
"""

from datavis_tools import DataVisTools
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_scrape_validate.py <URL> [user_query]")
        sys.exit(1)
    url = sys.argv[1]
    user_query = sys.argv[2] if len(sys.argv) > 2 else "GDP of UK in last 10 years"
    print(f"Scraping structured data from: {url}\n")
    ranked = [{
        'title': url,
        'link': url,
        'description': url
    }]
    scrape_result = DataVisTools.scrape_structured_data_from_sources(ranked, max_sources=1, table_hint=None)
    print(f"Status: {scrape_result['status']}")
    print(f"Method: {scrape_result['method']}")
    print(f"Source URL: {scrape_result['source_url']}")
    print(f"Message: {scrape_result['message']}")
    if scrape_result['data'] is not None:
        print("\nPreview of extracted data:")
        print(scrape_result['data'].head())
        print("\nValidating with LLM...")
        validation = DataVisTools.validate_scraped_data_with_llm(user_query, scrape_result['data'])
        print(f"\nLLM validation: {validation['llm_reason']}")
        print(f"Is valid: {validation['is_valid']}")
    else:
        print("No data extracted.") 