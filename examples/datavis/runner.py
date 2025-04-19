import pandas as pd
from datavis_tools import DataVisTools


def run_data_agent(
    user_prompt: str,
    max_outer_iterations: int = 5,
    max_sources_per_query: int = 5,
    table_hint: str = None,
    verbose: bool = True
) -> dict:
    """
    Orchestrates the agentic workflow: refine query (LLM) -> search -> rank -> scrape -> validate (LLM).
    Returns the first valid result or a summary if none found.
    """
    if verbose:
        print(f"[Agent] User prompt: {user_prompt}")
        print("[Agent] Step 1: Generating enhanced queries via LLM...")
    query_variants = DataVisTools.generate_query_variants(user_prompt, n=max_outer_iterations)
    if verbose:
        print(f"[Agent] Query variants: {query_variants}")
    for idx, variant in enumerate(query_variants):
        if verbose:
            print(f"\n[Agent] Step 2: Searching for variant {idx+1}/{len(query_variants)}: '{variant}'")
        results = DataVisTools.search_data_sources(variant, provider='google')
        ranked = DataVisTools.classify_and_rank_sources(results['items'], variant)
        if verbose:
            print(f"[Agent] Top {min(3, len(ranked))} ranked sources:")
            for i, item in enumerate(ranked[:3], 1):
                print(f"  {i}. {item.get('title', 'N/A')} ({item.get('link', 'N/A')}) [Score: {item.get('composite_score', 'N/A')}]" )
        if verbose:
            print(f"[Agent] Step 3: Scraping structured data from top {max_sources_per_query} sources...")
        scrape_result = DataVisTools.scrape_structured_data_from_sources(ranked, max_sources=max_sources_per_query, table_hint=table_hint)
        if scrape_result['status'] == 'success':
            if verbose:
                print(f"[Agent] Step 4: Validating scraped data with LLM...")
            validation = DataVisTools.validate_scraped_data_with_llm(user_prompt, scrape_result['data'])
            if verbose:
                print(f"[Agent] LLM validation: {validation['llm_reason']}")
            if validation['is_valid']:
                if verbose:
                    print(f"[Agent] SUCCESS: Valid data found!")
                return {
                    'status': 'success',
                    'data': scrape_result['data'],
                    'source_url': scrape_result['source_url'],
                    'query_used': variant,
                    'llm_reason': validation['llm_reason'],
                    'message': f"Valid data found for query: '{variant}' from {scrape_result['source_url']}"
                }
            else:
                if verbose:
                    print(f"[Agent] Data did not pass LLM validation. Trying next variant...")
        else:
            if verbose:
                print(f"[Agent] No structured data found for this query variant. Trying next...")
    if verbose:
        print(f"[Agent] No valid data found after trying all query variants.")
    return {
        'status': 'no_valid_data_found',
        'data': None,
        'source_url': None,
        'query_used': None,
        'llm_reason': None,
        'message': 'No valid structured data found for any query variant.'
    }


if __name__ == "__main__":
    import sys
    user_prompt = "GDP of UK in last 10 years"
    if len(sys.argv) > 1:
        user_prompt = ' '.join(sys.argv[1:])
    result = run_data_agent(user_prompt, verbose=True)
    print("\n=== FINAL RESULT ===")
    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")
    if result['data'] is not None:
        print("\nPreview of extracted data:")
        print(result['data'].head())
    if result['llm_reason']:
        print(f"\nLLM validation reason: {result['llm_reason']}")
    if result['source_url']:
        print(f"\nSource URL: {result['source_url']}")
    if result['query_used']:
        print(f"\nQuery used: {result['query_used']}") 