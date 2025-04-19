"""
Test script for LLM-powered data mining from a CSV file with unknown metadata/header lines.

Usage:
  python test_mine_llm.py <csv_file> [user_query]

- Loads the CSV file using LLM-generated code, sends the user query and a sample to the LLM,
  and runs the generated code to mine the relevant data.
- Then validates the mined data with the LLM to check if it aligns with the user query.
"""

from datavis_tools import DataVisTools
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_mine_llm.py <csv_file> [user_query]")
        sys.exit(1)
    csv_file = sys.argv[1]
    user_query = sys.argv[2] if len(sys.argv) > 2 else "GDP of UK in last 10 years"
    print(f"Mining data from: {csv_file}")
    print(f"\nMining data with LLM for query: '{user_query}'\n")
    result = DataVisTools.mine_csv_with_llm(user_query, csv_file)
    print("\n=== LLM-GENERATED CODE ===")
    print(result['llm_code'])
    print("\n=== LLM EXECUTION RESULT ===")
    print(f"Success: {result['success']}")
    print(f"Reason: {result['llm_reason']}")
    if result['result'] is not None:
        print("\nPreview of mined data:")
        print(result['result'].head())
        print("\nValidating mined data with LLM...")
        validation = DataVisTools.validate_scraped_data_with_llm(user_query, result['result'])
        print("\n=== LLM VALIDATION RESULT ===")
        print(f"Is valid: {validation['is_valid']}")
        print(f"LLM Reason: {validation['llm_reason']}")
        if validation['is_valid']:
            print("\nPreview of validated data:")
            print(result['result'].head())
    else:
        print("No mined data returned.") 