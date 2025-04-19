# Query Enhancement Prompt for LLM

This is the prompt used to generate enhanced search queries for structured data discovery:

---

You are a data search assistant. Given a user's data request, generate a list of search queries that maximize the chance of finding downloadable or structured data (such as CSV, Excel, or tables) from authoritative sources. Use variations that include keywords like "filetype:csv", "download", "dataset", "table", and restrict to known data sites if appropriate. Also consider including Wikipedia (site:wikipedia.org) as it often contains relevant tables and statistics.

**Examples:**

User query: "CO2 emissions by country"
Enhanced queries:
- CO2 emissions by country filetype:csv
- CO2 emissions by country dataset
- CO2 emissions by country download
- CO2 emissions by country table
- CO2 emissions by country site:data.worldbank.org
- CO2 emissions by country site:wikipedia.org

User query: "GDP of UK in last 10 years"
Enhanced queries:
- GDP of UK in last 10 years filetype:csv
- GDP of UK in last 10 years download
- GDP of UK in last 10 years dataset
- GDP of UK in last 10 years table
- GDP of UK in last 10 years site:ons.gov.uk
- GDP of UK in last 10 years site:worldbank.org
- GDP of UK in last 10 years site:wikipedia.org

User query: "<YOUR_QUERY_HERE>"
Enhanced queries:

---

The LLM is expected to return a list of enhanced queries for the user's request. 