 # Data Search Agent Strategy (Full Pipeline Overview)
**(LLM-powered steps are marked with 🧠)**

Our data search agent is an automated system that takes a user’s natural language query and delivers relevant, validated data from the web or APIs. The pipeline is orchestrated as a loop, ensuring robust, adaptive, and intelligent data retrieval. The main steps are:

## 1. User Query Enhancement 🧠
- The agent receives a user’s natural language query (e.g., “GDP of UK in the last 10 years”).
- The query is enhanced using prompt engineering and, if needed, LLM assistance to clarify intent, expand keywords, and generate effective search terms.

## 2. Web/API Search (Discovery)
- The enhanced query is used to search for relevant data sources using web search APIs (e.g., Bing via RapidAPI).
- The agent retrieves a list of candidate links, which may include direct data files, web pages, or API endpoints.

## 3. Ranking & Filtering
- The candidate links are ranked based on relevance to the user query, file type, recency, and other heuristics.
- The agent filters out low-quality or irrelevant links, prioritizing those most likely to contain structured, downloadable data.

## 4. Scraping & Downloading
- For each high-ranking link, the agent attempts to download the data file (CSV, Excel, JSON, ZIP, etc.).
- If the link is a web page, the agent scrapes it for embedded data links or tables.
- The agent handles authentication, redirects, and various file formats as needed.

## 5. Data Mining with LLM 🧠
- The agent extracts a sample of the downloaded data and combines it with the user query to create a context-rich prompt.
- This prompt is sent to a large language model (LLM), which generates Python code to load, filter, and extract the data relevant to the query.
- The generated code is executed in a controlled environment, and the resulting DataFrame is previewed.

## 6. LLM Validation 🧠
- The mined data and the original user query are sent back to the LLM for validation.
- The LLM checks if the extracted data aligns with the user’s intent and provides reasoning for its decision.

## 7. Orchestrator Loop
- The orchestrator manages the entire process, looping through candidate links and iterating as needed:
    - If a link fails (e.g., download error, irrelevant data, failed validation), the agent moves to the next candidate.
    - The loop continues until valid, relevant data is found or all options are exhausted.
- The orchestrator logs each step, handles errors gracefully, and ensures the user receives the best possible result.

## 8. User Feedback & Transparency
- The agent presents the validated data, the LLM’s reasoning, and a preview to the user.
- The process is transparent, with logs and explanations provided for each step.

---

**Summary:**  
This pipeline leverages LLMs for query enhancement, code generation, and validation, combines web/API search with intelligent ranking and scraping, and is orchestrated in a robust loop to maximize the chances of finding and delivering high-quality, relevant data in response to any user query.
