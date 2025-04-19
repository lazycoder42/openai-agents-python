import typing
import os
import requests
import json
from dotenv import load_dotenv
import nltk

class DataVisTools:
    """
    A collection of modular tools for a data visualization agent to search, extract, clean, validate, and format data for visualization.
    """

    @staticmethod
    def search_data_sources(query: str, provider: str = 'bing') -> typing.Dict[str, typing.Any]:
        """
        Search the internet for relevant data sources for the given query using Bing or Google Search via RapidAPI.
        Args:
            query (str): The user's data request in natural language.
            provider (str): 'bing' (default) or 'google'.
        Returns:
            Dict[str, Any]: Dictionary containing all search results and metadata, with keys:
                - 'items': List of search results with title, link, and description
                - 'related_keywords': List of related search terms
                - 'questions': List of related questions (if any)
                - 'query': The original query
                - 'metadata': Additional metadata about the request
        Raises:
            RuntimeError: If the .env file or required keys are missing.
        """
        # Load environment variables from .env file in the same directory as this script
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        if not os.path.exists(env_path):
            raise RuntimeError(f".env file not found at {env_path}. Please create it with RAPIDAPI_KEY, RAPIDAPI_BING_HOST, and RAPIDAPI_GOOGLE_HOST.")
        load_dotenv(env_path)
        rapidapi_key = os.getenv('RAPIDAPI_KEY')
        bing_host = os.getenv('RAPIDAPI_BING_HOST')
        google_host = os.getenv('RAPIDAPI_GOOGLE_HOST')
        if not rapidapi_key or not bing_host or not google_host:
            raise RuntimeError("RAPIDAPI_KEY, RAPIDAPI_BING_HOST, or RAPIDAPI_GOOGLE_HOST not set in .env file.")

        if provider == 'bing':
            url = f"https://{bing_host}/api/rapid/web_search"
            headers = {
                'x-rapidapi-key': rapidapi_key,
                'x-rapidapi-host': bing_host,
            }
            params = {
                'keyword': query,
                'page': 0,
                'size': 30,
            }
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            results = {
                'items': [],
                'related_keywords': [],
                'questions': [],
                'query': query,
                'metadata': {'request_info': data.get('request', {}), 'response_time': data.get('in_seconds', 0), 'provider': 'bing'}
            }
            if 'data' in data and 'items' in data['data']:
                results['items'] = data['data']['items']
            if 'data' in data and 'keyword_related' in data['data']:
                results['related_keywords'] = data['data']['keyword_related']
            if 'data' in data and 'question' in data['data']:
                results['questions'] = data['data']['question']
            return results
        elif provider == 'google':
            url = f"https://{google_host}/search"
            headers = {
                'x-rapidapi-key': rapidapi_key,
                'x-rapidapi-host': google_host,
            }
            params = {
                'q': query,
                'start': 1,
                'gl': 'US',
                'hl': 'en',
                'lr': 'lang_en',
            }
            # Print the actual request for debugging
            print(f"[DEBUG] Google API request: {url}")
            print(f"[DEBUG] Params: {params}")
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            raw = response.text
            print(f"[DEBUG] Raw Google API response: {raw[:1000]}")
            try:
                data = response.json()
            except Exception:
                # Try to load as a list
                try:
                    data = json.loads(raw)
                except Exception:
                    data = {'items': []}
            results = {
                'items': [],
                'related_keywords': [],
                'questions': [],
                'query': query,
                'metadata': {'provider': 'google'}
            }
            # Extract from 'organic_results' if present
            if isinstance(data, dict) and 'organic_results' in data:
                for item in data['organic_results']:
                    results['items'].append({
                        'title': item.get('title'),
                        'link': item.get('link'),
                        'description': item.get('snippet')
                    })
            # Fallback to previous logic
            elif isinstance(data, dict):
                if 'results' in data:
                    for item in data['results']:
                        results['items'].append({
                            'title': item.get('title'),
                            'link': item.get('link'),
                            'description': item.get('description')
                        })
                elif 'data' in data and 'results' in data['data']:
                    for item in data['data']['results']:
                        results['items'].append({
                            'title': item.get('title'),
                            'link': item.get('link'),
                            'description': item.get('description')
                        })
                elif isinstance(data.get('items'), list):
                    for item in data['items']:
                        results['items'].append(item)
                else:
                    # If dict but no 'results', treat as a single item
                    results['items'].append(data)
            elif isinstance(data, list):
                for item in data:
                    results['items'].append(item)
            return results
        else:
            raise ValueError("Provider must be either 'bing' or 'google'.")

    @staticmethod
    def classify_and_rank_sources(sources: typing.List[dict], query: str) -> typing.List[dict]:
        """
        Classify and rank sources by trust, relevance, data/format, and recency using a generic, context-aware scoring system.
        Uses NLTK for tokenization and stopword removal.
        
        Setup (run once):
            import nltk
            nltk.download('punkt')
            nltk.download('stopwords')
        
        Args:
            sources (List[dict]): List of search results (each with title, link, description, etc.)
            query (str): The original user query
        Returns:
            List[dict]: List of sources with added scores and source_type, sorted by composite_score (desc)
        """
        import re
        from urllib.parse import urlparse
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        import datetime

        # 1. Extract query terms
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(query)
        query_terms = [w.lower() for w in tokens if w.isalnum() and w.lower() not in stop_words]
        # print(f"[DEBUG] Query terms: {query_terms}")

        # 2. Scoring functions
        def domain_trust_score(url):
            try:
                domain = urlparse(url).netloc
            except Exception:
                return 30
            if domain.endswith('.gov') or domain.endswith('.edu') or domain.endswith('.org'):
                return 80
            if domain.endswith('.com') or domain.endswith('.net'):
                return 50
            if any(x in domain for x in ['data', 'stat', 'official', 'archive']):
                return 60
            return 30

        def relevance_score(texts):
            text = ' '.join([t for t in texts if t])
            text = text.lower()
            if not query_terms:
                return 0
            matches = sum(1 for term in query_terms if term in text)
            return int((matches / len(query_terms)) * 100)

        def data_signal_score(url, desc):
            signals = ["csv", "excel", "xls", "xlsx", "table", "dataset", "download"]
            score = 0
            for s in signals:
                if (url and s in url.lower()) or (desc and s in desc.lower()):
                    score += 20
            if url and (url.lower().endswith('.csv') or url.lower().endswith('.xls') or url.lower().endswith('.xlsx')):
                score += 30
            return min(score, 100)

        def recency_score(desc):
            # Look for years in the description (e.g., 2023, 2022)
            this_year = datetime.datetime.now().year
            years = re.findall(r'\b(20\d{2}|19\d{2})\b', desc or '')
            if years:
                most_recent = max(int(y) for y in years)
                if this_year - most_recent <= 2:
                    return 100
                elif this_year - most_recent <= 5:
                    return 70
                elif this_year - most_recent <= 10:
                    return 40
            return 0

        def source_type(url, desc):
            domain = urlparse(url).netloc if url else ''
            if domain.endswith('.gov') or 'official' in domain:
                return 'official'
            if 'data' in domain or 'dataset' in (desc or '').lower():
                return 'dataset'
            if 'news' in domain:
                return 'news'
            return 'other'

        # 3. Score and rank each source
        ranked = []
        for src in sources:
            url = src.get('link') or src.get('url')
            title = src.get('title', '')
            desc = src.get('description', '')
            trust = domain_trust_score(url)
            relevance = relevance_score([title, desc, url])
            data_signal = data_signal_score(url, desc)
            recency = recency_score(desc)
            composite = int(0.3 * trust + 0.4 * relevance + 0.2 * data_signal + 0.1 * recency)
            stype = source_type(url, desc)
            ranked.append({
                **src,
                'trust_score': trust,
                'relevance_score': relevance,
                'data_signal_score': data_signal,
                'recency_score': recency,
                'composite_score': composite,
                'source_type': stype
            })
        # Sort by composite_score descending
        ranked.sort(key=lambda x: x['composite_score'], reverse=True)
        return ranked

    @staticmethod
    def scrape_data_from_url(url: str) -> typing.Any:
        """
        Extract structured/tabular data from the given URL.
        Args:
            url (str): The URL to scrape data from.
        Returns:
            Any: Raw extracted data (tables, CSV, JSON, or plain text).
        """
        pass

    @staticmethod
    def clean_and_normalize_data(raw_data: typing.Any) -> typing.Any:
        """
        Iteratively clean, normalize, and structure the extracted data.
        Args:
            raw_data (Any): The raw data extracted from a source.
        Returns:
            Any: Cleaned and normalized data (e.g., pandas DataFrame, CSV, or JSON).
        """
        pass

    @staticmethod
    def validate_data(cleaned_data: typing.Any) -> dict:
        """
        Validate the cleaned data for quality, completeness, and consistency.
        Args:
            cleaned_data (Any): The cleaned data to validate.
        Returns:
            dict: Validation report and/or cleaned data.
        """
        pass

    @staticmethod
    def format_and_export_data(cleaned_data: typing.Any, format: str = 'csv') -> typing.Tuple[str, dict]:
        """
        Convert cleaned data to the desired output format (CSV or JSON) and attach attribution metadata.
        Args:
            cleaned_data (Any): The cleaned data to export.
            format (str): Output format, either 'csv' or 'json'.
        Returns:
            Tuple[str, dict]: Output data as a string and attribution info.
        """
        pass

    @staticmethod
    def generate_attribution(sources: typing.List[dict]) -> str:
        """
        Generate a summary of data sources for attribution and transparency.
        Args:
            sources (List[dict]): List of source metadata dicts.
        Returns:
            str: Attribution text or citation block.
        """
        pass

    @staticmethod
    def scrape_structured_data_from_sources(ranked_sources: list, max_sources: int = 5, table_hint: str = None) -> dict:
        """
        Try to extract structured data (CSV, Excel, JSON, HTML table, or ZIP containing these) from a list of ranked sources.
        First tries basic scraping (requests, pandas, BeautifulSoup), then advanced scraping with Playwright if needed.
        Now supports ZIP files containing CSV, Excel, or JSON.
        If no data is found directly, will also scan the HTML for <a href=...> links to downloadable files (one-level deep), including links with download-related keywords or query parameters.
        
        Playwright setup (run once):
            pip install playwright
            playwright install
        
        Args:
            ranked_sources (list): List of ranked source dicts (with at least 'link' or 'url')
            max_sources (int): How many sources to try (default: 5)
            table_hint (str): Optional keyword(s) to help select the right table
        Returns:
            dict: {
                'status': 'success' | 'no_data_found' | 'error',
                'data': DataFrame/CSV/JSON if found, else None,
                'source_url': URL where data was found (if any),
                'method': 'basic' | 'advanced',
                'message': str
            }
        """
        import os
        import requests
        import pandas as pd
        from bs4 import BeautifulSoup
        from urllib.parse import urlparse, urljoin
        import traceback
        import tempfile
        import zipfile
        import io
        from io import StringIO

        # Helper: Try to extract tables from HTML using pandas
        def extract_table_from_html(html, table_hint=None):
            try:
                tables = pd.read_html(StringIO(html))  # Fix warning by using StringIO
                if not tables:
                    return None
                if table_hint:
                    # Try to find a table with the hint in its columns
                    for t in tables:
                        if any(table_hint.lower() in str(col).lower() for col in t.columns):
                            return t
                # Otherwise, return the largest table
                return max(tables, key=lambda t: t.shape[0] * t.shape[1])
            except Exception:
                return None

        # Helper: Try to download and parse direct file links
        def try_direct_file(url):
            try:
                resp = None
                # For CSV, Excel, JSON, or ZIP download links
                is_csv = url.lower().endswith('.csv') or 'downloadformat=csv' in url.lower()
                is_excel = url.lower().endswith('.xls') or url.lower().endswith('.xlsx') or 'downloadformat=excel' in url.lower()
                is_json = url.lower().endswith('.json')
                is_zip = url.lower().endswith('.zip') or 'download' in url.lower()
                # Always use requests.get for download links
                if is_csv or is_excel or is_json or is_zip:
                    resp = requests.get(url, timeout=30)
                    print(f"[DEBUG] Downloading from {url} - status: {resp.status_code}")
                    print(f"[DEBUG] Response headers: {resp.headers}")
                    if resp.status_code != 200:
                        print(f"[DEBUG] Failed to download from {url}")
                        return None
                    content_type = resp.headers.get('Content-Type', '').lower()
                    content_disp = resp.headers.get('Content-Disposition', '').lower()
                    # If the response is a ZIP file (by content type or disposition), treat as ZIP
                    if (
                        'zip' in content_type or
                        content_disp.endswith('.zip"') or
                        (resp.content[:4] == b'PK\x03\x04')  # ZIP magic number
                    ):
                        print(f"[DEBUG] Treating response as ZIP file.")
                        import tempfile, zipfile, io
                        with tempfile.TemporaryDirectory() as tmpdir:
                            z = zipfile.ZipFile(io.BytesIO(resp.content))
                            z.extractall(tmpdir)
                            for name in z.namelist():
                                fpath = os.path.join(tmpdir, name)
                                if name.lower().endswith('.csv'):
                                    try:
                                        df = pd.read_csv(fpath)
                                        return df
                                    except Exception as e:
                                        print(f"[DEBUG] Failed to parse CSV in ZIP: {e}")
                                        continue
                                if name.lower().endswith('.json'):
                                    try:
                                        df = pd.read_json(fpath)
                                        return df
                                    except Exception as e:
                                        print(f"[DEBUG] Failed to parse JSON in ZIP: {e}")
                                        continue
                                if name.lower().endswith('.xls') or name.lower().endswith('.xlsx'):
                                    try:
                                        df = pd.read_excel(fpath)
                                        return df
                                    except Exception as e:
                                        print(f"[DEBUG] Failed to parse Excel in ZIP: {e}")
                                        continue
                        return None
                    # If CSV
                    if is_csv or 'csv' in content_type or url.lower().endswith('.csv'):
                        import io
                        try:
                            df = pd.read_csv(io.StringIO(resp.text))
                            return df
                        except Exception as e:
                            print(f"[DEBUG] Failed to parse CSV: {e}")
                            return None
                    # If Excel
                    if is_excel or 'excel' in content_type or url.lower().endswith('.xls') or url.lower().endswith('.xlsx'):
                        import io
                        try:
                            df = pd.read_excel(io.BytesIO(resp.content))
                            return df
                        except Exception as e:
                            print(f"[DEBUG] Failed to parse Excel: {e}")
                            return None
                    # If JSON
                    if is_json or 'json' in content_type or url.lower().endswith('.json'):
                        try:
                            df = pd.read_json(resp.content)
                            return df
                        except Exception as e:
                            print(f"[DEBUG] Failed to parse JSON: {e}")
                            return None
                # Fallback: try direct file by extension (legacy)
                if url.lower().endswith('.csv'):
                    try:
                        df = pd.read_csv(url)
                        return df
                    except Exception as e:
                        print(f"[DEBUG] Fallback failed to parse CSV: {e}")
                        return None
                if url.lower().endswith('.json'):
                    try:
                        df = pd.read_json(url)
                        return df
                    except Exception as e:
                        print(f"[DEBUG] Fallback failed to parse JSON: {e}")
                        return None
                if url.lower().endswith('.xls') or url.lower().endswith('.xlsx'):
                    try:
                        df = pd.read_excel(url)
                        return df
                    except Exception as e:
                        print(f"[DEBUG] Fallback failed to parse Excel: {e}")
                        return None
            except Exception as e:
                print(f"[DEBUG] Exception in try_direct_file for {url}: {e}")
                return None
            return None

        # Helper: Advanced scraping with Playwright
        async def playwright_scrape(url, table_hint=None):
            from playwright.async_api import async_playwright
            import asyncio
            try:
                async with async_playwright() as p:
                    browser = await p.chromium.launch(headless=True)
                    page = await browser.new_page()
                    await page.goto(url, timeout=30000)
                    await page.wait_for_load_state('networkidle')
                    html = await page.content()
                    await browser.close()
                    return extract_table_from_html(html, table_hint)
            except Exception:
                return None

        # Main round-robin loop
        for i, src in enumerate(ranked_sources[:max_sources]):
            url = src.get('link') or src.get('url')
            if not url:
                continue
            # 1. Basic scraping
            try:
                # Direct file or ZIP
                df = try_direct_file(url)
                if df is not None:
                    return {
                        'status': 'success',
                        'data': df,
                        'source_url': url,
                        'method': 'basic',
                        'message': f"Structured file found and parsed at {url}"
                    }
                # HTML table
                resp = requests.get(url, timeout=20)
                if resp.status_code == 200:
                    df = extract_table_from_html(resp.text, table_hint)
                    if df is not None:
                        return {
                            'status': 'success',
                            'data': df,
                            'source_url': url,
                            'method': 'basic',
                            'message': f"HTML table found and parsed at {url}"
                        }
                    # 1-level deep: look for downloadable links in HTML
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    links = [a.get('href') for a in soup.find_all('a', href=True)]
                    print(f"[DEBUG] All discovered <a href> links on {url}:")
                    for l in links:
                        print(f"  {l}")
                    # Filter for downloadable file types or download-related keywords
                    file_exts = ['.csv', '.xls', '.xlsx', '.json', '.zip']
                    file_keywords = ['downloadformat=csv', 'downloadformat=excel', 'download']
                    file_links = []
                    for l in links:
                        abs_url = urljoin(url, l)
                        if any(l.lower().endswith(ext) for ext in file_exts) or any(kw in l.lower() for kw in file_keywords):
                            file_links.append(abs_url)
                    print(f"[DEBUG] Filtered download/data links:")
                    for fl in file_links:
                        print(f"  {fl}")
                    for file_url in file_links:
                        df = try_direct_file(file_url)
                        if df is not None:
                            return {
                                'status': 'success',
                                'data': df,
                                'source_url': file_url,
                                'method': 'basic',
                                'message': f"Structured file found and parsed at {file_url} (discovered via HTML link on {url})"
                            }
            except Exception as e:
                # Log and continue to advanced scraping
                print(f"[Basic scraping failed for {url}]: {e}")
                # print(traceback.format_exc())

            # 2. Advanced scraping (Playwright)
            try:
                import asyncio
                df = asyncio.run(playwright_scrape(url, table_hint))
                if df is not None:
                    return {
                        'status': 'success',
                        'data': df,
                        'source_url': url,
                        'method': 'advanced',
                        'message': f"HTML table found with Playwright at {url}"
                    }
            except Exception as e:
                print(f"[Advanced scraping failed for {url}]: {e}")
                # print(traceback.format_exc())

        # If all sources fail
        return {
            'status': 'no_data_found',
            'data': None,
            'source_url': None,
            'method': None,
            'message': 'No structured data found in any of the top sources.'
        }

    @staticmethod
    def generate_query_variants(user_query: str, n: int = 7) -> list:
        """
        Use the OpenAI API (>=1.0.0) to generate a list of enhanced search queries for structured data discovery.
        Reads the OpenAI API key from the .env file as OPENAI_API_KEY.
        Args:
            user_query (str): The original user query.
            n (int): Number of variants to generate (default 7).
        Returns:
            list: List of enhanced query strings.
        Note:
            Requires openai Python package >=1.0.0
        """
        import os
        import openai
        from dotenv import load_dotenv

        # Load API key
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        load_dotenv(env_path)
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in .env file.")
        openai.api_key = api_key

        prompt = f"""
You are a data search assistant. Given a user's data request, generate a list of search queries that maximize the chance of finding downloadable or structured data (such as CSV, Excel, or tables) from authoritative sources. Use variations that include keywords like "filetype:csv", "download", "dataset", "table", and restrict to known data sites if appropriate.

Examples:
User query: "CO2 emissions by country"
Enhanced queries:
- CO2 emissions by country filetype:csv
- CO2 emissions by country dataset
- CO2 emissions by country download
- CO2 emissions by country table
- CO2 emissions by country site:data.worldbank.org

User query: "GDP of UK in last 10 years"
Enhanced queries:
- GDP of UK in last 10 years filetype:csv
- GDP of UK in last 10 years download
- GDP of UK in last 10 years dataset
- GDP of UK in last 10 years table
- GDP of UK in last 10 years site:ons.gov.uk
- GDP of UK in last 10 years site:worldbank.org

User query: "{user_query}"
Enhanced queries:
"""
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates search query variants for structured data discovery."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=256,
            temperature=0.2,
        )
        # Parse the response to extract the list of queries
        text = response.choices[0].message.content
        # Expecting a list of queries, one per line, possibly starting with '- '
        queries = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith('- '):
                queries.append(line[2:])
            elif line:
                queries.append(line)
        # Remove duplicates and limit to n
        queries = list(dict.fromkeys(queries))[:n]
        return queries

    @staticmethod
    def validate_scraped_data_with_llm(user_query: str, data_sample, model: str = "gpt-4o") -> dict:
        """
        Uses the OpenAI API to validate if the scraped data matches the user's intent.
        Args:
            user_query (str): The original user request.
            data_sample (pd.DataFrame): A pandas DataFrame (first 5 rows and all columns, or just column names if too large)
            model (str): OpenAI model to use (default: "gpt-4o")
        Returns:
            dict: {'is_valid': bool, 'llm_reason': str}
        """
        import os
        import openai
        from dotenv import load_dotenv
        import pandas as pd

        # Load API key
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        load_dotenv(env_path)
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in .env file.")
        openai.api_key = api_key

        # Prepare data sample for prompt
        if isinstance(data_sample, pd.DataFrame):
            columns = list(data_sample.columns)
            sample_rows = data_sample.head(5).to_csv(index=False)
        else:
            columns = []
            sample_rows = str(data_sample)

        prompt = f"""
User intent: "{user_query}"
Here are the columns and first 5 rows of the data I found:

Columns: {columns}
Sample:
{sample_rows}

Does this data match the user's request? Answer "yes" or "no" and explain briefly.
"""
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a data validation assistant. You help determine if a given data sample matches a user's request."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=128,
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip().lower()
        # Parse for yes/no
        is_valid = text.startswith('yes')
        return {'is_valid': is_valid, 'llm_reason': text}

    @staticmethod
    def mine_data_with_llm(user_query: str, df, model: str = "gpt-4o") -> dict:
        """
        Uses the OpenAI API to generate Python code for mining/filtering the relevant data from a DataFrame based on the user's intent.
        The LLM is given the user query, column names, and a sample of the data, and is asked to output code that creates a DataFrame named 'result'.
        Args:
            user_query (str): The original user request.
            df (pd.DataFrame): The loaded DataFrame to mine.
            model (str): OpenAI model to use (default: "gpt-4o")
        Returns:
            dict: {'success': bool, 'result': DataFrame or None, 'llm_code': str, 'llm_reason': str}
        """
        import os
        import openai
        from dotenv import load_dotenv
        import pandas as pd
        import traceback
        from io import StringIO

        # Load API key
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        load_dotenv(env_path)
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in .env file.")
        openai.api_key = api_key

        # Prepare sample for prompt
        columns = list(df.columns)
        sample_rows = df.head(20).to_csv(index=False)
        prompt = f"""
User intent: "{user_query}"
Here are the columns and first 20 rows of the data I found:

Columns: {columns}
Sample:
{sample_rows}

Write Python code using pandas to extract only the data relevant to the user's request. Assume the DataFrame is named df. The code should output the result as a new DataFrame named result. Only output the code, nothing else.
"""
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a data mining assistant. You help generate Python code to extract relevant data from a DataFrame."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.0,
        )
        code = response.choices[0].message.content.strip()
        print(f"[DEBUG] LLM-generated code:\n{code}")
        # Try to execute the code safely
        local_vars = {'df': df.copy(), 'pd': pd}
        try:
            exec(code, {}, local_vars)
            result = local_vars.get('result', None)
            if isinstance(result, pd.DataFrame):
                return {'success': True, 'result': result, 'llm_code': code, 'llm_reason': 'LLM code executed successfully.'}
            else:
                return {'success': False, 'result': None, 'llm_code': code, 'llm_reason': 'LLM code did not produce a DataFrame named result.'}
        except Exception as e:
            print(f"[DEBUG] Exception while executing LLM code: {e}\n{traceback.format_exc()}")
            return {'success': False, 'result': None, 'llm_code': code, 'llm_reason': f'Exception while executing LLM code: {e}'}

    @staticmethod
    def mine_csv_with_llm(user_query: str, csv_file: str, model: str = "gpt-4o", n_lines: int = 50, max_attempts: int = 3) -> dict:
        """
        Uses the OpenAI API to generate Python code for loading and mining/filtering relevant data from a CSV file with unknown metadata/header lines.
        The LLM is given the user query and the first n_lines of the CSV as plain text, and is asked to output code that loads and filters the data as a DataFrame named 'result'.
        The actual filename is provided and the code is saved to a .py file for testing.
        If the code fails, the error and code are sent back to the LLM for correction, up to max_attempts.
        Args:
            user_query (str): The original user request.
            csv_file (str): Path to the CSV file.
            model (str): OpenAI model to use (default: "gpt-4o")
            n_lines (int): Number of lines of the CSV to show the LLM (default: 50)
            max_attempts (int): Maximum number of correction attempts (default: 3)
        Returns:
            dict: {'success': bool, 'result': DataFrame or None, 'llm_code': str, 'llm_reason': str, 'code_file': str}
        """
        import os
        import openai
        from dotenv import load_dotenv
        import pandas as pd
        import traceback
        import time

        # Load API key
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        load_dotenv(env_path)
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in .env file.")
        openai.api_key = api_key

        # Read the static prompt template from markdown file
        prompt_path = os.path.join(os.path.dirname(__file__), 'mine_csv_prompt.md')
        with open(prompt_path, 'r', encoding='utf-8') as pf:
            static_prompt = pf.read()
        # Read first n_lines of the CSV as plain text
        with open(csv_file, 'r', encoding='utf-8') as f:
            sample_lines = ''.join([next(f) for _ in range(n_lines)])
        # Format the prompt
        prompt = static_prompt.format(
            user_query=user_query,
            n_lines=n_lines,
            sample_lines=sample_lines,
            csv_file=csv_file
        )
        code = None
        error_msg = None
        for attempt in range(1, max_attempts + 1):
            if attempt == 1:
                llm_prompt = prompt
            else:
                llm_prompt = (
                    f"The following code failed with this error:\n\n{code}\n\nError:\n{error_msg}\n\n"
                    f"{prompt}\nPlease fix the code so it works as intended."
                )
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a data mining assistant. You help generate Python code to load and extract relevant data from a CSV file."},
                    {"role": "user", "content": llm_prompt}
                ],
                max_tokens=512,
                temperature=0.0,
            )
            code = response.choices[0].message.content.strip()
            # Remove Markdown code block markers
            code = code.replace('```python', '').replace('```', '').strip()
            # Replace any instance of 'csv_file' with the actual filename
            code = code.replace("'csv_file'", f"'{csv_file}'").replace('"csv_file"', f'"{csv_file}"')
            # Save the code to a file
            ts = int(time.time())
            code_file = f"llm_code_{ts}_attempt{attempt}.py"
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            print(f"[DEBUG] LLM-generated code for CSV mining (attempt {attempt}) saved to: {code_file}\n{code}")
            # Try to execute the code safely
            local_vars = {'pd': pd}
            try:
                exec(code, {}, local_vars)
                result = local_vars.get('result', None)
                if isinstance(result, pd.DataFrame):
                    return {'success': True, 'result': result, 'llm_code': code, 'llm_reason': f'LLM code executed successfully on attempt {attempt}.', 'code_file': code_file}
                else:
                    error_msg = 'LLM code did not produce a DataFrame named result.'
                    print(f"[DEBUG] {error_msg}")
            except Exception as e:
                error_msg = f'Exception while executing LLM code: {e}\n{traceback.format_exc()}'
                print(f"[DEBUG] {error_msg}")
        return {'success': False, 'result': None, 'llm_code': code, 'llm_reason': f'Failed after {max_attempts} attempts. Last error: {error_msg}', 'code_file': code_file} 