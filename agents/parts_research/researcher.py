from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional
import logging
import random # For generating placeholder URLs
import asyncio
import json
import httpx # For calling Jina Reader API
import re
import urllib.parse # Import urlencode
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from bs4 import BeautifulSoup # Import BeautifulSoup

# Still need OpenAI client for the extraction step
from openai import AsyncOpenAI, APIError, Timeout
from config import get_settings
from models.parts_research import PartResearchResult, FoundPartOption # Assuming models exist

router = APIRouter()
settings = get_settings()
logger = logging.getLogger(__name__)

# OpenAI Client Initialization (ensure this is correctly placed, e.g., globally or passed appropriately)
try:
    async_openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    logger.info("AsyncOpenAI client initialized successfully.")
except Exception as e:
    async_openai_client = None
    logger.error(f"Failed to initialize AsyncOpenAI client: {e}", exc_info=True)

# --- Google Custom Search Client --- 
# MOVED CLASS DEFINITION EARLIER
class GoogleSearchClient:
    BASE_URL = "https://www.googleapis.com/customsearch/v1"

    def __init__(self, api_key: str, cse_id: str):
        if not api_key:
            raise ValueError("Google API key is required.")
        if not cse_id:
            raise ValueError("Google Custom Search Engine ID is required.")
        self.api_key = api_key
        self.cse_id = cse_id
        self.client = httpx.AsyncClient()
        logger.info("GoogleSearchClient initialized.")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(httpx.RequestError))
    async def search(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        params = {
            'key': self.api_key,
            'cx': self.cse_id,
            'q': query,
            'num': num_results
        }
        try:
            response = await self.client.get(self.BASE_URL, params=params)
            response.raise_for_status() # Raise HTTPStatusError for bad responses (4xx or 5xx)
            results = response.json()
            logger.debug(f"Google Search successful for query '{query}'. Items found: {len(results.get('items', []))}")
            return results
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.error(f"Google Search Quota Exceeded or Rate Limit Hit for query '{query}'. Status: {e.response.status_code}. Response: {e.response.text[:200]}")
            else:
                logger.error(f"Google Search HTTP Error for query '{query}': Status {e.response.status_code}. Response: {e.response.text[:200]}")
            # Reraise the specific error after logging, or return empty dict?
            # Returning empty allows workflow to potentially continue with other parts
            return {}
        except httpx.RequestError as e:
            logger.error(f"Google Search Request Error for query '{query}': {e}")
            # Reraise to allow tenacity to retry
            raise e
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response from Google Search for query '{query}': {e}")
            return {}
        except Exception as e:
            logger.exception(f"Unexpected error during Google Search for query '{query}': {e}")
            return {}

    async def close(self):
        await self.client.aclose()
        logger.info("GoogleSearchClient closed.")

# --- Constants --- 
JINA_READER_BASE_URL = "https://r.jina.ai/"
JINA_REQUEST_TIMEOUT = 45 # Timeout for Jina Reader call in seconds
LLM_EXTRACTION_TIMEOUT = 60 # Timeout for LLM extraction call
LLM_URL_SELECTION_TIMEOUT = 30.0
LLM_SELECTION_TIMEOUT = 30.0 # seconds for URL selection LLM call
MAX_URLS_TO_SELECT = 4 # Max URLs LLM should select

# Define standard browser-like headers
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0"
}
# Timeout for direct HTTP requests
HTTP_REQUEST_TIMEOUT = 30.0

# --- Helper Functions --- 

def construct_llm_selection_prompt(original_part: Dict[str, Any], search_results: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Constructs the prompt for the LLM to select the best URLs."""
    
    part_desc = original_part.get('description', 'N/A')
    part_num = original_part.get('part_number') or original_part.get('part_number', 'N/A') # Handle potential variations in key naming
    
    system_prompt = f"""
You are an AI assistant specialized in analyzing Google Search results for automotive parts.
Your goal is to select the most promising URLs that likely lead to a specific product page where the part can be purchased.
The target part is:
- Description: '{part_desc}'
- Part Number: '{part_num}'

Analyze the provided search results (title, link, snippet). Prioritize URLs from known auto parts retailers (e.g., RockAuto, FCP Euro, AutohausAZ, ECS Tuning, AutoZone, Advance Auto Parts, O'Reilly, NAPA, PartsGeek, CarParts.com) or official manufacturer/dealer sites. Avoid forums, informational pages (like Wikipedia), PDF links, news articles, or overly generic category pages unless they seem highly relevant.

From the list below, select up to {MAX_URLS_TO_SELECT} URLs that are most likely to be direct product pages for the target part. Return ONLY a valid JSON list of strings, where each string is a selected URL. Example: ["url1", "url2", "url3"]
"""
    
    results_text = "\n".join([
        f"- Title: {res['title']}\n  Link: {res['link']}\n  Snippet: {res.get('snippet', 'N/A')[:200]}..." 
        for res in search_results
    ])
    
    user_prompt = f"""
Based on the target part details provided in the system prompt, analyze these search results:

{results_text}

Return a JSON list containing up to {MAX_URLS_TO_SELECT} of the best URLs likely leading to a product page for the target part. Respond ONLY with the JSON list.
"""
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

async def call_llm_for_url_selection(original_part: Dict[str, Any], search_results: List[Dict[str, str]]) -> List[str]:
    """
    Calls an LLM to analyze search results and select the most promising URLs 
    that likely lead to product pages for the specified part.

    Args:
        original_part: Dictionary containing details of the part being searched (description, part_number, etc.).
        search_results: A list of dictionaries, each containing 'link', 'title', and 'snippet' from a web search.

    Returns:
        A list of selected URL strings. Returns an empty list if the LLM call fails or returns invalid data.
    """
    if not async_openai_client:
        logger.error("OpenAI client not available for URL selection.")
        return []
    if not search_results:
        logger.warning("No search results provided to LLM for URL selection.")
        return []

    part_desc = original_part.get('description', 'N/A')
    logger.info(f"Requesting LLM URL selection for part: '{part_desc}'")
    
    messages = construct_llm_selection_prompt(original_part, search_results)

    try:
        response = await async_openai_client.chat.completions.create(
            model=settings.OPENAI_MODEL, # Or a potentially faster/cheaper model if sufficient
            messages=messages,
            temperature=0.1, # Low temperature for deterministic selection
            response_format={"type": "json_object"}, # Expecting JSON output
            timeout=LLM_SELECTION_TIMEOUT
        )
        
        response_content = response.choices[0].message.content
        if not response_content:
            logger.warning(f"LLM returned empty response for URL selection for '{part_desc}'.")
            return []
            
        # The response *should* be a JSON object containing a list, like {"selected_urls": ["url1", ...]}
        # Or potentially just the list itself: ["url1", ...]
        # We need robust parsing
        selected_urls = []
        try:
            parsed_json = json.loads(response_content)
            if isinstance(parsed_json, list):
                selected_urls = [item for item in parsed_json if isinstance(item, str)]
            elif isinstance(parsed_json, dict):
                 # Look for a key that might contain the list (flexible)
                 for key, value in parsed_json.items():
                     if isinstance(value, list):
                         selected_urls = [item for item in value if isinstance(item, str)]
                         break # Take the first list found
                 if not selected_urls:
                     logger.warning(f"LLM JSON response for URL selection was a dict, but no list found. Content: {response_content[:500]}")
            else:
                logger.warning(f"LLM JSON response for URL selection was not a list or dict: {type(parsed_json)}. Content: {response_content[:500]}")

        except json.JSONDecodeError:
            logger.error(f"Failed to decode LLM JSON response for URL selection for '{part_desc}'. Content: {response_content[:500]}")
            # Fallback: try to extract URLs using regex if JSON fails (less reliable)
            potential_urls = re.findall(r'https?://[^\s"\'<>]+', response_content)
            if potential_urls:
                 logger.info(f"Falling back to regex extraction for URLs. Found: {len(potential_urls)}")
                 selected_urls = potential_urls[:MAX_URLS_TO_SELECT] # Limit results
            else:
                 selected_urls = []


        # Final validation and limit
        validated_urls = [url for url in selected_urls if url.startswith('http')]
        final_selection = validated_urls[:MAX_URLS_TO_SELECT]
        
        logger.info(f"LLM successfully selected {len(final_selection)} URLs for '{part_desc}': {final_selection}")
        return final_selection

    except Exception as e:
        logger.exception(f"Unexpected error during LLM URL selection for '{part_desc}': {e}")
        return []

async def get_target_urls_for_part(
    part_description: str,
    part_number: Optional[str],
    vehicle_year: Optional[int],
    vehicle_make: Optional[str],
    vehicle_model: Optional[str],
    google_search_client: GoogleSearchClient, 
    original_part_details: Dict[str, Any] 
) -> List[str]:
    """
    Uses Google Custom Search.
    1. Constructs targeted search queries for the top 5 priority sites.
    2. Constructs one general search query (no site restriction).
    3. Runs Google Custom Search for the targeted and general queries.
    4. Combines, deduplicates, and returns the top unique links found.
    """
    logger.info(f"Starting Google Search for: '{part_description}' (PN: {part_number})")
    
    # --- Construct Base Query --- 
    base_query_parts = []
    if vehicle_year and vehicle_make and vehicle_model:
         base_query_parts.append(f"{vehicle_year} {vehicle_make} {vehicle_model}")
    if part_description:
        base_query_parts.append(part_description)
    if part_number:
        base_query_parts.append(f'"{part_number}"') # Quote part number
        
    base_query = " ".join(base_query_parts)
    
    if not base_query:
        logger.warning("Cannot construct search query - missing description and part number.")
        return []
        
    # --- Construct Targeted Site Queries --- 
    target_sites = [ # Prioritize reliable sites first
        "fcpeuro.com",
        "rockauto.com",
        "autohausaz.com",
        "ecstuning.com",
        "partsgeek.com", 
    ]
    site_queries_to_run = [f"{base_query} site:{site}" for site in target_sites]
    
    # --- Define All Queries to Run --- 
    queries_to_run = site_queries_to_run + [base_query] # Add the general query
    logger.info(f"Running {len(queries_to_run)} Google searches (up to 5 site-specific + 1 general):")
    # Log first few queries for brevity if list is long
    for i, q in enumerate(queries_to_run):
        logger.info(f"  Query {i+1}: {q}")
    
    # --- Run Google Searches Concurrently (or sequentially if preferred) ---
    all_links_found = []
    search_tasks = []
    for query in queries_to_run:
        # Get slightly fewer results per query now that we have more queries
        search_tasks.append(google_search_client.search(query, num_results=2)) 

    try:
        search_results_list = await asyncio.gather(*search_tasks)
    except Exception as e:
        logger.error(f"Error gathering Google search results: {e}", exc_info=True)
        search_results_list = [] # Proceed with empty results if gather fails

    for i, search_results in enumerate(search_results_list):
        query_used = queries_to_run[i]
        if search_results and search_results.get("items"):
            logger.info(f"Got {len(search_results['items'])} results for query: '{query_used}'")
            for item in search_results["items"]:
                link = item.get("link")
                if link:
                    all_links_found.append(link)
        else:
             logger.warning(f"No items found in Google Search result for query: '{query_used}'")

    # --- Process aggregated links --- 
    # Deduplicate while preserving order (important for prioritizing site results)
    unique_links = list(dict.fromkeys(all_links_found))
    final_links = unique_links[:MAX_URLS_TO_SELECT] # Limit to max needed
    
    logger.info(f"Returning {len(final_links)} potential URLs from {len(queries_to_run)} searches: {final_links}")
    return final_links

def create_failure_stub(url: str, reason: str, vendor_hint: Optional[str] = None) -> Dict[str, Any]:
    """Creates a standardized dictionary for failed research attempts."""
    vendor = vendor_hint or "Unknown"
    if vendor == "Unknown": # Try to infer if hint not provided
        if "rockauto.com" in url: vendor = "RockAuto"
        elif "autozone.com" in url: vendor = "AutoZone"
        elif "amazon.com" in url: vendor = "Amazon"
        # Add others...
        
    return {
        "product_name": f"Failed: {reason}",
        "price": None,
        "currency": None,
        "availability": "Unknown (Failed)",
        "part_number": None,
        "vendor": vendor,
        "source_url": url,
        "status": "failed",
        "error_reason": reason
    }

async def scrape_with_httpx(url: str, client: httpx.AsyncClient) -> Dict[str, Any]:
    """Fetches raw HTML content directly using httpx."""
    logger.info(f"Requesting direct HTML scrape for URL: {url}")
    try:
        response = await client.get(
            url,
            headers=REQUEST_HEADERS, 
            timeout=HTTP_REQUEST_TIMEOUT, 
            follow_redirects=True
        )
        response.raise_for_status() # Raise exception for 4xx/5xx errors
        
        html_content = response.text
        final_url = str(response.url) # Get the URL after any redirects
        logger.info(f"Direct HTML scrape successful for {url} (final URL: {final_url}). Content length: {len(html_content)}")
        return {
            "status": "success", 
            "content": html_content, 
            "original_url": final_url, 
            "title": None # We don't easily get title from raw HTML without parsing
        }
            
    except httpx.HTTPStatusError as e:
        error_message = f"Direct HTTP Error {e.response.status_code}"
        logger.error(f"{error_message} for URL {url}: {e.response.text[:200]}")
        return create_failure_stub(url, error_message)
    except httpx.RequestError as e:
        # Includes timeouts, connection errors, etc.
        error_message = f"Direct Request Error: {type(e).__name__}"
        logger.error(f"{error_message} for URL {url}: {e}")
        return create_failure_stub(url, error_message)
    except Exception as e:
        error_message = f"Direct Scrape Unknown Error: {e}"
        logger.exception(f"Unexpected error during direct scrape for {url}: {e}")
        return create_failure_stub(url, error_message)

def normalize_part_number(pn: Optional[str]) -> Optional[str]:
    """Removes common separators and converts to uppercase for comparison."""
    if not pn: return None
    return re.sub(r'[-\s.]', '', str(pn)).upper()

def part_numbers_match(original_pn: Optional[str], extracted_pn: Optional[str]) -> bool:
    """Compares normalized part numbers. Handles None cases."""
    if not original_pn or not extracted_pn:
        return False # Cannot validate if either is missing
    
    norm_orig = normalize_part_number(original_pn)
    norm_extr = normalize_part_number(extracted_pn)
    
    return norm_orig == norm_extr

def construct_llm_extraction_prompt(
    page_content: str, 
    url: str, 
    original_description: str, 
    original_part_number: Optional[str]
) -> List[Dict[str, str]]:
    """Constructs the prompt for extracting part details, providing original part context."""
    system_prompt = f"""
You are an AI assistant analyzing raw HTML source code scraped from an automotive parts website.
Your task is to extract specific details for the product described on the page that BEST matches the following original part details:
- Description: '{original_description}'
- Part Number: '{original_part_number or "Not Provided"}'

Focus on finding the product name, price, currency, availability, vendor, and the specific part number listed on the page within the HTML content.
Respond ONLY with a valid JSON object containing these fields:
`product_name` (string | null), `price` (float | null), `currency` (string | null),
`availability` (string | null), `part_number` (string | null), `vendor` (string | null).

**IMPORTANT EXTRACTION GUIDELINES:**
*   **Price:** Search diligently for the price. Look for numbers near currency symbols (like '$', 'USD', 'CAD') or within common price tags/attributes (e.g., `class="price"`, `itemprop="price"`). Price might be in text nodes, meta tags (`og:price:amount`, `product:price:amount`), or JSON-LD script tags. Extract only the numerical value. Infer currency ('USD', 'CAD', etc.) if possible, otherwise use null.
*   **Part Number:** Look for the part number near labels like "Part #", "SKU", "Manufacturer Part Number", "Item #", "MPN". Also check attributes like `data-part-number`. Match against the original part number if provided.
*   **Availability:** Look for terms like "In Stock", "Out of Stock", "Backordered", "Available", "Ships in...", "Add to Cart".
*   **Vendor:** Infer the vendor from the URL (e.g., 'RockAuto', 'Pelican Parts', 'eBay') or look for it explicitly mentioned on the page.

If a value isn't found after careful searching, use null. Do not include introductory text or explanations.
Prioritize finding details for the specific part number if provided and found.
"""
    user_prompt = f"""
Analyze the following HTML source code scraped from {url} and extract the product details as JSON, ensuring they correspond to the original part details and guidelines provided in the system prompt:

```html
{page_content[:15000]} # Increased limit slightly more for complex HTML
```

Respond ONLY with the JSON object.
"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

async def call_llm_for_extraction_from_content(
    page_content: str, 
    url: str, 
    original_description: str, 
    original_part_number: Optional[str]
) -> Dict[str, Any]:
    """Calls OpenAI LLM, includes original part context. Returns dict with status."""
    if not async_openai_client:
        logger.error("OpenAI client not available for extraction.")
        return create_failure_stub(url, "OpenAI Client Not Configured")
    if not page_content:
        logger.warning(f"No page content provided for LLM extraction from {url}")
        return create_failure_stub(url, "Missing Page Content for LLM")
        
    logger.info(f"Requesting LLM extraction for content from: {url} (Target: '{original_description}' PN: {original_part_number})")
    messages = construct_llm_extraction_prompt(page_content, url, original_description, original_part_number)
    
    try:
        response = await async_openai_client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
            timeout=LLM_EXTRACTION_TIMEOUT
        )
        response_content = response.choices[0].message.content
        # --- ADDED DEBUG LOG --- 
        logger.debug(f"LLM RAW Extraction Response for {url}: {response_content[:500]}...")
        # --- END DEBUG LOG ---
        if not response_content:
            logger.warning(f"LLM returned empty response for extraction from {url}")
            return create_failure_stub(url, "LLM Returned Empty Response")
            
        extracted_data = json.loads(response_content)
        extracted_data['status'] = 'success'
        extracted_data['source_url'] = url 
        logger.info(f"LLM extraction successful for {url}. Extracted PN: {extracted_data.get('part_number')} Price: {extracted_data.get('price')}")
        return extracted_data
        
    except json.JSONDecodeError as e:
        error_message = f"LLM JSON Decode Error: {e}"
        logger.error(f"Failed to decode JSON from LLM extraction response for {url}: {response_content[:200]}")
        return create_failure_stub(url, error_message)
    except Exception as e:
        error_message = f"LLM API Exception: {type(e).__name__} - {e}"
        logger.exception(f"LLM extraction call failed for {url}: {e}")
        return create_failure_stub(url, error_message)

# --- Vendor Configuration ---
# Define vendor-specific search and parsing logic

def parse_pelican_parts_product_page(html_content: str, url: str) -> Optional[Dict[str, Any]]:
    """Specific parsing logic for Pelican Parts PRODUCT pages."""
    # This is the renamed 'extract_with_static_parser' function
    # (We'll move the actual implementation here in a moment)
    logger.debug(f"Attempting Pelican Parts product page parse for {url}")
    # ... (Implementation of the refined Pelican parser goes here) ...
    # Temporarily returning None until we move the code
    soup = BeautifulSoup(html_content, 'lxml') 
    # ... (Keep the full refined implementation) ...
    # --- Pelican Parts Specific Logic --- 
    if "pelicanparts.com" in url: # Keep this check just in case
        # ... (rest of the Pelican parsing logic from the previous version) ...
        pass # Placeholder for brevity
    return None # Return None if parsing fails or not applicable


# Rename function and change logic to extract product URLs
def extract_rockauto_product_links(html_content: str, url: str) -> List[str]:
    """Parses RockAuto search results page to extract links to individual product pages ('More Info' links)."""
    logger.debug(f"Attempting to extract RockAuto product links from {url}")
    if not html_content:
        logger.warning(f"No HTML content provided for RockAuto link extraction from {url}")
        return []
        
    soup = BeautifulSoup(html_content, 'lxml')
    product_links = []
    
    # Find the main container for listings based on the provided HTML structure
    parts_container = soup.find('div', class_='listing-container-border')
    
    if not parts_container:
        logger.warning(f"Could not find main parts container ('div.listing-container-border') on RockAuto page: {url}")
        # Fallback to previous logic just in case the structure varies?
        # listing_table = soup.find('table', class_='listing')
        # if listing_table:
        #     parts_container = listing_table.find('tbody')
        # if not parts_container:
        #      parts_container = soup.find('tbody', id=lambda x: x and x.startswith('listing_'))
        # if not parts_container:
        #      logger.error(f"Could not find parts container using ANY known method on RockAuto page: {url}")
        return [] # Cannot find links if container not found
        
    # Find all individual part containers (tbody elements with IDs starting with listingcontainer[)
    part_blocks = parts_container.find_all('tbody', id=lambda x: x and x.startswith('listingcontainer['))
    logger.debug(f"Found {len(part_blocks)} part blocks (tbody[id^=listingcontainer]) in container for {url}")

    if not part_blocks:
         logger.warning(f"Found the main container, but no individual part blocks (tbody[id^=listingcontainer]) within it for URL: {url}")
         # Maybe try the old row logic as a fallback inside the container?
         # rows = parts_container.find_all('tr') ... etc
         return []

    for part_block in part_blocks:
        # Find the 'More Info' link tag within this part's block
        link_tag = part_block.find('a', class_='ra-btn-moreinfo', href=lambda href: href and 'moreinfo.php' in href)
        if link_tag:
            href = link_tag.get('href')
            if href:
                 # Make sure the URL is absolute
                 absolute_url = urllib.parse.urljoin(url, href) 
                 product_links.append(absolute_url)
            else:
                 logger.warning(f"Found 'more info' link tag but it has no href in block {part_block.get('id', 'N/A')} on {url}")
        # else: # Log if a part block doesn't have an info link? Maybe too verbose.
             # logger.debug(f"No 'more info' link found in part block {part_block.get('id', 'N/A')} on {url}")

    # Deduplicate links
    unique_links = list(dict.fromkeys(product_links))
    if unique_links:
        logger.info(f"Extracted {len(unique_links)} unique product page ('More Info') links from RockAuto results page: {url}")
        # logger.debug(f"Links extracted: {unique_links}") # Log links if needed for debug
    else:
        logger.warning(f"Extraction complete, but no 'More Info' links were found on RockAuto results page: {url}")
        
    return unique_links

# Placeholder for the actual RockAuto product page parser
def parse_rockauto_product_page(html_content: str, url: str) -> Optional[Dict[str, Any]]:
    logger.debug(f"Attempting to parse RockAuto PRODUCT page ({url}) with static parser.")
    if not html_content:
        logger.warning(f"No HTML content provided for RockAuto product page parse: {url}")
        return create_failure_stub(url, "No HTML Content", "RockAuto")
        
    soup = BeautifulSoup(html_content, 'lxml')
    
    # Initialize data structure
    data = {
        'status': 'error',
        'url': url,
        'vendor': 'RockAuto',
        'part_number': None,
        'manufacturer': None,
        'price': None,
        'currency': None,
        'description': None,
        'image_url': None,
        'availability': 'Unknown',
        'specifications': None,
        'oem_numbers': None,
        'error_reason': None
    }

    try:
        # --- Extract from Buybox (Top Right) ---
        buybox = soup.find('div', class_='buybox')
        buybox_tbody = soup.find('tbody', id='mibuybox') # More specific target within buybox
        
        if buybox_tbody:
            # Manufacturer
            mfr_span = buybox_tbody.find('span', class_='listing-final-manufacturer')
            if mfr_span:
                data['manufacturer'] = mfr_span.get_text(strip=True)
                
            # Part Number
            pn_span = buybox_tbody.find('span', class_='listing-final-partnumber')
            if pn_span:
                data['part_number'] = pn_span.get_text(strip=True)

            # Price (Default visible price)
            price_span = buybox_tbody.find('span', id=lambda x: x and x.startswith('dprice[') and x.endswith('][v]'))
            if price_span:
                price_text = price_span.get_text(strip=True)
                # Clean price (remove $, commas, convert to float)
                price_match = re.search(r'[\$€£]?([\d,]+\.?\d*)', price_text) # Handle common currency symbols and commas
                if price_match:
                    try:
                        data['price'] = float(price_match.group(1).replace(',', ''))
                        # Try to infer currency
                        if '$' in price_text or 'USD' in price_text: data['currency'] = 'USD'
                        elif '€' in price_text or 'EUR' in price_text: data['currency'] = 'EUR'
                        elif '£' in price_text or 'GBP' in price_text: data['currency'] = 'GBP'
                        elif 'CAD' in price_text: data['currency'] = 'CAD' # Common for RockAuto Canada?
                        else: data['currency'] = 'USD' # Default assumption
                    except ValueError:
                        logger.warning(f"Could not convert extracted price '{price_match.group(1)}' to float on {url}")
                else:
                    logger.warning(f"Could not extract numerical price from text '{price_text}' on {url}")
            
            # Availability (Check for Add to Cart button)
            # Find either the input button (no JS) or the link wrapper (JS enabled)
            add_cart_btn = buybox_tbody.find('input', id=lambda x: x and x.startswith('addpart['))
            add_cart_link = buybox_tbody.find('div', id=lambda x: x and x.startswith('vew_btnaddtocart['))
            if add_cart_btn or (add_cart_link and 'ra-hide' not in add_cart_link.get('class', [])):
                 data['availability'] = 'Available' # Or maybe 'In Stock'?
            else:
                 # Check for notify button? For now, keep 'Unknown'
                 notify_btn = buybox_tbody.find('div', id=lambda x: x and x.startswith('vew_btnnotifyoos['))
                 if notify_btn and 'ra-hide' not in notify_btn.get('class', []):
                     data['availability'] = 'Out of Stock' # More specific if notify button is visible
        
        else:
            logger.warning(f"Could not find buybox tbody#mibuybox on {url}")
            data['error_reason'] = "Could not find buybox content"
            # Don't return yet, try to get other info

        # --- Extract Image ---
        img_tag = soup.find('img', class_='listing-inline-image-moreinfo')
        if img_tag and img_tag.get('src'):
            img_src = img_tag.get('src')
            if img_src:
                data['image_url'] = urllib.parse.urljoin(url, img_src)

        # --- Extract Description ---
        desc_section = soup.find('section', attrs={'aria-label': 'Part Description'})
        if desc_section:
            data['description'] = desc_section.get_text(strip=True)
        
        # --- Extract Specifications ---
        specs_table = soup.find('table', class_='moreinfotable')
        if specs_table and specs_table.find('th', string=lambda t: 'Specifications' in t):
            specs = {}
            rows = specs_table.find_all('tr')
            for row in rows[1:]: # Skip header row
                cols = row.find_all('td')
                if len(cols) == 2:
                    key = cols[0].get_text(strip=True)
                    value = cols[1].get_text(strip=True)
                    if key:
                        specs[key] = value
            if specs:
                data['specifications'] = specs

        # --- Extract OEM/Interchange Numbers ---
        oem_section = soup.find('section', attrs={'aria-label': 'Alternate/OEM Part Number(s)'})
        if oem_section:
             oem_text = oem_section.get_text(strip=True)
             # Extract numbers after the label, handle potential separators (comma, space)
             oem_string_only = oem_text.replace('OEM / Interchange Numbers:', '').strip()
             # Split by common delimiters (comma, space) and normalize each
             potential_oems = re.split(r'[,\s]+', oem_string_only)
             normalized_oems = [norm_pn for pn in potential_oems if (norm_pn := normalize_part_number(pn))] # Use assignment expression
             if normalized_oems:
                 data['oem_numbers'] = normalized_oems # Store as list of normalized strings
             else:
                  logger.debug(f"Found OEM section text but failed to extract/normalize numbers from '{oem_string_only}' on {url}")


        # --- Determine Success ---
        # Consider it a success if we got at least manufacturer, part number, and price
        if data['manufacturer'] and data['part_number'] and data['price'] is not None:
             data['status'] = 'success'
             logger.info(f"Successfully parsed RockAuto product page: {url}. Found: PN={data['part_number']}, Price={data['price']}")
             # Clear error reason if successful
             data.pop('error_reason', None) 
        else:
             # If we failed to get essential info, log it and keep status as error
             if not data['error_reason']: # Set a reason if none exists yet
                  missing = [k for k, v in data.items() if k in ['manufacturer', 'part_number', 'price'] and v is None]
                  data['error_reason'] = f"Missing essential fields: {', '.join(missing)}"
             logger.warning(f"Failed to parse essential info from RockAuto product page {url}. Reason: {data['error_reason']}")
             
        return data

    except Exception as e:
        logger.exception(f"Unexpected error parsing RockAuto product page {url}: {e}")
        data['status'] = 'error'
        data['error_reason'] = f"Parsing Exception: {type(e).__name__} - {e}"
        return data

VENDOR_CONFIG = {
    "pelicanparts.com": {
        "search_url_template": "https://www.pelicanparts.com/catalog/SuperCat/{part_number}_catalog.htm", # Might need adjustment
        "search_method": "GET",
        "product_page_parser": parse_pelican_parts_product_page, 
        "search_results_parser": None # TBD if needed
    },
    "rockauto.com": {
        "search_url_template": "https://www.rockauto.com/en/partsearch/?partnum={part_number}",
        "search_method": "GET",
        "product_page_parser": parse_rockauto_product_page, # Link placeholder
        "search_results_parser": extract_rockauto_product_links # Extracts links
    },
    # Add other vendors: fcpeuro.com, autohausaz.com, ecstuning.com, partsgeek.com
    "fcpeuro.com": {
        "search_url_template": "https://www.fcpeuro.com/products?keywords={part_number}", # Example
        "search_method": "GET",
        "product_page_parser": None, 
        "search_results_parser": None
    },
     "autohausaz.com": {
        "search_url_template": "https://www.autohausaz.com/catalog?q={part_number}", # Example
        "search_method": "GET",
        "product_page_parser": None, 
        "search_results_parser": None
    },
     "ecstuning.com": {
        "search_url_template": "https://www.ecstuning.com/Search/SiteSearch/{part_number}/", # Example
        "search_method": "GET",
        "product_page_parser": None, 
        "search_results_parser": None
    },
     "partsgeek.com": {
        "search_url_template": "https://www.partsgeek.com/catalog/{year}/{make}/{model}/{part_description}.html?find={part_number}", # Complex example, needs vehicle context
        "search_method": "GET",
        "product_page_parser": None, 
        "search_results_parser": None
    },
    
}

# --- NEW: Vendor Processing Logic --- (Modified)

async def search_and_extract_vendor_data(
    vendor: str, 
    config: Dict[str, Any], 
    part: Dict[str, Any], 
    vehicle_info: Dict[str, Any], 
    http_client: httpx.AsyncClient
) -> List[Dict[str, Any]]:
    """
    Handles searching ONE vendor, scraping, and extracting data.
    Prioritizes extracting product links from search results (if parser exists),
    then scrapes and parses the first product page.
    Falls back to LLM on search results page if other methods fail.
    """
    logger.info(f"--- Processing Vendor: {vendor} for Part: {part.get('description')} (PN: {part.get('part_number')}) ---")
    found_options = []
    part_number = part.get('part_number')
    part_description = part.get('description')
    original_search_url = "N/A" # Keep track of the initial URL

    if not part_number:
        logger.warning(f"Skipping vendor {vendor} - Part number missing for {part.get('description')}")
        return []
        
    search_url_template = config.get("search_url_template")
    if not search_url_template:
        logger.warning(f"Skipping vendor {vendor} - No search URL template defined.")
        return []

    # --- Construct and Scrape Initial Search URL ---
    try:
        encoded_part_number = urllib.parse.quote_plus(part_number)
        format_args = {"part_number": encoded_part_number}
        if '{year}' in search_url_template: format_args['year'] = vehicle_info.get('year', 'any')
        if '{make}' in search_url_template: format_args['make'] = vehicle_info.get('make', 'any')
        if '{model}' in search_url_template: format_args['model'] = vehicle_info.get('model', 'any')
        if '{part_description}' in search_url_template: format_args['part_description'] = part_description or 'part'
        placeholders = re.findall(r'{([^}]+)}', search_url_template)
        final_format_args = {k: format_args[k] for k in placeholders if k in format_args}
        original_search_url = search_url_template.format(**final_format_args)

        logger.info(f"Attempting search on {vendor} using URL: {original_search_url}")
        scrape_result = await scrape_with_httpx(original_search_url, http_client)

    except KeyError as e:
         logger.error(f"Missing key '{e}' needed for vendor {vendor} URL template: {search_url_template}")
         return [create_failure_stub(original_search_url, f"URL Formatting Error: Missing key '{e}'", vendor)]
    except Exception as e:
         logger.exception(f"Error formatting/scraping initial URL for vendor {vendor}: {e}")
         return [create_failure_stub(original_search_url, f"Initial Scrape/URL Error: {e}", vendor)]

    if scrape_result.get("status") != "success" or not scrape_result.get("content"):
        logger.warning(f"Failed to scrape initial search URL for {vendor}: {original_search_url}")
        return [create_failure_stub(original_search_url, f"Failed to scrape vendor search URL ({scrape_result.get('error_reason', 'Unknown')})", vendor)]
        
    initial_html_content = scrape_result["content"]
    actual_initial_url = scrape_result["original_url"] # URL after redirects

    # --- Strategy 1: Extract Product Links and Process First One --- 
    product_links = []
    link_extractor = config.get("search_results_parser") # e.g., extract_rockauto_product_links
    
    if link_extractor:
        logger.debug(f"Attempting to extract product links using {link_extractor.__name__} for {vendor}")
        try:
            product_links = link_extractor(initial_html_content, actual_initial_url)
        except Exception as e:
            logger.error(f"Error executing link extractor for {vendor}: {e}", exc_info=True)
            product_links = []
            
    if product_links:
        logger.info(f"Found {len(product_links)} product links. Processing the first one: {product_links[0]}")
        first_product_url = product_links[0]
        
        # Scrape the actual product page
        product_scrape_result = await scrape_with_httpx(first_product_url, http_client)
        
        if product_scrape_result.get("status") == "success" and product_scrape_result.get("content"):
            product_html_content = product_scrape_result["content"]
            actual_product_url = product_scrape_result["original_url"]
            
            # Try vendor-specific product page parser
            product_page_parser = config.get("product_page_parser") # e.g., parse_rockauto_product_page (placeholder)
            product_data = None
            if product_page_parser:
                logger.debug(f"Attempting static product page parse using {product_page_parser.__name__} for {vendor}.")
                try:
                    product_data = product_page_parser(product_html_content, actual_product_url)
                except Exception as e:
                    logger.error(f"Error executing product page parser for {vendor} on {actual_product_url}: {e}", exc_info=True)
                    product_data = None
            
            # If product parser failed or doesn't exist, use LLM on product page
            if not product_data:
                 logger.info(f"Static product page parse failed or N/A for {vendor}. Using LLM on product page: {actual_product_url}")
                 product_data = await call_llm_for_extraction_from_content(
                     page_content=product_html_content, 
                     url=actual_product_url,
                     original_description=part_description,
                     original_part_number=part_number
                 )
            
            # Add the result (success or failure stub from parser/LLM)
            if product_data:
                 if not product_data.get("vendor"): product_data["vendor"] = vendor.split('.')[0].capitalize()
                 product_data["source_url"] = actual_product_url # Use the product page URL
                 found_options.append(product_data)
                 # Successfully processed the first product link, we can stop here for this vendor
                 logger.info(f"Finished processing vendor {vendor} after parsing product page.")
                 return found_options
        else:
            # Failed to scrape the product page itself
             logger.warning(f"Failed to scrape product page link {first_product_url} for {vendor}. Reason: {product_scrape_result.get('error_reason', 'Unknown')}")
             # Add a failure stub for this attempt
             found_options.append(create_failure_stub(first_product_url, f"Failed to scrape product page ({product_scrape_result.get('error_reason', 'Unknown')})", vendor))
             # Continue to LLM fallback on search results page below

    # --- Strategy 2: Fallback to LLM on original search results page --- 
    # This runs if: 
    #   - No link extractor exists for the vendor OR
    #   - Link extractor failed to find links OR
    #   - Scraping the first product link failed
    if not found_options: # Only run if we haven't already added an option
        logger.info(f"Product link processing failed or N/A for {vendor}. Falling back to LLM extraction on initial search page: {actual_initial_url}")
        llm_fallback_data = await call_llm_for_extraction_from_content(
            page_content=initial_html_content, 
            url=actual_initial_url,
            original_description=part_description,
            original_part_number=part_number
        )
        
        if llm_fallback_data: # Could be success or failure stub
             if not llm_fallback_data.get("vendor"): llm_fallback_data["vendor"] = vendor.split('.')[0].capitalize()
             llm_fallback_data["source_url"] = actual_initial_url # URL of the search page
             found_options.append(llm_fallback_data)
        else: 
             # If even LLM fallback returns nothing (shouldn't happen with stubs)
             logger.error(f"LLM fallback on search page returned None for {vendor}. Creating generic failure.")
             found_options.append(create_failure_stub(actual_initial_url, "All parsing/extraction failed", vendor))

    logger.info(f"Finished processing vendor {vendor}. Found {len(found_options)} option(s).")
    return found_options


# --- Main Workflow Function (Refactored) --- 

async def research_parts_workflow(parts_to_research: List[Dict[str, Any]], http_client: httpx.AsyncClient, vehicle_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Orchestrates the parts research process by directly searching known vendors.
    1. Iterates through parts needing research.
    2. For each part, iterates through configured vendors.
    3. Calls vendor-specific search and extraction logic.
    4. Aggregates results, validates part numbers, compares prices.
    """
    results = []
    
    # No longer need Google Search Client here
    # google_search_client = None 
    # ... (google client init removed) ...

    for part in parts_to_research:
        part_description = part.get('description')
        part_number = part.get('part_number') # Original part number
        original_price = part.get('unit_price') 
        
        if not part_description:
            logger.warning(f"Skipping part due to missing description: {part}")
            results.append({"original_part": part, "potential_urls_attempted": [], "found_options": []}) # Keep structure similar
            continue
            
        # Part number is crucial for direct vendor search
        if not part_number:
            logger.warning(f"Skipping part '{part_description}' - Missing Part Number required for direct vendor search.")
            results.append({
                "original_part": part, 
                "potential_urls_attempted": [], 
                "found_options": [create_failure_stub("N/A", "Missing Part Number for Vendor Search", "System")]
            })
            continue

        logger.info(f"--- Starting research for: '{part_description}' (PN: {part_number}) ---")
        
        all_vendor_options = []
        potential_urls_list = [] # Keep track of URLs we actually attempted to scrape/process

        # Iterate through configured vendors
        vendor_tasks = []
        for vendor, config in VENDOR_CONFIG.items():
             # Create concurrent tasks for each vendor search/extraction
             vendor_tasks.append(
                 search_and_extract_vendor_data(
                     vendor=vendor, 
                     config=config, 
                     part=part, 
                     vehicle_info=vehicle_info, 
                     http_client=http_client
                 )
             )
             
        # Run vendor processing concurrently
        try:
            list_of_vendor_results = await asyncio.gather(*vendor_tasks)
            # Flatten the list of lists
            for vendor_result_list in list_of_vendor_results:
                all_vendor_options.extend(vendor_result_list)
                # Add the source URLs from successful results to our attempted list
                for option in vendor_result_list:
                    if option.get('source_url') and option['source_url'] != "N/A":
                        potential_urls_list.append(option['source_url'])
                        
        except Exception as e:
            logger.error(f"Error gathering vendor search results for part '{part_description}': {e}", exc_info=True)
            # Continue processing with potentially partial results if some vendors failed

        # Deduplicate potential URLs list
        potential_urls_list = list(dict.fromkeys(potential_urls_list))

        # Post-process results: Validate Part Number and Calculate Price Difference
        final_options = []
        for option in all_vendor_options:
            if option.get('status') == 'success':
                extracted_pn = option.get('part_number')
                extracted_oem_list = option.get('oem_numbers') # This is now a list of normalized strings
                
                # --- Part Number Validation (Modified) --- 
                validated = True # Assume valid initially
                if part_number and extracted_pn:
                     # Primary Check: Direct Match
                     if not part_numbers_match(part_number, extracted_pn):
                         # Secondary Check: Original PN in Extracted OEM List?
                         normalized_original_pn = normalize_part_number(part_number)
                         if isinstance(extracted_oem_list, list) and normalized_original_pn in extracted_oem_list:
                              logger.info(f"Primary PN mismatch for '{part_description}' from {option.get('vendor', 'Unknown')}, but original PN '{part_number}' found in extracted OEM list ({extracted_oem_list}). Accepting option.")
                              # validated remains True
                         else:
                              logger.warning(f"Part number mismatch for '{part_description}' from {option.get('vendor', 'Unknown')}. Original: '{part_number}', Extracted: '{extracted_pn}'. Extracted OEMs: {extracted_oem_list}. Discarding option.")
                              validated = False # Discard
                elif part_number and not extracted_pn:
                     logger.debug(f"Could not extract primary part number from {option.get('vendor', 'Unknown')} ({option.get('source_url')}) for validation against original PN '{part_number}'. Keeping option for now, but validation is incomplete.")
                     # validated remains True, but maybe flag as needs review?
                # No need for 'elif not part_number:' case as we check for part_number earlier

                # Discard if validation failed
                if not validated:
                     continue 

                # --- Price Comparison --- 
                found_price = option.get('price')
                option['price_comparison_status'] = "unknown"
                option['price_difference'] = None
                if isinstance(found_price, (int, float)) and isinstance(original_price, (int, float)):
                    difference = round(found_price - original_price, 2)
                    option['price_difference'] = difference
                    if difference < 0:
                        option['price_comparison_status'] = "cheaper"
                    elif difference > 0:
                        option['price_comparison_status'] = "more_expensive"
                    else:
                        option['price_comparison_status'] = "same_price"
            # Keep failure stubs as well
            final_options.append(option)

        # Sort successful options by price (cheapest first), keep failures at the end
        final_options.sort(key=lambda x: (x.get('status') != 'success', x.get('price') if isinstance(x.get('price'), (int, float)) else float('inf'))) 

        results.append({
            "original_part": part,
            "potential_urls_attempted": potential_urls_list, # URLs actually processed
            "found_options": final_options
        })
        logger.info(f"--- Finished research for: '{part_description}'. Found {len([opt for opt in final_options if opt.get('status') == 'success'])} valid options across {len(VENDOR_CONFIG)} vendors. ---")
        
        # Keep delay between PARTS, not vendors
        await asyncio.sleep(2) 

    # No Google client to close anymore
    # finally:
    #     if google_search_client:
    #         await google_search_client.close()

    return results

# --- API Endpoint --- 

@router.post("/research")
async def research_parts_endpoint(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    """
    Receives parts list AND vehicle info, simulates search, uses Jina & LLM.
    TEMPORARILY LIMITS to first 5 parts.
    """
    # Extract parts and vehicle info from payload
    if "line_items" not in payload:
        raise HTTPException(status_code=400, detail="Missing 'line_items' field in request body")
    # Expecting vehicle info potentially at the top level or within quote_analysis
    # Let's assume it might be passed directly in the payload for now
    vehicle_info = {
        "year": payload.get('year'),
        "make": payload.get('make'),
        "model": payload.get('model')
    }
    logger.debug(f"Received line items for research: {payload['line_items']}")
    logger.debug(f"Received vehicle info for research: {vehicle_info}")
    
    all_parts = [item for item in payload["line_items"] if item.get("item_type") == "part"]
    logger.info(f"Identified {len(all_parts)} total parts.")
    parts_to_research = all_parts # Use all parts
    
    if not parts_to_research:
        return JSONResponse(status_code=200, content={"message": "No parts required research.", "results": []})
        
    if not async_openai_client:
         detail = "Parts research feature is not fully configured (Missing: OpenAI API Key)"
         logger.error(detail)
         raise HTTPException(status_code=501, detail=detail)
         
    try:
        async with httpx.AsyncClient() as client:
            # Pass vehicle_info to the workflow
            results = await research_parts_workflow(parts_to_research, client, vehicle_info)
            
        logger.info(f"Successfully completed parts research using Google/Jina/LLM for {len(parts_to_research)} parts.")
        response_content = {"results": results}
        return JSONResponse(status_code=200, content=response_content)

    except Exception as e:
        logger.exception(f"An unexpected error occurred during Google/Jina/LLM parts research endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during parts research.")