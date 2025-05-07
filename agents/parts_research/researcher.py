from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional
import logging
import asyncio
import json
import httpx
import re
import urllib.parse
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from bs4 import BeautifulSoup

from .rockauto_parser import extract_rockauto_product_links, parse_rockauto_product_page
from .utils import normalize_part_number, create_failure_stub
from .fcpeuro_parser import extract_fcpeuro_product_links, parse_fcpeuro_product_page

from openai import AsyncOpenAI, APIError, Timeout
from config import get_settings
from models.parts_research import PartResearchResult, FoundPartOption

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

# --- Constants --- 
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

def part_numbers_match(original_pn: Optional[str], extracted_pn: Optional[str]) -> bool:
    """Compares normalized part numbers. Handles None cases."""
    if not original_pn or not extracted_pn:
        return False # Cannot validate if either is missing
    
    norm_orig = normalize_part_number(original_pn)
    norm_extr = normalize_part_number(extracted_pn)
    
    logger.debug(f"Part number comparison - Original: '{original_pn}' -> '{norm_orig}', Extracted: '{extracted_pn}' -> '{norm_extr}'")
    
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
            timeout=60
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

# --- FCP Euro Placeholder Parsers ---
def extract_fcpeuro_product_links(html_content: str, url: str) -> List[str]:
    """Parses FCP Euro search results to find product links."""
    logger.info(f"[FCP Euro] Attempting to extract product links from {url}")
    if not html_content:
        logger.warning(f"[FCP Euro] No HTML content provided for link extraction from {url}")
        return []

    soup = BeautifulSoup(html_content, 'lxml')
    product_links = []
    base_url = "https://www.fcpeuro.com"

    # Products are in <div class="grid-x hit" data-href="...">
    product_divs = soup.find_all('div', class_='hit')

    if not product_divs:
        logger.warning(f"[FCP Euro] No 'div.hit' elements found on {url}. Link extraction might fail or page structure may have changed.")
        return []

    for div in product_divs:
        data_href = div.get('data-href')
        if data_href:
            # Ensure the URL is absolute
            absolute_url = urllib.parse.urljoin(base_url, data_href)
            product_links.append(absolute_url)
        else:
            logger.debug(f"[FCP Euro] Found a 'div.hit' without a 'data-href' attribute on {url}. Skipping.")

    unique_links = list(dict.fromkeys(product_links)) # Deduplicate while preserving order
    if unique_links:
        logger.info(f"[FCP Euro] Extracted {len(unique_links)} unique product links from {url}")
    else:
        logger.warning(f"[FCP Euro] Link extraction complete, but no product links were found on {url}")
        
    return unique_links

def parse_fcpeuro_product_page(html_content: str, url: str) -> Optional[Dict[str, Any]]:
    """Parses an FCP Euro product page to extract product details."""
    logger.info(f"[FCP Euro] Attempting to parse product page: {url}")
    if not html_content:
        logger.warning(f"[FCP Euro] No HTML content for product page parse: {url}")
        return create_failure_stub(url, "No HTML Content", "FCP Euro")

    soup = BeautifulSoup(html_content, 'lxml')
    data = {
        'status': 'error',
        'source_url': url,
        'vendor': 'FCP Euro',
        'product_name': None,
        'part_number': None, # This will be the FCP Euro SKU
        'manufacturer': None,
        'price': None,
        'currency': None,
        'availability': 'Unknown',
        'image_url': None,
        'oem_numbers': [],
        'specifications': {},
        'error_reason': None
    }

    try:
        # Product Name
        name_tag = soup.find('h1', class_='listing__name')
        if name_tag: data['product_name'] = name_tag.get_text(strip=True)

        # Price and Currency (from listing__amount)
        price_container = soup.find('div', class_='listing__amount')
        if price_container:
            price_span = price_container.find('span')
            if price_span:
                price_text = price_span.get_text(strip=True)
                price_match = re.search(r'([\$€£]?)([\d,]+\.?\d*)', price_text)
                if price_match:
                    try:
                        data['price'] = float(price_match.group(2).replace(',', ''))
                        currency_symbol = price_match.group(1)
                        if currency_symbol == '$': data['currency'] = 'USD'
                        elif currency_symbol == '€': data['currency'] = 'EUR'
                        elif currency_symbol == '£': data['currency'] = 'GBP'
                        else: data['currency'] = 'USD' # Default assumption
                    except ValueError:
                        logger.warning(f"[FCP Euro] Could not convert price '{price_match.group(2)}' to float on {url}")
        
        # SKU (FCP Euro's part number)
        sku_div = soup.find('div', class_='listing__sku')
        if sku_div:
            sku_span = sku_div.find_all('span')
            if len(sku_span) > 1: data['part_number'] = sku_span[1].get_text(strip=True)

        # Manufacturer/Brand
        # Try data-brand attribute first as it's more direct
        info_row_with_brand = soup.find('div', class_='listing__infoRow', attrs={'data-brand': True})
        if info_row_with_brand and info_row_with_brand.get('data-brand'):
            data['manufacturer'] = info_row_with_brand['data-brand']
        else:
            brand_div = soup.find('div', class_='listing__brand')
            if brand_div:
                brand_img = brand_div.find('img', alt=True)
                if brand_img: data['manufacturer'] = brand_img['alt']

        # Availability
        fulfillment_div = soup.find('div', class_='listing__fulfillment')
        if fulfillment_div:
            availability_text = fulfillment_div.get_text(strip=True).lower()
            if "available" in availability_text:
                data['availability'] = 'Available'
                desc_div = soup.find('div', class_='listing__fulfillmentDesc')
                if desc_div:
                    desc_span = desc_div.find('span')
                    if desc_span: 
                        detailed_availability = desc_span.get_text(strip=True)
                        if detailed_availability: data['availability'] = detailed_availability # e.g. In Stock
            elif "not available" in availability_text or "sold out" in availability_text: # Check for other terms
                data['availability'] = 'Not Available' # Or Out of Stock
        
        # Image URL
        img_tag = soup.find('img', class_='listing__mainImage')
        if img_tag and img_tag.get('src'):
            img_src = img_tag['src']
            data['image_url'] = urllib.parse.urljoin(url, img_src) # url is the product page url, which is a good base

        # --- Extended Information (Description Tab) ---
        description_tab_panel = soup.find('div', id='description', class_='tabs-panel')
        if description_tab_panel:
            # OE Numbers
            oe_numbers_div = description_tab_panel.find('div', class_='extended__oeNumbers')
            if oe_numbers_div:
                oe_dd = oe_numbers_div.find('dd')
                if oe_dd and oe_dd.get_text(strip=True).lower() != 'n/a':
                    oe_text = oe_dd.get_text(strip=True)
                    data['oem_numbers'] = [normalize_part_number(pn.strip()) for pn in oe_text.split(',') if pn.strip()]
            
            # MFG Numbers (Treat as additional OEM numbers if they are actual part numbers)
            mfg_numbers_div = description_tab_panel.find('div', class_='extended__mfgNumbers')
            if mfg_numbers_div:
                mfg_dd = mfg_numbers_div.find('dd')
                if mfg_dd and mfg_dd.get_text(strip=True).lower() != 'n/a':
                    mfg_text = mfg_dd.get_text(strip=True)
                    # Add these to oem_numbers as well if they are distinct and valid part numbers
                    for pn_str in mfg_text.split(','):
                        norm_pn = normalize_part_number(pn_str.strip())
                        if norm_pn and norm_pn not in data['oem_numbers']:
                            data['oem_numbers'].append(norm_pn)
            
            # Quality (as a specification)
            details_div = description_tab_panel.find('div', class_='extended__details')
            if details_div:
                dt_list = details_div.find_all('dt')
                for dt in dt_list:
                    if dt.get_text(strip=True).lower() == 'quality:':
                        dd = dt.find_next_sibling('dd')
                        if dd: data['specifications']['Quality'] = dd.get_text(strip=True)
                        break

        # Clean up empty OEM numbers list if nothing was added
        if not data['oem_numbers']: data['oem_numbers'] = None # Or keep as empty list based on preference
        if not data['specifications']: data['specifications'] = None

        # Determine Success (Part Name, SKU, and Price are key)
        if data['product_name'] and data['part_number'] and data['price'] is not None:
            data['status'] = 'success'
            logger.info(f"[FCP Euro] Successfully parsed: {url}. PN={data['part_number']}, Price={data['price']}")
            data.pop('error_reason', None)
        else:
            missing = []
            if not data['product_name']: missing.append('product_name')
            if not data['part_number']: missing.append('part_number (SKU)')
            if data['price'] is None: missing.append('price')
            data['error_reason'] = f"Missing essential fields: {', '.join(missing)}"
            logger.warning(f"[FCP Euro] Failed to parse essential info from {url}. Reason: {data['error_reason']}")

        return data

    except Exception as e:
        logger.exception(f"[FCP Euro] Unexpected error parsing product page {url}: {e}")
        data['status'] = 'error'
        data['error_reason'] = f"Parsing Exception: {type(e).__name__} - {str(e)[:100]}"
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
        "product_page_parser": parse_rockauto_product_page, 
        "search_results_parser": extract_rockauto_product_links
    },
    # Add other vendors: fcpeuro.com, autohausaz.com, ecstuning.com, partsgeek.com
    "fcpeuro.com": {
        "search_url_template": "https://www.fcpeuro.com/products?keywords={part_number}", # Example
        "search_method": "GET",
        "product_page_parser": parse_fcpeuro_product_page, # Updated
        "search_results_parser": extract_fcpeuro_product_links # Updated
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
    - For RockAuto: Extracts ALL product links, scrapes EACH product page, 
                    parses *all options* using parse_rockauto_product_page for EACH page.
    - For others: Uses product page parser or LLM fallback on first found link or search page.
    Returns a LIST of found options (dictionaries).
    """
    logger.info(f"--- Processing Vendor: {vendor} for Part: {part.get('description')} (PN: {part.get('part_number')}) ---")
    found_options_list = [] # Accumulates options from all processed links
    part_number = part.get('part_number')
    part_description = part.get('description')
    original_search_url = "N/A"

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
        # --- Debug Log --- 
        logger.debug(f"[{vendor}] Attempting initial search scrape: {original_search_url}")
        scrape_result = await scrape_with_httpx(original_search_url, http_client)
        # --- Debug Log --- 
        scrape_status = scrape_result.get("status", "error")
        logger.debug(f"[{vendor}] Initial search scrape status: {scrape_status}")

    except KeyError as e:
         logger.error(f"Missing key '{e}' needed for vendor {vendor} URL template: {search_url_template}")
         return [create_failure_stub(original_search_url, f"URL Formatting Error: Missing key '{e}'", vendor)]
    except Exception as e:
         logger.exception(f"Error formatting/scraping initial URL for vendor {vendor}: {e}")
         return [create_failure_stub(original_search_url, f"Initial Scrape/URL Error: {e}", vendor)]

    # Check scrape status *after* try block
    if scrape_status != "success" or not scrape_result.get("content"):
        logger.warning(f"[{vendor}] Failed initial search scrape. URL: {original_search_url}. Reason: {scrape_result.get('error_reason', 'Unknown')}")
        # Return the failure stub contained within scrape_result if scrape failed
        # Ensure create_failure_stub is used within scrape_with_httpx on failure
        if isinstance(scrape_result, dict) and scrape_result.get('status') == 'failed':
             return [scrape_result] # Return the failure stub directly
        else:
             # Create a generic one if scrape_result structure is wrong
             return [create_failure_stub(original_search_url, f"Failed to scrape vendor search URL ({scrape_result.get('error_reason', 'Unknown')})", vendor)]
        
    initial_html_content = scrape_result["content"]
    actual_initial_url = scrape_result["original_url"] 

    # --- Strategy 1: Extract Product Link(s) and Process --- 
    product_links = []
    link_extractor = config.get("search_results_parser") 
    
    if link_extractor:
        logger.debug(f"[{vendor}] Attempting link extraction using {link_extractor.__name__}")
        try: 
            product_links = link_extractor(initial_html_content, actual_initial_url)
            # --- Debug Log --- 
            logger.debug(f"[{vendor}] Link extractor found {len(product_links)} links.")
            if product_links:
                 logger.debug(f"[{vendor}] First few extracted links: {product_links[:3]}")
        except Exception as e: 
            logger.error(f"[{vendor}] Error executing link extractor: {e}", exc_info=True)
            product_links = []
            
    # If links found, proceed to process them
    if product_links:
        logger.info(f"[{vendor}] Link extractor found {len(product_links)} product links. Processing applicable ones.")
        
        # --- RockAuto Specific Logic: Process ALL Links --- 
        if vendor == "rockauto.com":
            # --- Debug Log --- 
            logger.info(f"[{vendor}] Entering RockAuto specific block to process {len(product_links)} links.")
            link_tasks = []
            for product_url in product_links:
                link_tasks.append(scrape_and_parse_product_page(
                    product_url=product_url, vendor=vendor, config=config,
                    part_description=part_description, part_number=part_number,
                    http_client=http_client
                ))
            
            # --- Debug Log --- 
            logger.debug(f"[{vendor}] Gathering results for {len(link_tasks)} link processing tasks...")
            results_from_links = await asyncio.gather(*link_tasks) # No return_exceptions needed here, helper handles errors
            logger.debug(f"[{vendor}] Gathered results from link processing.")
            
            # Flatten results (ensure helper returns list)
            for options_list in results_from_links: 
                if isinstance(options_list, list):
                     found_options_list.extend(options_list)
                else:
                     logger.error(f"[{vendor}] scrape_and_parse_product_page helper returned non-list: {type(options_list)}")
                
        # --- Logic for OTHER Vendors: Process ALL links for FCP Euro as well ---
        elif vendor == "fcpeuro.com":
            logger.info(f"[{vendor}] Entering FCP Euro specific block to process {len(product_links)} links.")
            link_tasks = []
            for product_url in product_links:
                link_tasks.append(scrape_and_parse_product_page(
                    product_url=product_url, vendor=vendor, config=config,
                    part_description=part_description, part_number=part_number,
                    http_client=http_client
                ))
            
            logger.debug(f"[{vendor}] Gathering results for {len(link_tasks)} FCP Euro link processing tasks...")
            results_from_links = await asyncio.gather(*link_tasks)
            logger.debug(f"[{vendor}] Gathered results from FCP Euro link processing.")
            
            for options_list_item in results_from_links: # Each item from gather is a list from scrape_and_parse
                if isinstance(options_list_item, list):
                    found_options_list.extend(options_list_item)
                else:
                    logger.error(f"[{vendor}] scrape_and_parse_product_page helper returned non-list for an FCP Euro link: {type(options_list_item)}")
        else:
            # Original logic for other vendors: process only the first link
            if product_links: # Ensure there is at least one link
                logger.info(f"[{vendor}] Processing only the first product link: {product_links[0]}")
                first_product_url = product_links[0]
                options_from_first_link = await scrape_and_parse_product_page(
                    product_url=first_product_url, vendor=vendor, config=config,
                    part_description=part_description, part_number=part_number,
                    http_client=http_client
                )
                found_options_list.extend(options_from_first_link)
            else:
                logger.warning(f"[{vendor}] Link extractor returned no links, cannot process first link.")

        # After processing link(s), if we found options, return them
        if found_options_list:
            logger.info(f"Finished processing vendor {vendor} via product link(s). Returning {len(found_options_list)} options/stubs.")
            return found_options_list
        else:
             logger.warning(f"Processed product links for {vendor}, but no valid options/stubs were generated.")
             found_options_list.append(create_failure_stub(actual_initial_url, "Product link(s) found, but processing/parsing failed for all.", vendor))
             return found_options_list

    # --- Strategy 2: Fallback to LLM on original search results page --- 
    else: # No product links found by extractor (or no extractor defined)
        logger.info(f"[{vendor}] No product links found or extractor not defined. Falling back to LLM on search page: {actual_initial_url}")
        llm_fallback_data = await call_llm_for_extraction_from_content(
            page_content=initial_html_content, url=actual_initial_url,
            original_description=part_description, original_part_number=part_number
        )
        if llm_fallback_data:
             if not llm_fallback_data.get("vendor"): llm_fallback_data["vendor"] = vendor.split('.')[0].capitalize()
             llm_fallback_data["source_url"] = actual_initial_url
             found_options_list.append(llm_fallback_data)
        else:
             found_options_list.append(create_failure_stub(actual_initial_url, "Link extraction failed AND LLM fallback failed", vendor))

    logger.info(f"[{vendor}] Finished processing. Found {len(found_options_list)} total option(s)/stub(s).")
    return found_options_list

# --- NEW HELPER FUNCTION for processing a single product page --- 
async def scrape_and_parse_product_page(
    product_url: str, vendor: str, config: Dict[str, Any],
    part_description: Optional[str], part_number: Optional[str],
    http_client: httpx.AsyncClient
) -> List[Dict[str, Any]]:
    """Helper function to scrape and parse a single product page URL."""
    options_found = [] 
    product_scrape_result = await scrape_with_httpx(product_url, http_client)
    if product_scrape_result.get("status") == "success" and product_scrape_result.get("content"):
        product_html_content = product_scrape_result["content"]
        actual_product_url = product_scrape_result["original_url"]
        product_page_parser = config.get("product_page_parser")
        parsed_data_list = [] 
        if product_page_parser:
            logger.debug(f"Attempting static parse using {product_page_parser.__name__} for {vendor} on URL: {actual_product_url}")
            try:
                parsed_data_item = product_page_parser(product_html_content, actual_product_url) 
                if parsed_data_item: 
                    logger.info(f"Parser {product_page_parser.__name__} returned data with {len(parsed_data_item)} keys for {actual_product_url}")
                    options_found.append(parsed_data_item) 
                else: 
                    logger.warning(f"Parser {product_page_parser.__name__} returned no options for {actual_product_url}")
            except Exception as e: 
                logger.error(f"Error executing parser {product_page_parser.__name__} for {vendor} on {actual_product_url}: {e}", exc_info=True)
        
        if not options_found: # Use LLM only if static parser failed/didn't exist/returned empty
             logger.info(f"Static parse failed/N/A for {vendor} on {actual_product_url}. Using LLM.")
             llm_product_data = await call_llm_for_extraction_from_content(
                 page_content=product_html_content, url=actual_product_url,
                 original_description=part_description, original_part_number=part_number
             )
             if llm_product_data:
                  if not llm_product_data.get("vendor"): llm_product_data["vendor"] = vendor.split('.')[0].capitalize()
                  llm_product_data["source_url"] = actual_product_url
                  options_found.append(llm_product_data)
             else:
                  options_found.append(create_failure_stub(actual_product_url, "Static parse and LLM extraction failed on product page", vendor))
    else:
         options_found.append(create_failure_stub(product_url, f"Failed to scrape product page ({product_scrape_result.get('error_reason', 'Unknown')})", vendor))
    return options_found

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
        all_vendor_options = [] # Reset for each part
        potential_urls_list = [] # Reset for each part
        
        # Create vendor tasks
        vendor_tasks = []
        for vendor, config in VENDOR_CONFIG.items():
            vendor_tasks.append(search_and_extract_vendor_data(
                vendor=vendor, config=config, part=part, 
                vehicle_info=vehicle_info, http_client=http_client
            ))
             
        # Run vendor tasks and process results robustly
        try:
            task_results_or_exceptions = await asyncio.gather(*vendor_tasks, return_exceptions=True)
            for i, result_or_exc in enumerate(task_results_or_exceptions):
                vendor_name = list(VENDOR_CONFIG.keys())[i]
                if isinstance(result_or_exc, Exception):
                    logger.error(f"Task for vendor {vendor_name} failed with exception: {result_or_exc}", exc_info=result_or_exc)
                    # Add a failure stub for task exceptions
                    all_vendor_options.append(create_failure_stub("N/A", f"Task Exception: {result_or_exc}", vendor_name))
                elif isinstance(result_or_exc, list):
                    valid_items = [item for item in result_or_exc if isinstance(item, dict)]
                    invalid_items = [item for item in result_or_exc if not isinstance(item, dict)]
                    if invalid_items:
                        logger.error(f"Vendor {vendor_name} task returned list containing non-dict items: {invalid_items}. Discarding them.")
                    
                    # Enhanced Debug Log
                    logger.info(f"[WORKFLOW DEBUG] Processing results for vendor: '{vendor_name}'. Result type: {type(result_or_exc)}. Number of items in result_or_exc: {len(result_or_exc) if isinstance(result_or_exc, list) else 'N/A'}. Number of valid_items: {len(valid_items)}. First valid item (if any): {valid_items[0] if valid_items else 'None'}")
                        
                    all_vendor_options.extend(valid_items)
                else:
                    logger.error(f"Task for vendor {vendor_name} returned unexpected type: {type(result_or_exc)}. Value: {str(result_or_exc)[:200]}")
                    all_vendor_options.append(create_failure_stub("N/A", f"Unexpected task return type: {type(result_or_exc)}", vendor_name))
        except Exception as e:
            logger.error(f"Error during asyncio.gather for part '{part_description}': {e}", exc_info=True)
            # Add a general failure? Maybe not needed if individual task failures are logged.

        # --- Process URLs --- (Operates on the now validated all_vendor_options)
        potential_urls_list = []
        for option in all_vendor_options: 
            # Should be safe now, but double-check doesn't hurt
            if isinstance(option, dict) and option.get('source_url') and option.get('source_url') != "N/A":
                potential_urls_list.append(option.get('source_url'))
        potential_urls_list = list(dict.fromkeys(potential_urls_list))

        # --- Post-process results: Validate Part Number etc. --- 
        final_options_by_vendor = {} # Initialize as a dictionary
        for option in all_vendor_options:
             # Ensure it's a dictionary before processing
             if not isinstance(option, dict):
                 logger.error(f"[POST-PROCESS] Skipping non-dict item found in all_vendor_options: {type(option)} - {str(option)[:100]}")
                 continue 
            
             vendor_name = option.get('vendor')
             if not vendor_name:
                 logger.warning(f"Option from {option.get('source_url', 'Unknown URL')} is missing vendor name, defaulting to 'Unknown'. Option: {str(option)[:100]}")
                 vendor_name = "Unknown" # Fallback vendor name

             if option.get('status') == 'success':
                extracted_pn = option.get('part_number')
                extracted_oem_list = option.get('oem_numbers')
                validated = True
                if part_number and extracted_pn:
                     if not part_numbers_match(part_number, extracted_pn):
                         normalized_original_pn = normalize_part_number(part_number)
                         if isinstance(extracted_oem_list, list) and normalized_original_pn in extracted_oem_list:
                              logger.info(f"Primary PN mismatch for '{part_description}' from {vendor_name}, but original PN '{part_number}' found in extracted OEM list ({extracted_oem_list}). Accepting option.")
                         else:
                              logger.warning(f"Part number mismatch for '{part_description}' from {vendor_name}. Original: '{part_number}', Extracted: '{extracted_pn}'. Extracted OEMs: {extracted_oem_list}. Discarding option.")
                              validated = False 
                elif part_number and not extracted_pn:
                     logger.debug(f"Could not extract primary part number from {vendor_name} ({option.get('source_url')}) for validation against original PN '{part_number}'. Keeping option for now, but validation is incomplete.")
                
                if not validated:
                     continue 

                found_price = option.get('price')
                option['price_comparison_status'] = "unknown"
                option['price_difference'] = None
                if isinstance(found_price, (int, float)) and isinstance(original_price, (int, float)):
                    difference = round(found_price - original_price, 2)
                    option['price_difference'] = difference
                    if difference < 0: option['price_comparison_status'] = "cheaper"
                    elif difference > 0: option['price_comparison_status'] = "more_expensive"
                    else: option['price_comparison_status'] = "same_price"
            
             if vendor_name not in final_options_by_vendor:
                 final_options_by_vendor[vendor_name] = []
             final_options_by_vendor[vendor_name].append(option)

        # Sort options within each vendor's list
        for vendor_key in final_options_by_vendor:
            final_options_by_vendor[vendor_key].sort(key=lambda x: (
                x.get('status') != 'success', 
                x.get('price') if isinstance(x.get('price'), (int, float)) else float('inf')
            ))
        
        # Append results for this part
        results.append({
            "original_part": part,
            "potential_urls_attempted": potential_urls_list, 
            "found_options": final_options_by_vendor # Use the new dictionary structure
        })
        
        total_successful_options = sum(
            1 for options_list in final_options_by_vendor.values() 
            for opt in options_list 
            if isinstance(opt, dict) and opt.get('status') == 'success'
        )
        logger.info(f"--- Finished research for: '{part_description}'. Found {total_successful_options} valid options, organized by {len(final_options_by_vendor)} vendors. ---")
        await asyncio.sleep(2)

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