import logging
import re
import urllib.parse
import json
from typing import Dict, List, Any, Optional

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# --- Helper Functions ---

def normalize_part_number(pn: Optional[str]) -> Optional[str]:
    """Removes common separators and converts to uppercase for comparison."""
    if not pn: return None
    return re.sub(r'[-\s.]', '', str(pn)).upper()

# --- RockAuto Specific Parsers ---

def extract_rockauto_product_links(html_content: str, url: str) -> List[str]:
    """Parses RockAuto search results page to extract links to individual product pages ('More Info' links)."""
    logger.debug(f"Attempting to extract RockAuto product links from {url}")
    if not html_content:
        logger.warning(f"No HTML content provided for RockAuto link extraction from {url}")
        return []
        
    soup = BeautifulSoup(html_content, 'lxml')
    product_links = []
    
    parts_container = soup.find('div', class_='listing-container-border')
    
    if not parts_container:
        logger.warning(f"Could not find main parts container ('div.listing-container-border') on RockAuto page: {url}")
        return [] 
        
    part_blocks = parts_container.find_all('tbody', id=lambda x: x and x.startswith('listingcontainer['))
    logger.debug(f"Found {len(part_blocks)} part blocks (tbody[id^=listingcontainer]) in container for {url}")

    if not part_blocks:
         logger.warning(f"Found the main container, but no individual part blocks (tbody[id^=listingcontainer]) within it for URL: {url}")
         return []

    for part_block in part_blocks:
        link_tag = part_block.find('a', class_='ra-btn-moreinfo', href=lambda href: href and 'moreinfo.php' in href)
        if link_tag:
            href = link_tag.get('href')
            if href:
                 absolute_url = urllib.parse.urljoin(url, href) 
                 product_links.append(absolute_url)
            else:
                 logger.warning(f"Found 'more info' link tag but it has no href in block {part_block.get('id', 'N/A')} on {url}")

    unique_links = list(dict.fromkeys(product_links))
    if unique_links:
        logger.info(f"Extracted {len(unique_links)} unique product page ('More Info') links from RockAuto results page: {url}")
    else:
        logger.warning(f"Extraction complete, but no 'More Info' links were found on RockAuto results page: {url}")
        
    return unique_links


# Function to parse RockAuto 'More Info' pages, extracting all purchase options.
def parse_rockauto_product_page(html_content: str, url: str) -> List[Dict[str, Any]]:
    """
    Parses a RockAuto 'More Info' product page.
    
    Extracts common details and then iterates through purchase options 
    (e.g., Wholesaler Closeout, Regular Inventory) found in dropdowns/hidden data.
    
    Returns:
        A list of dictionaries, where each dictionary represents one distinct 
        purchase option found on the page. Returns an empty list if parsing fails 
        or no valid options are found.
    """
    logger.debug(f"Attempting to parse RockAuto PRODUCT page ({url}) for all options.")
    results_list = []
    if not html_content:
        logger.warning(f"No HTML content provided for RockAuto product page parse: {url}")
        return [] 
        
    soup = BeautifulSoup(html_content, 'lxml')
    
    # --- Extract Common Information ---
    common_data = {
        'url': url,
        'vendor': 'RockAuto',
        'part_number': None,
        'manufacturer': None,
        'image_url': None,
        'specifications': None,
        'oem_numbers': None,
        'base_description': None, 
    }

    try:
        # Find the buybox first (often contains manufacturer/PN even if options are elsewhere)
        buybox_tbody = soup.find('tbody', id='mibuybox') 
        if buybox_tbody:
            mfr_span = buybox_tbody.find('span', class_='listing-final-manufacturer')
            if mfr_span: common_data['manufacturer'] = mfr_span.get_text(strip=True)
            pn_span = buybox_tbody.find('span', class_='listing-final-partnumber')
            if pn_span: common_data['part_number'] = pn_span.get_text(strip=True)

        # Image (usually outside buybox on 'more info')
        img_tag = soup.find('img', class_='listing-inline-image-moreinfo')
        if img_tag and img_tag.get('src'):
            img_src = img_tag.get('src')
            if img_src: common_data['image_url'] = urllib.parse.urljoin(url, img_src)

        # Basic Description
        desc_section = soup.find('section', attrs={'aria-label': 'Part Description'})
        if desc_section: common_data['base_description'] = desc_section.get_text(strip=True)

        # Specifications
        specs_table = soup.find('table', class_='moreinfotable')
        if specs_table and specs_table.find('th', string=lambda t: 'Specifications' in t):
            specs = {}
            rows = specs_table.find_all('tr')
            for row in rows[1:]:
                cols = row.find_all('td')
                if len(cols) == 2:
                    key = cols[0].get_text(strip=True)
                    value = cols[1].get_text(strip=True)
                    if key: specs[key] = value
            if specs: common_data['specifications'] = specs

        # OEM/Interchange Numbers
        oem_section = soup.find('section', attrs={'aria-label': 'Alternate/OEM Part Number(s)'})
        if oem_section:
            oem_text = oem_section.get_text(strip=True)
            oem_string_only = oem_text.replace('OEM / Interchange Numbers:', '').strip()
            potential_oems = re.split(r'[\s,]+', oem_string_only)
            normalized_oems = [norm_pn for pn in potential_oems if (norm_pn := normalize_part_number(pn))]
            if normalized_oems: common_data['oem_numbers'] = normalized_oems
        
        # If we didn't get basic manufacturer/PN, log warning but continue
        if not common_data['manufacturer'] or not common_data['part_number']:
             logger.warning(f"Could not extract common Manufacturer/Part Number from {url}")

        # --- Extract Purchase Options ---
        option_select = soup.find('select', attrs={'name': lambda x: x and x.startswith('optionchoice[')})
        
        if not option_select:
            logger.warning(f"Could not find option <select> element on {url}. Cannot extract options.")
            return [] 

        option_elements = option_select.find_all('option')
        
        if not option_elements:
            logger.warning(f"Found option <select> but no <option> tags within it on {url}.")
            return []

        select_name = option_select.get('name')
        group_match = re.search(r'\[(\d+)\]', select_name)
        if not group_match:
             logger.error(f"Could not extract group index from select name '{select_name}' on {url}. Cannot match hidden data.")
             return []
        group_index = group_match.group(1)

        # Iterate through the <option> tags in the <select>
        for option_tag in option_elements:
            option_key = option_tag.get('value')
            option_text_raw = option_tag.get_text(strip=True)
            
            if not option_key or not option_text_raw or option_text_raw == u'\xa0': 
                continue

            option_desc = re.sub(r'\(.*?\)\s*\^?$', '', option_text_raw).strip()
            option_desc = option_desc.replace('[', '').replace(']', '').strip()
            
            price_input_id = f"pricebreakdown[{group_index}][{option_key}]"
            price_input = soup.find('input', {'type': 'hidden', 'id': price_input_id})
            
            option_price = None
            option_currency = None
            
            if price_input and price_input.get('value'):
                try:
                    price_data = json.loads(price_input.get('value'))
                    price_text = price_data.get('v', '')
                    price_match = re.search(r'[$€£]?([\d,]+\.?\d*)', price_text)
                    if price_match:
                        try:
                            option_price = float(price_match.group(1).replace(',', ''))
                            if '$' in price_text or 'USD' in price_text: option_currency = 'USD'
                            elif '€' in price_text or 'EUR' in price_text: option_currency = 'EUR'
                            elif '£' in price_text or 'GBP' in price_text: option_currency = 'GBP'
                            elif 'CAD' in price_text: option_currency = 'CAD'
                            else: option_currency = 'USD'
                        except ValueError:
                            logger.warning(f"Could not convert price '{price_match.group(1)}' for option '{option_key}' on {url}")
                    else:
                         logger.warning(f"Could not extract numeric price from hidden input '{price_text}' for option '{option_key}' on {url}")
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Failed to parse price JSON for option '{option_key}' on {url}: {e}")
            else:
                 logger.warning(f"Could not find hidden price input for option '{option_key}' (ID: {price_input_id}) on {url}")

            option_result = {
                **common_data, 
                'status': 'error', 
                'description': option_desc or common_data.get('base_description'), 
                'price': option_price,
                'currency': option_currency,
                'availability': 'Available' if option_price is not None else 'Unknown', 
                'source_url': url, 
            }
            
            if option_price is not None:
                option_result['status'] = 'success'
                logger.debug(f"Successfully parsed option '{option_desc}' with price {option_price} on {url}")
                results_list.append(option_result)
            else:
                logger.warning(f"Skipping option '{option_desc}' because price could not be extracted on {url}")

        if not results_list:
             logger.warning(f"Parsing completed, but no valid purchase options with prices found on {url}")

        return results_list

    except Exception as e:
        logger.exception(f"Unexpected error parsing RockAuto product page options {url}: {e}")
        return [] 