import logging
import re
import urllib.parse
from typing import Optional, Dict, List, Any
from bs4 import BeautifulSoup

from .utils import create_failure_stub, normalize_part_number

logger = logging.getLogger(__name__)

# Moved from researcher.py
def extract_fcpeuro_product_links(html_content: str, url: str) -> List[str]:
    """Parses FCP Euro search results to find product links."""
    logger.info(f"[FCP Euro] Attempting to extract product links from {url}")
    if not html_content:
        logger.warning(f"[FCP Euro] No HTML content provided for link extraction from {url}")
        return []

    soup = BeautifulSoup(html_content, 'lxml')
    product_links = []
    base_url = "https://www.fcpeuro.com"

    product_divs = soup.find_all('div', class_='hit')

    if not product_divs:
        logger.warning(f"[FCP Euro] No 'div.hit' elements found on {url}. Link extraction might fail or page structure may have changed.")
        return []

    for div in product_divs:
        data_href = div.get('data-href')
        if data_href:
            absolute_url = urllib.parse.urljoin(base_url, data_href)
            product_links.append(absolute_url)
        else:
            logger.debug(f"[FCP Euro] Found a 'div.hit' without a 'data-href' attribute on {url}. Skipping.")

    unique_links = list(dict.fromkeys(product_links))
    if unique_links:
        logger.info(f"[FCP Euro] Extracted {len(unique_links)} unique product links from {url}")
    else:
        logger.warning(f"[FCP Euro] Link extraction complete, but no product links were found on {url}")
        
    return unique_links

# Moved from researcher.py
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
        'part_number': None,
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
        name_tag = soup.find('h1', class_='listing__name')
        if name_tag: data['product_name'] = name_tag.get_text(strip=True)

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
                        else: data['currency'] = 'USD'
                    except ValueError:
                        logger.warning(f"[FCP Euro] Could not convert price '{price_match.group(2)}' to float on {url}")
        
        sku_div = soup.find('div', class_='listing__sku')
        if sku_div:
            sku_span = sku_div.find_all('span')
            if len(sku_span) > 1: data['part_number'] = sku_span[1].get_text(strip=True)

        info_row_with_brand = soup.find('div', class_='listing__infoRow', attrs={'data-brand': True})
        if info_row_with_brand and info_row_with_brand.get('data-brand'):
            data['manufacturer'] = info_row_with_brand['data-brand']
        else:
            brand_div = soup.find('div', class_='listing__brand')
            if brand_div:
                brand_img = brand_div.find('img', alt=True)
                if brand_img: data['manufacturer'] = brand_img['alt']

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
                        if detailed_availability: data['availability'] = detailed_availability
            elif "not available" in availability_text or "sold out" in availability_text:
                data['availability'] = 'Not Available'
        
        img_tag = soup.find('img', class_='listing__mainImage')
        if img_tag and img_tag.get('src'):
            img_src = img_tag['src']
            data['image_url'] = urllib.parse.urljoin(url, img_src)

        description_tab_panel = soup.find('div', id='description', class_='tabs-panel')
        if description_tab_panel:
            oe_numbers_div = description_tab_panel.find('div', class_='extended__oeNumbers')
            if oe_numbers_div:
                oe_dd = oe_numbers_div.find('dd')
                if oe_dd and oe_dd.get_text(strip=True).lower() != 'n/a':
                    oe_text = oe_dd.get_text(strip=True)
                    oem_numbers = [normalize_part_number(pn.strip()) for pn in oe_text.split(',') if pn.strip()]
                    if oem_numbers:
                        data['oem_numbers'] = oem_numbers
                        logger.debug(f"[FCP Euro] Extracted OEM numbers: {oem_numbers}")
            
            mfg_numbers_div = description_tab_panel.find('div', class_='extended__mfgNumbers')
            if mfg_numbers_div:
                mfg_dd = mfg_numbers_div.find('dd')
                if mfg_dd and mfg_dd.get_text(strip=True).lower() != 'n/a':
                    mfg_text = mfg_dd.get_text(strip=True)
                    for pn_str in mfg_text.split(','):
                        norm_pn = normalize_part_number(pn_str.strip())
                        if norm_pn and norm_pn not in data['oem_numbers']:
                            data['oem_numbers'].append(norm_pn)
            
            details_div = description_tab_panel.find('div', class_='extended__details')
            if details_div:
                dt_list = details_div.find_all('dt')
                for dt in dt_list:
                    if dt.get_text(strip=True).lower() == 'quality:':
                        dd = dt.find_next_sibling('dd')
                        if dd: data['specifications']['Quality'] = dd.get_text(strip=True)
                        break

        if not data['specifications']: data['specifications'] = None

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

        logger.debug(f"[FCP Euro] Final oem_numbers state for {url}: {data['oem_numbers']}")
        return data

    except Exception as e:
        logger.exception(f"[FCP Euro] Unexpected error parsing product page {url}: {e}")
        data['status'] = 'error'
        data['error_reason'] = f"Parsing Exception: {type(e).__name__} - {str(e)[:100]}"
        return data 