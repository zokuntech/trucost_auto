from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from bs4 import BeautifulSoup
import re

class PartImage(BaseModel):
    """Represents the image information for a part."""
    small_url: str = Field(..., description="URL of the small version of the image")
    large_url: str = Field(..., description="URL of the large version of the image")
    alt_text: str = Field(..., description="Alt text for the image")
    title: str = Field(..., description="Title of the image")

class PartAvailability(BaseModel):
    """Represents the availability information for a part."""
    status: str = Field(..., description="Availability status (e.g., 'Ships Tomorrow')")
    description: str = Field(..., description="Detailed description of availability")
    icon_url: str = Field(..., description="URL of the availability status icon")

class PartPricing(BaseModel):
    """Represents the pricing information for a part."""
    sale_price: float = Field(..., description="Current sale price")
    regular_price: float = Field(..., description="Regular price")
    list_price: float = Field(..., description="List price")
    currency: str = Field(default="USD", description="Currency of the prices")

class PartSearchResult(BaseModel):
    """Represents a single part from the search results."""
    part_id: str = Field(..., description="Unique identifier for the part")
    name: str = Field(..., description="Name of the part")
    part_number: Optional[str] = Field(None, description="Part number")
    brand: str = Field(..., description="Brand of the part")
    description: Optional[str] = Field(None, description="Description of the part")
    compatible_vehicles: List[str] = Field(default_factory=list, description="List of compatible vehicles")
    image: PartImage = Field(..., description="Image information")
    availability: PartAvailability = Field(..., description="Availability information")
    pricing: PartPricing = Field(..., description="Pricing information")
    url: str = Field(..., description="URL to the part's detail page")
    timestamp: datetime = Field(default_factory=datetime.now, description="When this result was parsed")

class SearchResults(BaseModel):
    """Represents a collection of part search results."""
    parts: List[PartSearchResult] = Field(default_factory=list, description="List of found parts")
    total_results: int = Field(..., description="Total number of results found")
    search_timestamp: datetime = Field(default_factory=datetime.now, description="When the search was performed")
    source_url: str = Field(..., description="URL where the search was performed")

def extract_price(price_text: str) -> float:
    """Extract numeric price from text."""
    if not price_text:
        return 0.0
    # Remove currency symbols and convert to float
    price = re.sub(r'[^\d.]', '', price_text)
    return float(price) if price else 0.0

def parse_part_from_html(html_element: BeautifulSoup) -> PartSearchResult:
    """
    Parse a single part result from the HTML element.
    """
    # Extract part ID from the div id
    part_id = html_element.get('id', '').replace('part', '')
    
    # Extract image information
    img_container = html_element.select_one('.div-part-image-container')
    img = img_container.select_one('img') if img_container else None
    img_link = img_container.select_one('a') if img_container else None
    
    image = PartImage(
        small_url=img.get('src', '') if img else '',
        large_url=img_link.get('href', '') if img_link else '',
        alt_text=img.get('alt', '') if img else '',
        title=img.get('title', '') if img else ''
    )
    
    # Extract part information
    info_container = html_element.select_one('.div-part-info-container')
    name_link = info_container.select_one('.name a') if info_container else None
    name = name_link.get_text(strip=True) if name_link else ''
    url = name_link.get('href', '') if name_link else ''
    
    # Extract brand and part number from name
    brand = ''
    part_number = ''
    if name:
        parts = name.split('|')
        if len(parts) > 1:
            brand = parts[0].strip()
            part_number = parts[1].strip()
    
    # Extract compatible vehicles
    vehicles_div = info_container.select_one('.appvehicles div') if info_container else None
    compatible_vehicles = []
    if vehicles_div:
        vehicles_text = vehicles_div.get_text(strip=True)
        compatible_vehicles = [v.strip() for v in vehicles_text.split(',')]
    
    # Extract availability information
    availability_div = html_element.select_one('.div-stock-availability')
    availability = PartAvailability(
        status=availability_div.select_one('.availability-message').get_text(strip=True) if availability_div else '',
        description=availability_div.select_one('.availability-description').get_text(strip=True) if availability_div else '',
        icon_url=availability_div.select_one('.availability-icon img').get('src', '') if availability_div else ''
    )
    
    # Extract pricing information
    pricing_div = html_element.select_one('.div-part-price')
    sale_price = extract_price(pricing_div.select_one('.saleprice .price').get_text(strip=True) if pricing_div else '')
    regular_price = extract_price(pricing_div.select_one('.regprice .price').get_text(strip=True) if pricing_div else '')
    list_price = extract_price(pricing_div.select_one('.listprice .price').get_text(strip=True) if pricing_div else '')
    
    pricing = PartPricing(
        sale_price=sale_price,
        regular_price=regular_price,
        list_price=list_price
    )
    
    return PartSearchResult(
        part_id=part_id,
        name=name,
        part_number=part_number,
        brand=brand,
        description='',  # Description field is empty in the example
        compatible_vehicles=compatible_vehicles,
        image=image,
        availability=availability,
        pricing=pricing,
        url=url
    )

def parse_search_results(html_content: str, source_url: str) -> SearchResults:
    """
    Parse the entire search results page.
    """
    soup = BeautifulSoup(html_content, 'lxml')
    part_divs = soup.select('.div-part')
    
    parts = []
    for part_div in part_divs:
        try:
            part = parse_part_from_html(part_div)
            parts.append(part)
        except Exception as e:
            print(f"Error parsing part: {str(e)}")
            continue
    
    return SearchResults(
        parts=parts,
        total_results=len(parts),
        source_url=source_url
    ) 