import re
from typing import Optional, Dict, Any

# Moved from researcher.py
def normalize_part_number(part_number: str) -> str:
    """Normalizes a part number by removing non-alphanumeric characters and converting to uppercase."""
    if not isinstance(part_number, str):
        return "" # Or raise an error, depending on desired handling
    
    # Remove common vendor prefixes (e.g., MER-, ELR-, etc.)
    pn = re.sub(r'^[A-Z]+-', '', part_number.upper())
    
    # Remove all non-alphanumeric characters
    return re.sub(r'[^A-Z0-9]', '', pn)

# Moved from researcher.py
def create_failure_stub(url: str, reason: str, vendor_hint: Optional[str] = None) -> Dict[str, Any]:
    """Creates a standardized dictionary for failed research attempts."""
    vendor = vendor_hint or "Unknown"
    if vendor == "Unknown": # Try to infer if hint not provided
        if "rockauto.com" in url: vendor = "RockAuto"
        elif "autozone.com" in url: vendor = "AutoZone"
        elif "amazon.com" in url: vendor = "Amazon"
        elif "fcpeuro.com" in url: vendor = "FCP Euro"
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