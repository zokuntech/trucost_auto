from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal

class FoundPartOption(BaseModel):
    """Represents a single potential part option found during research."""
    product_name: Optional[str] = Field(None, description="Name of the product found")
    price: Optional[float] = Field(None, description="Price of the product found")
    currency: Optional[str] = Field(None, description="Currency of the price (e.g., USD, $)")
    availability: Optional[str] = Field(None, description="Availability status (e.g., In Stock, Ships Tomorrow)")
    part_number: Optional[str] = Field(None, description="Part number extracted from the page")
    vendor: Optional[str] = Field(None, description="Vendor/Seller identified on the page")
    source_url: str = Field(..., description="The specific URL where this option was found")
    status: Literal["success", "failed", "failed_validation"] = Field(..., description="Status of processing this option")
    error_reason: Optional[str] = Field(None, description="Reason for failure, if status is 'failed'")
    price_comparison_status: Literal["cheaper", "more_expensive", "same_price", "unknown", "comparison_error"] = Field("unknown", description="Comparison vs original quote price")
    price_difference: Optional[float] = Field(None, description="Numeric difference vs original quote price (found_price - original_price)")

class PartResearchResult(BaseModel):
    """Represents the overall research result for a single part from the original quote."""
    original_part: Dict[str, Any] = Field(..., description="The original part details from the quote")
    potential_urls_attempted: List[str] = Field(default_factory=list, description="URLs selected by AI/Search for scraping")
    found_options: List[FoundPartOption] = Field(default_factory=list, description="List of potential options found (including failures)")

# Example Usage (Optional, for testing)
# if __name__ == "__main__":
#     failure_option = FoundPartOption(
#         source_url="https://example.com/failed",
#         status="failed",
#         error_reason="Scraping timeout",
#         vendor="Example Vendor"
#     )
#     success_option = FoundPartOption(
#         product_name="Example Part",
#         price=19.99,
#         currency="USD",
#         availability="In Stock",
#         part_number="XYZ-123",
#         vendor="Example Vendor",
#         source_url="https://example.com/success",
#         status="success",
#         price_comparison_status="cheaper",
#         price_difference=-5.01
#     )
#     research = PartResearchResult(
#         original_part={"description": "Test Part", "part_number": "XYZ-123", "unit_price": 25.00},
#         potential_urls_attempted=["https://example.com/failed", "https://example.com/success"],
#         found_options=[failure_option, success_option]
#     )
#     print(research.model_dump_json(indent=2)) 