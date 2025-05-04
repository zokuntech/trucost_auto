from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Dict
from api.upload import router as upload_router
from api.audit import router as audit_router
from agents.quote_parser.parser import router as quote_parser_router
from agents.parts_research.researcher import router as parts_research_router
from config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(
    title="TruCost Auto Backend",
    description="Backend service for car repair audit platform",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload_router, prefix="/api/v1", tags=["upload"])
app.include_router(audit_router, prefix="/api/v1", tags=["audit"])
app.include_router(quote_parser_router, prefix="/api/v1", tags=["quote-parser"])
app.include_router(parts_research_router, prefix="/api/v1", tags=["parts-research"])

@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint for health check"""
    return {"status": "healthy", "message": "TruCost Auto Backend is running"}

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 