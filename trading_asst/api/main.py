from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from .controller import AnalysisController

app = FastAPI(
    title="Trading Assistant API",
    description="Multi-agent stock analysis system powered by LangChain",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize controller
controller = AnalysisController()

class AnalysisRequest(BaseModel):
    ticker: str
    timeframe: Optional[str] = "1y"

class QuickAnalysisRequest(BaseModel):
    ticker: str

@app.get("/")
async def health_check():
    """Basic health check endpoint for Cloud Run."""
    return {"status": "ok"}

@app.post("/api/v1/analyze")
async def analyze_stock(request: AnalysisRequest):
    """
    Perform comprehensive stock analysis using all agents.
    """
    try:
        result = await controller.analyze_stock(
            ticker=request.ticker,
            timeframe=request.timeframe
        )
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/quick-analysis")
async def quick_analysis(request: QuickAnalysisRequest):
    """
    Perform quick analysis focusing on technical and news aspects.
    """
    try:
        result = await controller.get_quick_analysis(request.ticker)
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health")
async def health_check():
    """
    API health check endpoint.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def cloud_run_health_check():
    """Health check endpoint for Cloud Run."""
    return {
        "status": "healthy",
        "service": "trading-asst",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Trading Assistant API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "api_health": "/api/v1/health",
            "analyze": "/api/v1/analyze",
            "quick_analysis": "/api/v1/quick-analysis"
        }
    }

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port) 