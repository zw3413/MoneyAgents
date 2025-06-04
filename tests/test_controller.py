import pytest
from app.controller import AnalysisController
from datetime import datetime

@pytest.fixture
def controller():
    return AnalysisController()

@pytest.mark.asyncio
async def test_quick_analysis(controller):
    result = await controller.get_quick_analysis("AAPL")
    assert result is not None
    assert "ticker" in result
    assert result["ticker"] == "AAPL"
    assert "technical_summary" in result
    assert "news_summary" in result
    assert "analysis_date" in result
    assert "status" in result

@pytest.mark.asyncio
async def test_comprehensive_analysis(controller):
    result = await controller.analyze_stock("AAPL", "1m")
    assert result is not None
    assert result["ticker"] == "AAPL"
    assert result["timeframe"] == "1m"
    assert "fundamental_analysis" in result
    assert "technical_analysis" in result
    assert "news_analysis" in result
    assert "sentiment_analysis" in result
    assert "strategy_recommendation" in result
    assert "metadata" in result

@pytest.mark.asyncio
async def test_invalid_ticker(controller):
    result = await controller.analyze_stock("INVALID")
    assert result is not None
    assert result["status"] == "error"
    assert "error" in result

@pytest.mark.asyncio
async def test_timeframe_parsing(controller):
    result = await controller.analyze_stock("AAPL", "7d")
    assert result is not None
    assert result["timeframe"] == "7d"
    metadata = result.get("metadata", {})
    start_date = datetime.fromisoformat(metadata.get("start_date", ""))
    end_date = datetime.fromisoformat(metadata.get("end_date", ""))
    assert (end_date - start_date).days == 7 