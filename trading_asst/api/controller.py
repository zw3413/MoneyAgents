from typing import Dict, Any
from datetime import datetime, timedelta
from trading_asst.core.agents.analysis_agents import (
    FundamentalAnalysisAgent,
    TechnicalAnalysisAgent,
    NewsAnalysisAgent,
    SentimentAnalysisAgent,
    StrategyAgent
)
from trading_asst.core.config.config import get_settings
from langchain.memory import ConversationBufferMemory
import asyncio

class AnalysisController:
    def __init__(self):
        settings = get_settings()
        
        # Initialize shared memory for context preservation using new pattern
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize all agents
        self.fundamental_agent = FundamentalAnalysisAgent(
            settings.FUNDAMENTAL_AGENT_MODEL,
            memory=self.memory
        )
        self.technical_agent = TechnicalAnalysisAgent(
            settings.TECHNICAL_AGENT_MODEL,
            memory=self.memory
        )
        self.news_agent = NewsAnalysisAgent(
            settings.NEWS_AGENT_MODEL,
            memory=self.memory
        )
        self.sentiment_agent = SentimentAnalysisAgent(
            settings.SENTIMENT_AGENT_MODEL,
            memory=self.memory
        )
        self.strategy_agent = StrategyAgent(
            settings.DEFAULT_LLM_MODEL,
            memory=self.memory
        )

    async def analyze_stock(self, ticker: str, timeframe: str = "1y") -> Dict[str, Any]:
        """
        Perform comprehensive stock analysis using all agents.
        
        Args:
            ticker (str): Stock ticker symbol
            timeframe (str): Analysis timeframe (e.g., "1d", "1w", "1m", "1y")
            
        Returns:
            Dict[str, Any]: Comprehensive analysis results
        """
        try:
            # Calculate date range based on timeframe
            end_date = datetime.now()
            if timeframe.endswith('d'):
                start_date = end_date - timedelta(days=int(timeframe[:-1]))
            elif timeframe.endswith('w'):
                start_date = end_date - timedelta(weeks=int(timeframe[:-1]))
            elif timeframe.endswith('m'):
                start_date = end_date - timedelta(days=int(timeframe[:-1]) * 30)
            elif timeframe.endswith('y'):
                start_date = end_date - timedelta(days=int(timeframe[:-1]) * 365)
            else:
                start_date = end_date - timedelta(days=365)  # Default to 1 year

            # Run all analyses in parallel
            analyses = await asyncio.gather(
                self.fundamental_agent.analyze(
                    f"Analyze the fundamental factors for {ticker}"
                ),
                self.technical_agent.analyze(
                    f"Analyze the technical indicators for {ticker} from {start_date.date()} to {end_date.date()}"
                ),
                self.news_agent.analyze(
                    f"Analyze recent news and their impact on {ticker}"
                ),
                self.sentiment_agent.analyze(
                    f"Analyze market sentiment for {ticker}"
                )
            )
            
            # Generate comprehensive strategy
            strategy = await self.strategy_agent.generate_strategy(
                ticker=ticker,
                fundamental_analysis=analyses[0],
                technical_analysis=analyses[1],
                news_analysis=analyses[2],
                sentiment_analysis=analyses[3]
            )
            
            return {
                "ticker": ticker,
                "timeframe": timeframe,
                "analysis_date": end_date.isoformat(),
                "fundamental_analysis": analyses[0],
                "technical_analysis": analyses[1],
                "news_analysis": analyses[2],
                "sentiment_analysis": analyses[3],
                "strategy_recommendation": strategy,
                "metadata": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "ticker": ticker,
                "timeframe": timeframe
            }

    async def get_quick_analysis(self, ticker: str) -> Dict[str, Any]:
        """
        Perform a quick analysis focusing only on technical and news aspects.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            Dict[str, Any]: Quick analysis results
        """
        try:
            # Run quick analyses in parallel
            technical, news = await asyncio.gather(
                self.technical_agent.analyze(
                    f"Give a quick technical overview for {ticker} focusing on immediate trends"
                ),
                self.news_agent.analyze(
                    f"Summarize the most recent significant news for {ticker}"
                )
            )
            
            return {
                "ticker": ticker,
                "analysis_date": datetime.now().isoformat(),
                "technical_summary": technical,
                "news_summary": news,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "ticker": ticker
            } 