from typing import Dict, Any, List, Type, Optional
from pydantic import BaseModel, Field
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from langchain_community.tools import BaseTool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

class NewsInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol")
    days_back: int = Field(default=7, description="Number of days to look back for news")

class SentimentInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol")

class NewsSearchTool(BaseTool):
    name: str = "search_stock_news"
    description: str = "Search for recent news articles about a stock"
    args_schema: Type[BaseModel] = NewsInput

    def _run(self, input: Dict[str, Any]) -> Dict[str, Any]:
        ticker = input["ticker"]
        days_back = input.get("days_back", 7)
        
        try:
            # This is a simplified example. In production, use proper news APIs
            url = f"https://finance.yahoo.com/quote/{ticker}/news"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            news_items = []
            for article in soup.find_all('div', {'class': 'Py(14px)'}):
                title = article.find('h3').text if article.find('h3') else ""
                link = article.find('a')['href'] if article.find('a') else ""
                news_items.append({
                    "title": title,
                    "link": link,
                    "source": "Yahoo Finance"
                })
            
            return {
                "ticker": ticker,
                "period": f"Last {days_back} days",
                "news": news_items[:10]  # Limit to 10 items
            }
        except Exception as e:
            return {"error": str(e)}

    async def _arun(self, input: Dict[str, Any]):
        raise NotImplementedError("Async version not implemented")

class SentimentAnalysisTool(BaseTool):
    name: str = "analyze_market_sentiment"
    description: str = "Analyze market sentiment from social media and news"
    args_schema: Type[BaseModel] = SentimentInput
    embeddings: Optional[OpenAIEmbeddings] = None
    vector_store: Optional[FAISS] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        self.embeddings = OpenAIEmbeddings()

    def _run(self, input: Dict[str, Any]) -> Dict[str, Any]:
        ticker = input["ticker"]
        try:
            # Get news and social media content
            news_data = self._fetch_news(ticker)
            social_data = self._fetch_social_media(ticker)
            
            # Combine all text content
            all_content = news_data + social_data
            
            # Create vector store if not exists
            if not self.vector_store:
                self.vector_store = FAISS.from_texts(
                    all_content,
                    self.embeddings,
                    metadatas=[{"source": "news" if i < len(news_data) else "social"} 
                              for i in range(len(all_content))]
                )
            
            # Analyze sentiment (simplified example)
            sentiment_scores = self._analyze_sentiment(all_content)
            
            return {
                "ticker": ticker,
                "overall_sentiment": self._aggregate_sentiment(sentiment_scores),
                "news_sentiment": sum(sentiment_scores[:len(news_data)]) / len(news_data) if news_data else 0,
                "social_sentiment": sum(sentiment_scores[len(news_data):]) / len(social_data) if social_data else 0,
                "total_sources": len(all_content)
            }
        except Exception as e:
            return {"error": str(e)}

    def _fetch_news(self, ticker: str) -> List[str]:
        # Simplified news fetching (replace with actual news API in production)
        return [
            f"Sample news content about {ticker}",
            f"Another news article about {ticker}"
        ]

    def _fetch_social_media(self, ticker: str) -> List[str]:
        # Simplified social media fetching (replace with actual API calls)
        return [
            f"Reddit discussion about {ticker}",
            f"Twitter sentiment about {ticker}"
        ]

    def _analyze_sentiment(self, texts: List[str]) -> List[float]:
        # Simplified sentiment scoring (replace with actual sentiment analysis)
        return [0.5 for _ in texts]  # Neutral sentiment

    def _aggregate_sentiment(self, scores: List[float]) -> str:
        if not scores:
            return "Neutral"
        avg_score = sum(scores) / len(scores)
        if avg_score > 0.6:
            return "Bullish"
        elif avg_score < 0.4:
            return "Bearish"
        return "Neutral"

    async def _arun(self, input: Dict[str, Any]):
        raise NotImplementedError("Async version not implemented") 