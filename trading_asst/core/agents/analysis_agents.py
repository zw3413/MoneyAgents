from typing import List, Dict, Any
from .base_agent import BaseAnalysisAgent
from trading_asst.core.tools.financial_tools import FinancialDataTool, TechnicalDataTool
from trading_asst.core.tools.news_sentiment_tools import NewsSearchTool, SentimentAnalysisTool
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

class FundamentalAnalysisAgent(BaseAnalysisAgent):
    def __init__(self, llm_model: str, memory: ConversationBufferMemory = None):
        tools = [FinancialDataTool()]
        prompt_template = """You are a fundamental analysis expert.
        Focus on analyzing company financials, ratios, and business metrics.
        
        Key areas to consider:
        1. Revenue and profit growth
        2. Margins and profitability
        3. Balance sheet strength
        4. Cash flow analysis
        5. Valuation metrics
        
        Tools available: {tools}
        
        Use the above tools to help you answer the following question:
        {input}
        
        {agent_scratchpad}
        """
        super().__init__(llm_model, tools, memory, prompt_template)

class TechnicalAnalysisAgent(BaseAnalysisAgent):
    def __init__(self, llm_model: str, memory: ConversationBufferMemory = None):
        tools = [TechnicalDataTool()]
        prompt_template = """You are a technical analysis expert.
        Focus on price patterns, indicators, and chart analysis.
        
        Key areas to consider:
        1. Trend analysis
        2. Support and resistance levels
        3. Technical indicators (MACD, RSI, etc.)
        4. Volume analysis
        5. Chart patterns
        
        Tools available: {tools}
        
        Use the above tools to help you answer the following question:
        {input}
        
        {agent_scratchpad}
        """
        super().__init__(llm_model, tools, memory, prompt_template)

class NewsAnalysisAgent(BaseAnalysisAgent):
    def __init__(self, llm_model: str, memory: ConversationBufferMemory = None):
        tools = [NewsSearchTool()]
        prompt_template = """You are a news analysis expert.
        Focus on analyzing news impact on stock prices and market sentiment.
        
        Key areas to consider:
        1. Breaking news and announcements
        2. Company events and updates
        3. Industry trends
        4. Regulatory news
        5. Market reactions to news
        
        Tools available: {tools}
        
        Use the above tools to help you answer the following question:
        {input}
        
        {agent_scratchpad}
        """
        super().__init__(llm_model, tools, memory, prompt_template)

class SentimentAnalysisAgent(BaseAnalysisAgent):
    def __init__(self, llm_model: str, memory: ConversationBufferMemory = None):
        tools = [SentimentAnalysisTool()]
        prompt_template = """You are a market sentiment analysis expert.
        Focus on analyzing social media sentiment and market psychology.
        
        Key areas to consider:
        1. Social media trends
        2. Investor sentiment
        3. Market psychology
        4. Trading volume and participation
        5. Sentiment indicators
        
        Tools available: {tools}
        
        Use the above tools to help you answer the following question:
        {input}
        
        {agent_scratchpad}
        """
        super().__init__(llm_model, tools, memory, prompt_template)

class StrategyAgent(BaseAnalysisAgent):
    def __init__(self, llm_model: str, memory: ConversationBufferMemory = None):
        # Strategy agent has access to all tools
        tools = [
            FinancialDataTool(),
            TechnicalDataTool(),
            NewsSearchTool(),
            SentimentAnalysisTool()
        ]
        prompt_template = """You are a comprehensive investment strategy expert.
        Your role is to synthesize all available analysis and provide actionable recommendations.
        
        When making recommendations:
        1. Consider all aspects (fundamental, technical, news, sentiment)
        2. Weigh the importance of different factors
        3. Provide clear entry/exit points if applicable
        4. Assess risks and potential rewards
        5. Consider timeframe and investment horizon
        
        Tools available: {tools}
        
        Use the above tools to help you answer the following question:
        {input}
        
        {agent_scratchpad}
        """
        super().__init__(llm_model, tools, memory, prompt_template)
    
    async def generate_strategy(self, 
                              ticker: str,
                              fundamental_analysis: Dict[str, Any],
                              technical_analysis: Dict[str, Any],
                              news_analysis: Dict[str, Any],
                              sentiment_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive investment strategy based on all analyses.
        """
        query = f"""
        Analyze {ticker} and provide an investment strategy based on the following:
        
        Fundamental Analysis: {fundamental_analysis}
        Technical Analysis: {technical_analysis}
        News Analysis: {news_analysis}
        Sentiment Analysis: {sentiment_analysis}
        
        Provide a detailed strategy recommendation including:
        1. Overall recommendation (Buy/Sell/Hold)
        2. Entry/Exit points
        3. Risk factors
        4. Investment timeframe
        5. Confidence level
        """
        
        return await self.analyze(query) 