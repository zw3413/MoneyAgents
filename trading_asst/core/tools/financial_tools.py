from typing import Dict, Any, List, Type
import yfinance as yf
import pandas as pd
from langchain_community.tools import BaseTool
from pydantic import BaseModel, Field

class StockData(BaseModel):
    ticker: str = Field(description="Stock ticker symbol")
    metric: str = Field(description="Financial metric to retrieve (income_statement, balance_sheet, cash_flow, info)")

class TechnicalData(BaseModel):
    ticker: str = Field(description="Stock ticker symbol")
    start_date: str = Field(description="Start date for analysis (YYYY-MM-DD)")
    end_date: str = Field(description="End date for analysis (YYYY-MM-DD)")

class FinancialDataTool(BaseTool):
    name: str = "get_financial_data"
    description: str = "Get financial data for a given stock ticker"
    args_schema: Type[BaseModel] = StockData

    def _run(self, input: Dict[str, Any]) -> Dict[str, Any]:
        ticker = input["ticker"]
        metric = input["metric"]
        try:
            stock = yf.Ticker(ticker)
            
            if metric == "income_statement":
                data = stock.income_stmt
            elif metric == "balance_sheet":
                data = stock.balance_sheet
            elif metric == "cash_flow":
                data = stock.cashflow
            elif metric == "info":
                data = stock.info
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            return {
                "ticker": ticker,
                "metric": metric,
                "data": data.to_dict() if isinstance(data, pd.DataFrame) else data
            }
        except Exception as e:
            return {"error": str(e)}

    async def _arun(self, input: Dict[str, Any]):
        raise NotImplementedError("Async version not implemented")

class TechnicalDataTool(BaseTool):
    name: str = "get_technical_data"
    description: str = "Get technical indicators for a given stock ticker"
    args_schema: Type[BaseModel] = TechnicalData

    def _run(self, input: Dict[str, Any]) -> Dict[str, Any]:
        ticker = input["ticker"]
        start_date = input["start_date"]
        end_date = input["end_date"]
        try:
            # Download historical data
            data = yf.download(ticker, start=start_date, end=end_date)
            
            # Calculate technical indicators
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['RSI'] = self._calculate_rsi(data['Close'])
            data['MACD'], data['Signal'] = self._calculate_macd(data['Close'])
            
            return {
                "ticker": ticker,
                "period": f"{start_date} to {end_date}",
                "data": data.to_dict(orient='index')
            }
        except Exception as e:
            return {"error": str(e)}

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series, 
                       fast_period: int = 12, 
                       slow_period: int = 26, 
                       signal_period: int = 9) -> tuple:
        exp1 = prices.ewm(span=fast_period, adjust=False).mean()
        exp2 = prices.ewm(span=slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        return macd, signal

    async def _arun(self, input: Dict[str, Any]):
        raise NotImplementedError("Async version not implemented") 