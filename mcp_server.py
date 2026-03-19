"""
mcp_server.py — Artha MCP Tool Server (FastMCP)

Exposes all financial tools over the MCP stdio protocol.
Can be connected to by any MCP-compatible client.

Design: the real functions in tools/ stay plain Python.
This file creates thin wrapper functions with descriptive docstrings
and registers them with FastMCP — same pattern as agent.py's @tool wrappers.
This keeps tools/ free of any framework dependency.

Run: python mcp_server.py
"""

from mcp.server.fastmcp import FastMCP

from tools.stock_data import (
    get_stock_info,
    get_stock_history,
    get_financials,
    get_corporate_actions,
    get_analyst_data,
    get_holders,
    get_esg_data,
    get_upcoming_events,
)
from tools.web_search import search_web
from tools.news_search import search_news
from tools.ticker_lookup import search_ticker
from tools.ts_model import predict_stock_prices
from utils.doc_parser import parse_uploaded_file
from utils.rag_engine import query_documents


mcp = FastMCP("artha-tools")


# ─────────────────────────────────────────────────────────────────────────────
# STOCK DATA TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_stock_info_tool(symbol: str, exchange: str = "NSE") -> dict:
    """Get real-time price, 52-week range, PE ratio, margins, debt ratios, and analyst targets.
    Call when the user asks about a stock's current state, price, or basic fundamentals.
    symbol: NSE/BSE ticker WITHOUT suffix. Examples: TCS, WIPRO, INFY, HDFCBANK, SBIN.
    exchange: NSE (default) or BSE."""
    return get_stock_info(symbol, exchange)


@mcp.tool()
def get_stock_history_tool(symbol: str, exchange: str = "NSE", period: str = "1mo", interval: str = "1d") -> dict:
    """Get OHLCV historical price data for candlestick or line charts.
    Call when the user asks for price charts, trend analysis, or historical performance.
    period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y.
    interval: 1m (last 7d only), 1h, 1d, 1wk."""
    return get_stock_history(symbol, exchange, period, interval)


@mcp.tool()
def get_financials_tool(symbol: str, exchange: str = "NSE", statement: str = "income", quarterly: bool = False) -> dict:
    """Get financial statements: income statement, balance sheet, or cash flow.
    statement: 'income', 'balance_sheet', or 'cashflow'.
    quarterly: True for last 4 quarters, False for last 4 annual periods."""
    return get_financials(symbol, exchange, statement, quarterly)


@mcp.tool()
def get_corporate_actions_tool(symbol: str, exchange: str = "NSE") -> dict:
    """Get dividend history and stock split history.
    Call when the user asks about dividends, shareholder returns, or splits."""
    return get_corporate_actions(symbol, exchange)


@mcp.tool()
def get_analyst_data_tool(symbol: str, exchange: str = "NSE") -> dict:
    """Get analyst consensus: price targets and buy/hold/sell recommendation counts.
    Call when the user asks what analysts think about a stock."""
    return get_analyst_data(symbol, exchange)


@mcp.tool()
def get_holders_tool(symbol: str, exchange: str = "NSE") -> dict:
    """Get top institutional and mutual fund shareholders.
    Call when the user asks about ownership structure or institutional interest."""
    return get_holders(symbol, exchange)


@mcp.tool()
def get_esg_data_tool(symbol: str, exchange: str = "NSE") -> dict:
    """Get ESG risk scores from Sustainalytics.
    Call when the user asks about sustainability or ethical investing.
    Only available for large-cap stocks."""
    return get_esg_data(symbol, exchange)


@mcp.tool()
def get_upcoming_events_tool(symbol: str, exchange: str = "NSE") -> dict:
    """Get upcoming earnings dates and ex-dividend dates.
    Call when the user asks when the next earnings report or dividend is."""
    return get_upcoming_events(symbol, exchange)


# ─────────────────────────────────────────────────────────────────────────────
# WEB AND NEWS TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def search_web_tool(query: str, max_results: int = 5) -> dict:
    """Search the internet for current information using Tavily.
    Use for macro events, regulatory changes, or anything needing live data.
    Do NOT use for stock prices — use get_stock_info_tool instead."""
    return {"results": search_web(query, max_results)}


@mcp.tool()
def search_news_tool(query: str, days_back: int = 7) -> dict:
    """Search recent news articles with source, date, and description metadata.
    Prefer over search_web_tool when the user asks specifically about news."""
    return {"results": search_news(query, days_back)}


# ─────────────────────────────────────────────────────────────────────────────
# TICKER LOOKUP
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def search_ticker_tool(query: str) -> dict:
    """Find NSE/BSE ticker symbol for an Indian company by name or partial name.
    Call FIRST when the user gives a company name instead of a ticker symbol.
    Example: 'HDFC Bank' -> returns 'HDFCBANK'."""
    return {"results": search_ticker(query)}


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT TOOLS
# These take filepath/filepaths directly since MCP server runs as a subprocess
# and cannot access the parent process's session_store.
# The agent passes the filepath from the system note injected into each message.
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def parse_document_tool(filepath: str) -> dict:
    """Parse an uploaded document and return its full extractable content.
    Supports PDF, DOCX, Excel, CSV, TXT, PPT.
    Call for broad questions like 'summarise this file'.
    filepath: absolute path to the file on disk (from the system note in the message)."""
    if not filepath:
        return {"error": "No filepath provided."}
    return parse_uploaded_file(filepath)


@mcp.tool()
def search_documents_tool(filepaths: str, query: str, top_k: int = 5) -> dict:
    """Semantically search across uploaded documents for a specific answer.
    Prefer over parse_document_tool for specific questions.
    filepaths: comma-separated absolute paths (from the system note in the message).
    query: natural-language question to search for.
    top_k: number of most relevant passages to return."""
    if not filepaths:
        return {"error": "No filepaths provided."}
    path_list = [p.strip() for p in filepaths.split(",") if p.strip()]
    return query_documents(path_list, query, top_k)


# ─────────────────────────────────────────────────────────────────────────────
# FORECAST TOOL
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def predict_stock_tool(symbol: str, exchange: str = "NSE", horizon_days: int = 10) -> dict:
    """Forecast next N closing prices using Amazon Chronos (zero-shot model).
    Call ONLY when the user explicitly asks for a forecast or prediction.
    Do NOT use for current prices — use get_stock_info_tool instead.
    horizon_days: trading days to forecast, recommended range 5 to 20."""
    return predict_stock_prices(symbol, exchange, horizon_days)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
