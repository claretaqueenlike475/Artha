"""
test_tools.py
Run with: python test_tools.py

Verbose manual sanity-check suite. 
Prints inputs, outputs, and JSON-safety checks.
"""

import json
import os
import textwrap

# --- FORMATTING HELPERS ---

W = 80

def _line(char="-"):
    print(char * W)

def _header(title: str):
    print("\n" + "=" * W)
    print(f" {title} ".center(W, "="))
    _line()

def _section(title: str):
    print(f"\n [ {title} ]")

def _kv(key: str, value, indent: int = 2):
    prefix = " " * indent
    val_str = str(value)
    if len(val_str) > W - indent - len(key) - 4:
        wrapped = textwrap.fill(
            val_str,
            width=W - indent - 2,
            initial_indent=prefix + f"{key}: ",
            subsequent_indent=prefix + " " * (len(key) + 2),
        )
        print(wrapped)
    else:
        print(f"{prefix}{key}: {value}")

def _preview(label: str, text: str, chars: int = 220, indent: int = 2):
    prefix = " " * indent
    clean = text.replace("\n", " ").strip()
    snippet = clean[:chars] + ("..." if len(clean) > chars else "")
    wrapped = textwrap.fill(
        snippet,
        width=W - indent,
        initial_indent=prefix + f"{label}: ",
        subsequent_indent=prefix + " " * (len(label) + 2),
    )
    print(wrapped)

def _pass(label: str = ""):
    print(f"  [PASS] {label}")

def _fail(label: str, reason: str):
    print(f"  [FAIL] {label}: {reason}")

def _skip(reason: str):
    print(f"  [SKIP] {reason}")

def assert_json_safe(obj, label: str) -> bool:
    try:
        json.dumps(obj)
        print(f"  [PASS] JSON-safe: {label}")
        return True
    except (TypeError, ValueError) as e:
        print(f"  [FAIL] JSON-safe ({label}) - {e}")
        return False

# --- 1. SESSION STORE ---

def test_session_store():
    _header("TEST 1: Session Store")
    from utils.session_store import append_message, get_history, add_file, get_files, clear_session

    sid = "test_session_001"
    append_message(sid, "user", "What is the stock price?")
    append_message(sid, "assistant", "Let me check that for you.")
    
    hist = get_history(sid)
    _section("History Output")
    for m in hist:
        _kv(f"Role: {m['role']}", m["content"])
    
    assert len(hist) == 2
    _pass("Session Store working")

# --- 2. FORMATTERS ---

def test_formatters():
    _header("TEST 2: Formatters (Data Sanitization)")
    import pandas as pd
    import numpy as np
    from utils.formatters import sanitize_dataframe, sanitize_info_dict

    # Realistic yfinance OHLCV DataFrame for TCS.NS
    _section("Input DataFrame (simulated yfinance OHLCV for TCS.NS)")
    df = pd.DataFrame({
        "Date": [
            pd.Timestamp("2025-03-10 00:00:00+0530"),
            pd.Timestamp("2025-03-11 00:00:00+0530"),
            pd.Timestamp("2025-03-12 00:00:00+0530"),
            pd.Timestamp("2025-03-13 00:00:00+0530"),
            pd.Timestamp("2025-03-14 00:00:00+0530"),
        ],
        "Open":   [np.float64(3812.5), np.float64(3798.0), np.float64(3820.0), np.float64(3756.0), np.float64(3780.0)],
        "High":   [np.float64(3845.0), np.float64(3831.5), np.float64(3835.0), np.float64(3795.0), np.float64(3810.0)],
        "Low":    [np.float64(3790.0), np.float64(3780.0), np.float64(3798.0), np.float64(3740.0), np.float64(3762.0)],
        "Close":  [np.float64(3801.25), np.float64(float("nan")), np.float64(3822.75), np.float64(3758.50), np.float64(3795.00)],
        "Volume": [np.int64(2_341_200), np.int64(1_987_500), np.int64(2_104_800), np.int64(3_215_600), np.int64(1_876_300)],
    })
    df = df.set_index("Date")
    print("  " + str(df).replace("\n", "\n  "))

    _section("Output Dictionary (JSON-safe, ready for frontend chart)")
    result = sanitize_dataframe(df)
    for col, vals in result.items():
        _kv(col, vals)

    assert result["Close"][1] is None, "NaN close should become None"
    assert isinstance(result["Volume"][0], int), "numpy int64 should become plain int"
    assert isinstance(result["Close"][0], float), "numpy float64 should become plain float"
    assert_json_safe(result, "sanitize_dataframe")

    # Realistic yfinance .info dict for TCS — with typical NaN fields yfinance produces
    _section("Input Info Dict (simulated TCS.NS ticker.info)")
    info = {
        "longName": "Tata Consultancy Services Limited",
        "shortName": "TCS.NS",
        "currentPrice": np.float64(3795.00),
        "previousClose": np.float64(3758.50),
        "open": np.float64(3780.00),
        "dayHigh": np.float64(3812.40),
        "dayLow": np.float64(3762.10),
        "volume": np.int64(1_876_300),
        "marketCap": np.int64(1_378_542_329_856),
        "financialCurrency": "INR",
        "exchange": "NSE",
        "fiftyTwoWeekHigh": np.float64(4592.25),
        "fiftyTwoWeekLow": np.float64(3056.05),
        "trailingPE": np.float64(28.43),
        "forwardPE": np.float64(float("nan")),      # yfinance frequently returns NaN
        "priceToBook": np.float64(13.21),
        "dividendYield": np.float64(0.0172),
        "targetMeanPrice": np.float64(4280.0),
        "recommendationKey": "buy",
        "profitMargins": np.float64(0.1872),
        "totalRevenue": np.int64(2_408_270_004_224),
        "debtToEquity": np.float64(float("nan")),   # TCS near-zero debt, often NaN
        "returnOnEquity": np.float64(0.5231),
        "irrelevantKey": "should be filtered out by whitelist",
    }
    _kv("Total keys in raw info", len(info))

    _section("Output Dict (whitelist-filtered, all numpy types converted)")
    result2 = sanitize_info_dict(info)
    for k, v in result2.items():
        if v is not None:
            _kv(k, v)
    _kv("Keys returned as None (NaN or missing from input)", [k for k, v in result2.items() if v is None])

    assert "irrelevantKey" not in result2,        "non-whitelisted key must be excluded"
    assert result2["forwardPE"] is None,           "NaN forwardPE should become None"
    assert result2["debtToEquity"] is None,        "NaN debtToEquity should become None"
    assert result2["currentPrice"] == 3795.0,      "float64 should become plain float"
    assert result2["marketCap"] == 1_378_542_329_856, "int64 should become plain int"
    assert result2["recommendationKey"] == "buy",  "string should pass through unchanged"
    assert_json_safe(result2, "sanitize_info_dict")

# --- 3. STOCK INFO ---

def test_stock_info():
    _header("TEST 3: Stock Info (TCS.NS)")
    from tools.stock_data import get_stock_info

    result = get_stock_info("TCS", "NSE")
    if "error" in result:
        _fail("get_stock_info", result["error"])
        return

    _section("Stock Output")
    keys_to_show = ["longName", "currentPrice", "trailingPE", "marketCap"]
    for k in keys_to_show:
        if k in result:
            _kv(k, result[k])
            
    assert_json_safe(result, "get_stock_info")

# --- 4. STOCK HISTORY ---

def test_stock_history():
    _header("TEST 4: Stock History (WIPRO 1mo)")
    from tools.stock_data import get_stock_history

    result = get_stock_history("WIPRO", "NSE", "1mo", "1d")
    if "error" in result:
        _fail("get_stock_history", result["error"])
        return

    dates = result.get("dates", [])
    closes = result.get("close", [])
    
    _section("History Output")
    _kv("Total Days", len(dates))
    if dates:
        _kv("First Day", f"{dates[0]} | Close: {closes[0]}")
        _kv("Last Day", f"{dates[-1]} | Close: {closes[-1]}")
        
    assert_json_safe(result, "get_stock_history")

# --- 5. WEB SEARCH ---

def test_web_search():
    _header("TEST 5: Web Search")
    from tools.web_search import search_web

    query = "TCS results 2025"
    _section(f"Input Query: '{query}'")
    
    result = search_web(query, max_results=2)
    if not result or (len(result) == 1 and "error" in result[0]):
        _fail("search_web", result[0].get("error", "empty response"))
        return

    _section("Search Outputs")
    for i, r in enumerate(result, 1):
        _kv(f"Result {i} Title", r.get("title", "N/A"))
        _preview("Snippet", r.get("content", "N/A"), chars=100)
        
    assert_json_safe(result, "search_web")

# --- 6. NEWS SEARCH ---

def test_news_search():
    _header("TEST 6: News Search")
    from tools.news_search import search_news

    query = "Infosys"
    _section(f"Input Query: '{query}'")
    
    result = search_news(query, days_back=7)
    if not result or (len(result) == 1 and "error" in result[0]):
        _fail("search_news", result[0].get("error", "empty response"))
        return

    _section("News Outputs")
    for i, a in enumerate(result[:2], 1):
        _kv(f"Article {i} Title", a.get("title", "N/A"))
        _kv("Source", a.get("source", "N/A"))
        
    assert_json_safe(result, "search_news")

# --- 7. TICKER LOOKUP ---

def test_ticker_lookup():
    _header("TEST 7: Ticker Lookup")
    from tools.ticker_lookup import search_ticker

    query = "tata steel"
    _section(f"Input Query: '{query}'")
    
    results = search_ticker(query)
    for r in results:
        _kv("Match", f"{r.get('company_name')} | NSE: {r.get('nse_symbol')}")

# --- 8. DOCUMENT PARSER ---

def test_document_parser():
    _header("TEST 8: Document Parser")
    from utils.doc_parser import parse_uploaded_file

    test_dir = "test_files"
    if not os.path.exists(test_dir):
        _skip(f"Directory '{test_dir}' not found.")
        return

    files = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
    for filename in files:
        filepath = os.path.join(test_dir, filename)
        _section(f"Parsing: {filename}")
        
        result = parse_uploaded_file(filepath)
        if result["type"] == "error":
            _fail(filename, result["content"])
            continue

        _kv("Type Detected", result["type"])
        
        content = result["content"]
        if isinstance(content, str):
            _preview("Extracted Text", content, chars=150)
        elif isinstance(content, dict):
            sheets = list(content.keys())
            _kv("Sheets/Tables Found", sheets)
            if sheets and content[sheets[0]]:
                _kv("Row 1 Preview", content[sheets[0]][0])
                
        assert_json_safe(result, f"parse: {filename}")

# --- 9. RAG ENGINE ---

def test_rag_search():
    _header("TEST 9: RAG Engine")
    from utils.doc_parser import parse_uploaded_file
    from utils.rag_engine import index_document
    from tools.document_search import search_uploaded_documents

    test_dir = "test_files"
    if not os.path.exists(test_dir) or not os.listdir(test_dir):
        _skip("No test files found for RAG.")
        return

    _section("Indexing Documents")
    for filename in os.listdir(test_dir):
        filepath = os.path.join(test_dir, filename)
        if os.path.isfile(filepath):
            parsed = parse_uploaded_file(filepath)
            if parsed["type"] != "error":
                index_document(filename, parsed)
                print(f"  Indexed: {filename}")

    query = "What activation function is used in CNNs?"
    _section(f"Querying DB: '{query}'")
    
    res = search_uploaded_documents(query)
    if res.get("status") == "success":
        results = res.get("results", [])
        for i, chunk in enumerate(results[:2], 1):
            _preview(f"Match {i}", chunk, chars=200)
    else:
        _fail("Query Failed", res.get("message", "unknown error"))
        
    assert_json_safe(res, "search_uploaded_documents")

# --- 10. FORECASTING ---

def test_forecasting():
    _header("TEST 10: Chronos T5 Forecasting")
    from tools.ts_model import predict_stock_prices

    symbol = "WIPRO"
    horizon = 10
    _section(f"Input: {symbol} for {horizon} days")
    
    result = predict_stock_prices(symbol, "NSE", horizon_days=horizon)

    if "error" in result:
        _fail("predict_stock_prices", result["error"])
        return

    hist_closes = result.get("historical_closes", [])
    med = result.get("forecast_median", [])
    
    _section("Forecasting Output")
    if hist_closes:
        _kv("Last Historical Close", hist_closes[-1])
    
    _kv(f"Predicted Median path ({horizon} days)", med)
    _kv("Predicted Low path", result.get("forecast_low", []))
    _kv("Predicted High path", result.get("forecast_high", []))

    assert_json_safe(result, "predict_stock_prices")

# --- RUNNER ---

if __name__ == "__main__":
    _header("Shree - Tool Test Suite")

    test_functions = [
        test_session_store,
        test_formatters,
        test_stock_info,
        test_stock_history,
        test_web_search,
        test_news_search,
        test_ticker_lookup,
        test_document_parser,
        test_rag_search,
        test_forecasting,
    ]

    passed, failed = 0, 0
    for test_fn in test_functions:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"\n  [FAIL] {test_fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"\n  [ERROR] {test_fn.__name__}: {type(e).__name__}: {e}")
            failed += 1

    _line("=")
    print(f"Results: {passed} tests completed | {failed} failed")
    _line("=")
