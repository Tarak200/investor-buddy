# Financial Research Copilot

A multi-agent research assistant that produces deep-dive equity analysis reports for **US** (NYSE / NASDAQ) and **Indian** (NSE / BSE) stocks.

---

## Prerequisites

- Python 3.12+
- Docker & Docker Compose (for the vector store)
- API keys (see [Configuration](#configuration))

---

## Installation

```bash
# Clone and enter the project
git clone <repo-url>
cd financial_assistant

# Create a virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -e .
```

---

## Configuration

Copy the example environment file and fill in your keys:

```bash
cp .env.example .env
```

Open `.env` and set at minimum one LLM key and one search key:

| Variable | Where to get it |
|---|---|
| `GROQ_API_KEY` | https://console.groq.com |
| `OPENROUTER_API_KEY` | https://openrouter.ai |
| `GOOGLE_API_KEY` | https://aistudio.google.com |
| `NEWSAPI_KEY` | https://newsapi.org |
| `TAVILY_API_KEY` | https://app.tavily.com |
| `REDDIT_CLIENT_ID` | https://www.reddit.com/prefs/apps |
| `REDDIT_CLIENT_SECRET` | https://www.reddit.com/prefs/apps |

---

## Starting the Services

### Option A — Docker Compose (recommended)

Starts Weaviate, the API, and the frontend together:

```bash
docker compose -f infrastructure/docker-compose.yml up --build
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| REST API | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |

### Option B — Run locally

Start Weaviate first (Docker required for the vector store):

```bash
docker compose -f infrastructure/docker-compose.yml up weaviate -d
```

Then in separate terminals:

```bash
# Terminal 1 — API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Frontend
streamlit run frontend/app.py
```

---

## Using the Web UI

Open **http://localhost:8501** in your browser.

### Analyse a Company

1. In the **left sidebar**, enter:
   - **Company Name** — e.g. `Reliance Industries` or `Apple Inc`
   - **Ticker Symbol** — e.g. `RELIANCE` (India) or `AAPL` (US)
   - **Market** — `INDIA` or `US`
   - **Sector** — e.g. `Energy`, `Technology` (optional, improves peer comparison)
   - **Time Horizon** — projection window: 1 Y, 2 Y, 3 Y, 5 Y, 7 Y, or 10 Y
2. Click **Run Analysis**.
3. The UI polls for progress and displays the full report across 15 tabs once complete:

| Tab | Content |
|---|---|
| Overview | Executive summary, investment verdict |
| Projections | Bull / Base / Bear price targets by year |
| Valuation & Technical | PE, PB, DCF, RSI, moving averages |
| Financials | P&L, balance sheet, cash flow, EPS trend |
| Ownership | Promoter holding, institutional & FII/FPI stake |
| Peers | Comparative metrics against sector peers |
| Ratings | Analyst ratings and credit ratings |
| News & Sentiment | Recent news with LLM sentiment scores |
| Legal | Regulatory actions, litigation, SEBI/SEC filings |
| Orders & Tenders | Order book, government tenders, contract wins |
| Products | Product/service portfolio and pipeline |
| Management | Key executives, track record, insider activity |
| Culture | Employee reviews (Glassdoor / AmbitionBox) |
| Innovation & Global | R&D spend, patents, international presence |
| Sources | Every data point linked back to its original source |

### Discover Interesting Stocks

1. In the sidebar under **Discover Interesting Stocks**, choose:
   - **Market** — `INDIA` or `US`
   - **Sector** — filter to a specific sector, or leave as `All Sectors`
   - **Market Cap** — one or more of Large Cap, Mid Cap, Small Cap
2. Click **Discover**.
3. The tool returns a ranked shortlist of stocks with composite scores and rationale.

---

## Using the REST API

The API is fully documented at **http://localhost:8000/docs**.

### Submit an analysis job

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "company": "Infosys",
    "ticker": "INFY",
    "market": "INDIA",
    "sector": "IT & Technology",
    "time_horizon_years": 3
  }'
```

Response:
```json
{ "job_id": "abc123", "status": "queued", "message": "Analysis job queued" }
```

### Poll job status

```bash
curl http://localhost:8000/api/v1/status/abc123
```

### Retrieve the completed report

```bash
curl http://localhost:8000/api/v1/report/abc123
```

### Discover stocks via API

```bash
curl -X POST http://localhost:8000/api/v1/discover \
  -H "Content-Type: application/json" \
  -d '{ "market": "INDIA", "sector": "Pharmaceuticals", "market_caps": ["Mid Cap"] }'
```

Poll the result:
```bash
curl http://localhost:8000/api/v1/discover/abc123
```

---

## Supported Tickers

- **India:** NSE/BSE symbols without suffix — e.g. `RELIANCE`, `TCS`, `HDFCBANK`
- **US:** Standard symbols — e.g. `AAPL`, `MSFT`, `TSLA`
