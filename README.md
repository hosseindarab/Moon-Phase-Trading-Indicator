# Moon Phase Trading Indicator 🌕📈

Visualize how lunar phases align with market prices. This Streamlit app plots Full/New Moons on a price chart, overlays portfolio value, backtests a simple signals strategy, and lists executed trades.

- Entrypoint: [moon_trading_app.py](moon_trading_app.py)
- Dependencies: [requirements.txt](requirements.txt)

## Features
- Lunar markers
  - Past and upcoming Full/New Moons
  - Future markers pinned at the latest price (▲ green Full Moon, ▼ red New Moon)
- Dual y-axes
  - Left: Price
  - Right: Portfolio Value (overlay)
- Trades table
  - Most recent first with dates normalized to day
- Interactive Plotly chart embedded in Streamlit
- Global results summary at the end of the app

## Quickstart

1) Create and activate a virtual environment (recommended)

- Windows (PowerShell)
  ```sh
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- Windows (CMD)
  ```bat
  python -m venv venv
  .\venv\Scripts\activate.bat
  ```
- macOS/Linux
  ```sh
  python -m venv venv
  source venv/bin/activate
  ```

2) Install dependencies
```sh
pip install -r requirements.txt
```

3) Run the app
```sh
streamlit run moon_trading_app.py
```

## How to use
- Pick a ticker and date range in the UI
- Review the price series with lunar markers (past + future)
- Compare left-axis Price vs right-axis Portfolio Value
- Review the Trades table (newest first)
- Check the global summary at the bottom for aggregate results

## Project layout
- App: moon_trading_app.py
- Deps: requirements.txt
- Repo meta: .gitignore, README.md

## Notes
- Requires internet access to fetch market data
- Future moon markers indicate timing only; they do not predict price
- If fonts or icons don’t display, ensure Plotly renders via Streamlit (disable ad/script blockers if needed)