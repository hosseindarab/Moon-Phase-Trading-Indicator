"""
moon_trading_app.py
Interactive Streamlit dashboard for Moon Phase trading:
- Buy on Full Moon, Sell on New Moon
- Grid-search optimize Stop-Loss and Take-Profit
- Multi-ticker backtesting, charts, metrics and CSV export
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ephem
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import requests

st.set_page_config(layout="wide", page_title="Moon Phase Trading Lab 🌙")

# -----------------------
# Helper: moon event calculator
# -----------------------
def moon_events(start_date_str, end_date_str):
    """Return lists of new moon & full moon datetimes between start and end (local naive datetimes)."""
    obs = ephem.Date(start_date_str)
    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
    new_moons = []
    full_moons = []
    # iterate by jumping approx one lunar cycle each loop
    while obs < ephem.Date(end_date_str):
        try:
            nm = ephem.localtime(ephem.next_new_moon(obs))
            fm = ephem.localtime(ephem.next_full_moon(obs))
        except Exception:
            break
        if nm < end_dt:
            new_moons.append(nm)
        if fm < end_dt:
            full_moons.append(fm)
        obs = ephem.Date(obs + 29)  # jump ~29 days forward
    return new_moons, full_moons

# -----------------------
# Helper: download & prepare data (cached)
# -----------------------
@st.cache_data(show_spinner=False)
def load_price_data(ticker, start, end):
    df = yf.download(
    ticker,
    start=start,
    end=end,
    progress=False,
    auto_adjust=False,  # explicit to suppress warning and keep raw Close
    )[['Close']].dropna()
    return df

# -----------------------
# Helper: fetch Fear and Greed Index data
# -----------------------
@st.cache_data(show_spinner=False, ttl=3600)  # cache for 1 hour
def fetch_fear_greed_data(limit=100):
    """Fetch Fear and Greed Index data from CoinMarketCap API."""
    url = "https://pro-api.coinmarketcap.com/v3/fear-and-greed/historical"
    headers = {
        'X-CMC_PRO_API_KEY': '8e705997255b4be89255f44602a22b9f',
        'Accept': 'application/json'
    }
    params = {
        'limit': limit
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status', {}).get('error_code') != 0:
            error_msg = data.get('status', {}).get('error_message', 'Unknown error')
            st.error(f"API Error: {error_msg}")
            return pd.DataFrame()
        
        # Parse the data
        records = data.get('data', [])
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch Fear and Greed Index data: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error processing Fear and Greed Index data: {str(e)}")
        return pd.DataFrame()

def nearest_trading_date(df, target_date):
    """Return the index of the row in df whose 'Date' is closest to target_date (by absolute days)."""
    diffs = (df["Date"].dt.normalize() - pd.Timestamp(target_date)).abs()
    return diffs.idxmin()


# -----------------------
# Backtest logic
# -----------------------
def run_backtest(df, new_moons, full_moons, initial_cash, stop_loss, take_profit):
    # copy to avoid mutation
    data = df.copy()
    # Convert index to pandas Series of dates
    dates = pd.Series(data.index.date, index=data.index)  # keep index aligned
    full_moon_dates = set(pd.Timestamp(d).date() for d in full_moons)
    new_moon_dates = set(pd.Timestamp(d).date() for d in new_moons)

    # Assign boolean columns
    data['FullMoon'] = dates.apply(lambda x: x in full_moon_dates)
    data['NewMoon'] = dates.apply(lambda x: x in new_moon_dates)



    cash = initial_cash
    position = 0.0
    entry_price = 0.0
    portfolio_values = []
    trades = []

    for i in range(len(data)):
        dt = data.index[i]
        price = float(data['Close'].iloc[i].item())
        is_full = bool(data['FullMoon'].iloc[i])
        is_new = bool(data['NewMoon'].iloc[i])

        # stop-loss / take-profit
        if position > 0:
            change = (price - entry_price) / entry_price
            if change <= -stop_loss:
                cash = position * price
                position = 0.0
                trades.append({'date': dt, 'type': 'STOP LOSS', 'price': price})
            elif change >= take_profit:
                cash = position * price
                position = 0.0
                trades.append({'date': dt, 'type': 'TAKE PROFIT', 'price': price})

        # buy/sell
        if is_full and position == 0:
            position = float(cash / price)
            entry_price = price
            cash = 0.0
            trades.append({'date': dt, 'type': 'BUY 🌕', 'price': price})
        elif is_new and position > 0:
            cash = position * price
            position = 0.0
            trades.append({'date': dt, 'type': 'SELL 🌑', 'price': price})

        portfolio_values.append(cash + position * price)


    data['Portfolio'] = portfolio_values

    final_value = portfolio_values[-1] if len(portfolio_values) else initial_cash
    total_return_pct = (final_value / initial_cash - 1) * 100

    # performance metrics
    daily_ret = pd.Series(data['Portfolio']).pct_change().dropna()
    if len(daily_ret) > 1 and daily_ret.std() != 0:
        sharpe = np.sqrt(252) * (daily_ret.mean() / daily_ret.std())
    else:
        sharpe = 0.0
    cummax = data['Portfolio'].cummax()
    drawdown = (cummax - data['Portfolio']) / cummax
    max_dd = drawdown.max() * 100 if not drawdown.empty else 0.0

    buys = sum(1 for t in trades if t['type'].startswith('BUY'))
    sells = sum(1 for t in trades if t['type'].startswith('SELL') or t['type'] == 'TAKE PROFIT')
    win_rate = (sells / buys * 100) if buys > 0 else 0.0

    result = {
        'final_value': final_value,
        'return_pct': total_return_pct,
        'sharpe': sharpe,
        'max_drawdown_pct': max_dd,
        'win_rate_pct': win_rate,
        'trades': trades,
        'equity_curve': data[['Portfolio', 'Close']].copy()
    }
    return result

# -----------------------
# UI: sidebar controls
# -----------------------
st.sidebar.title("Moon Strategy Controls")
st.sidebar.markdown("Select ticker(s), date range, risk params, and which data to analyze.")

default_tickers = "BTC-USD, ETH-USD"
tickers_input = st.sidebar.text_input("Tickers (comma separated)", value=default_tickers)
tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

start_date = st.sidebar.date_input("Start date", value=datetime(2023,1,1))
end_date = st.sidebar.date_input("End date", value=datetime.today())
initial_cash = st.sidebar.number_input("Initial cash (USD)", value=10000, min_value=1)
stop_loss = st.sidebar.slider("Stop-loss %", 0.0, 0.5, value=0.10, step=0.01)
take_profit = st.sidebar.slider("Take-profit %", 0.0, 1.0, value=0.30, step=0.01)
do_grid_search = st.sidebar.checkbox("Grid search optimize SL/TP", value=False)
sl_grid = st.sidebar.multiselect("SL grid (if grid enabled)", options=[0.02,0.05,0.10,0.15,0.20,0.25], default=[0.05,0.10,0.15])
tp_grid = st.sidebar.multiselect("TP grid (if grid enabled)", options=[0.10,0.20,0.30,0.40,0.50], default=[0.20,0.30,0.40])

st.sidebar.markdown("---")
st.sidebar.markdown("Notes: Buy on Full Moon (🌕), Sell on New Moon (🌑). Moon times computed with the `ephem` library.")

# -----------------------
# Main UI layout
# -----------------------
st.title("Moon Phase Trading Lab 🌙")
st.write("Interactive exploration of the strategy: **Buy on Full Moon**, **Sell on New Moon**. Use the sidebar to control parameters.")

# validate dates
if start_date >= end_date:
    st.error("Start date must be earlier than end date.")
    st.stop()

# -----------------------
# Fear and Greed Index Section
# -----------------------
st.header("📊 Crypto Fear and Greed Index")
st.write("Current market sentiment from CoinMarketCap. Lower values indicate fear, higher values indicate greed.")

with st.spinner("Fetching Fear and Greed Index data..."):
    fg_data = fetch_fear_greed_data(limit=100)

if not fg_data.empty:
    # Create Fear and Greed chart
    fig_fg = go.Figure()
    
    # Add line trace with color gradient based on value
    fig_fg.add_trace(go.Scatter(
        x=fg_data['timestamp'],
        y=fg_data['value'],
        mode='lines+markers',
        name='Fear & Greed Index',
        line=dict(color='rgba(75, 192, 192, 1)', width=2),
        marker=dict(size=6),
        hovertemplate='<b>Date</b>: %{x}<br>' +
                      '<b>Value</b>: %{y}<br>' +
                      '<b>Classification</b>: %{text}<br>' +
                      '<extra></extra>',
        text=fg_data['value_classification']
    ))
    
    # Add color zones
    fig_fg.add_hrect(y0=0, y1=25, fillcolor="red", opacity=0.1, line_width=0, annotation_text="Extreme Fear", annotation_position="left")
    fig_fg.add_hrect(y0=25, y1=45, fillcolor="orange", opacity=0.1, line_width=0, annotation_text="Fear", annotation_position="left")
    fig_fg.add_hrect(y0=45, y1=55, fillcolor="yellow", opacity=0.1, line_width=0, annotation_text="Neutral", annotation_position="left")
    fig_fg.add_hrect(y0=55, y1=75, fillcolor="lightgreen", opacity=0.1, line_width=0, annotation_text="Greed", annotation_position="left")
    fig_fg.add_hrect(y0=75, y1=100, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Extreme Greed", annotation_position="left")
    
    fig_fg.update_layout(
        title="Fear and Greed Index Over Time",
        xaxis_title="Date",
        yaxis_title="Index Value (0-100)",
        yaxis=dict(range=[0, 100]),
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig_fg, use_container_width=True, config={"displaylogo": False})
    
    # Show latest value with metrics
    if len(fg_data) > 0:
        latest = fg_data.iloc[-1]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Latest Value", f"{latest['value']:.0f}", help="Fear and Greed Index (0-100)")
        with col2:
            st.metric("Classification", latest['value_classification'])
        with col3:
            st.metric("Last Updated", latest['timestamp'].strftime('%Y-%m-%d'))
    
    # Option to show data table
    with st.expander("View Fear and Greed Historical Data"):
        display_df = fg_data.copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(display_df[['timestamp', 'value', 'value_classification']], use_container_width=True)
        
        # Download button for Fear and Greed data
        csv_fg = fg_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            "💾 Download Fear and Greed CSV",
            data=csv_fg,
            file_name="fear_greed_index.csv",
            mime="text/csv"
        )
else:
    st.warning("Unable to fetch Fear and Greed Index data. Please check your internet connection or try again later.")

st.markdown("---")

cols = st.columns([2,1])

with cols[0]:
    st.header("Backtest Results")

with cols[1]:
    st.header("Export")
    st.write("Export selected results as CSV.")

# -----------------------
# Run backtests per ticker
# -----------------------
all_results = []
for ticker in tickers:
    with st.expander(f"Run: {ticker}", expanded=True):
        st.write(f"Downloading {ticker} data from {start_date} to {end_date} ...")
        df = load_price_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        if df.empty:
            st.warning("No data returned for this ticker/date range.")
            continue

        # compute moon events
        new_moons, full_moons = moon_events(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        st.write(f"Found {len(full_moons)} Full Moons and {len(new_moons)} New Moons in range.")

        # single run or grid search
        if not do_grid_search:
            res = run_backtest(df, new_moons, full_moons, initial_cash, stop_loss, take_profit)
            all_results.append({'ticker': ticker, 'sl': stop_loss, 'tp': take_profit, **res})
        else:
            best = None
            best_ret = -np.inf
            grid_rows = []
            pbar = st.progress(0)
            combos = [(sl, tp) for sl in sl_grid for tp in tp_grid]
            for i, (sl, tp) in enumerate(combos):
                tmp = run_backtest(df, new_moons, full_moons, initial_cash, sl, tp)
                grid_rows.append({'ticker': ticker, 'sl': sl, 'tp': tp, 'return_pct': tmp['return_pct'], 'sharpe': tmp['sharpe'], 'max_drawdown_pct': tmp['max_drawdown_pct']})
                if tmp['return_pct'] > best_ret:
                    best_ret = tmp['return_pct']
                    best = {'ticker': ticker, 'sl': sl, 'tp': tp, **tmp}
                pbar.progress(int((i+1)/len(combos)*100))
            pbar.empty()
            all_results.append(best)

        # show a detailed view for this ticker's best result
        best_display = all_results[-1]
        st.metric("Final Portfolio Value (USD)", f"{best_display['final_value']:.2f}", delta=f"{best_display['return_pct']:.2f}%")
        st.write(f"Sharpe: {best_display['sharpe']:.3f} | Max Drawdown: {best_display['max_drawdown_pct']:.2f}% | Win Rate: {best_display['win_rate_pct']:.1f}%")
        st.write(f"Trades: {len(best_display['trades'])} events")

        # -----------------------
        # Equity & Price Chart with Past & Future Moon Markers
        # -----------------------

        equity_df = best_display['equity_curve'].reset_index().rename(columns={'index': 'Date'})
        fig = go.Figure()

        # plot historical close and portfolio
        fig.add_trace(go.Scatter(
            x=equity_df['Date'],
            y=equity_df['Close'],
            name=f"{ticker} Close",
            line=dict(color='gray'),
            yaxis="y1"
        ))

        fig.add_trace(go.Scatter(
            x=equity_df['Date'],
            y=equity_df['Portfolio'],
            name="Portfolio Value",
            line=dict(color='blue'),
            yaxis="y2"
        ))

        # prepare moon dates
        fm_dates = [pd.Timestamp(d).date() for d in full_moons]
        nm_dates = [pd.Timestamp(d).date() for d in new_moons]
        today = pd.Timestamp.today().date()

        # split past vs future moons
        fm_past = [d for d in fm_dates if d <= today]
        fm_future = [d for d in fm_dates if d > today]
        nm_past = [d for d in nm_dates if d <= today]
        nm_future = [d for d in nm_dates if d > today]

        # past moon markers
        price_on_fm = equity_df[equity_df['Date'].dt.date.isin(fm_past)]
        price_on_nm = equity_df[equity_df['Date'].dt.date.isin(nm_past)]

        if not price_on_fm.empty:
            fig.add_trace(go.Scatter(
                x=price_on_fm['Date'],
                y=price_on_fm['Close'],
                mode='markers',
                marker_symbol='triangle-up',
                marker_size=10,
                marker_color='green',
                name='Full Moon (past buy)'
            ))

        if not price_on_nm.empty:
            fig.add_trace(go.Scatter(
                x=price_on_nm['Date'],
                y=price_on_nm['Close'],
                mode='markers',
                marker_symbol='triangle-down',
                marker_size=10,
                marker_color='red',
                name='New Moon (past sell)'
            ))

        # future moon markers (at last known price, semi-transparent)
        last_price = equity_df['Close'].iloc[-1] if not equity_df.empty else 1.0

        if fm_future:
            fig.add_trace(go.Scatter(
                x=fm_future,
                y=[last_price]*len(fm_future),
                mode='markers+text',
                marker_symbol='triangle-up',
                marker_size=12,
                marker_color='green',
                marker_opacity=0.4,
                text=['Full Moon']*len(fm_future),
                textposition='top center',
                name='Full Moon (future)'
            ))

        if nm_future:
            fig.add_trace(go.Scatter(
                x=nm_future,
                y=[last_price]*len(nm_future),
                mode='markers+text',
                marker_symbol='triangle-down',
                marker_size=12,
                marker_color='red',
                marker_opacity=0.4,
                text=['New Moon']*len(nm_future),
                textposition='bottom center',
                name='New Moon (future)'
            ))

        # layout with two y-axes
        fig.update_layout(
            xaxis=dict(domain=[0, 0.99], title="Date"),
            yaxis=dict(title="Price", side='left'),
            yaxis2=dict(title="Portfolio Value", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # Do not render the base fig; we will render the dynamic one below

        # -----------------------
        # Interactive Moon Selection Table + Dynamic Highlight
        # -----------------------
        
        

        st.subheader("🔍 Explore Moon Events (Interactive)")
        moon_table = equity_df[['Date', 'Close']].copy()
        moon_table['FullMoon'] = moon_table['Date'].dt.date.isin(fm_dates)
        moon_table['NewMoon'] = moon_table['Date'].dt.date.isin(nm_dates)
        # make sure Streamlit treats them as boolean checkboxes
        moon_table = moon_table.astype({'FullMoon': 'bool', 'NewMoon': 'bool'})

        selected_table = st.data_editor(
            moon_table,
            key=f"moon_table_{ticker}",
            hide_index=True,
            column_config={
                "FullMoon": st.column_config.CheckboxColumn(
                    "FullMoon", help="Select to highlight", default=False
                ),
                "NewMoon": st.column_config.CheckboxColumn(
                    "NewMoon", help="Select to highlight", default=False
                ),
            },
            disabled=False,
            width="stretch",  # use_container_width deprecated
        )

        # Extract selections safely
        if 'FullMoon' in selected_table.columns:
            selected_fm = selected_table.loc[selected_table['FullMoon'] == True, 'Date'].dt.date.tolist()
        else:
            selected_fm = []

        if 'NewMoon' in selected_table.columns:
            selected_nm = selected_table.loc[selected_table['NewMoon'] == True, 'Date'].dt.date.tolist()
        else:
            selected_nm = []


        # Build highlight figure from the base chart
        highlight_fig = go.Figure(fig)
        
        # Highlight selected Full Moons
        if selected_fm:
            highlight_points = []
            for d in selected_fm:
                idx = nearest_trading_date(equity_df, d)
                highlight_points.append((equity_df.loc[idx, "Date"], equity_df.loc[idx, "Close"]))

            if highlight_points:
                highlight_fig.add_trace(go.Scatter(
                    x=[x for x, _ in highlight_points],
                    y=[y for _, y in highlight_points],
                    mode="markers",
                    marker_symbol="triangle-up",
                    marker_color="lime",
                    marker_size=14,
                    name="Selected Full Moon",
                    yaxis="y1",
                ))

        # Highlight selected New Moons
        if selected_nm:
            highlight_points = []
            for d in selected_nm:
                idx = nearest_trading_date(equity_df, d)
                highlight_points.append((equity_df.loc[idx, "Date"], equity_df.loc[idx, "Close"]))

            if highlight_points:
                highlight_fig.add_trace(go.Scatter(
                    x=[x for x, _ in highlight_points],
                    y=[y for _, y in highlight_points],
                    mode="markers",
                    marker_symbol="triangle-down",
                    marker_color="orange",
                    marker_size=14,
                    name="Selected New Moon",
                    yaxis="y1",
                ))


        # Render only the dynamic figure
        st.plotly_chart(highlight_fig, use_container_width=True, config={"displaylogo": False})

        st.download_button(
            "💾 Download selected moon table CSV",
            data=selected_table.to_csv(index=False),
            file_name=f"{ticker}_moon_selections.csv",
            mime="text/csv"
        )


        # trades table
        trades_df = pd.DataFrame(best_display['trades'])
        if not trades_df.empty:
            trades_df['date'] = pd.to_datetime(trades_df['date']).dt.date
            st.write("Trades (most recent first)")
            st.dataframe(trades_df.sort_values('date', ascending=False).reset_index(drop=True))

# -----------------------
# Global results & export
# -----------------------
if all_results:
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            'Ticker': r.get('ticker'),
            'SL': r.get('sl'),
            'TP': r.get('tp'),
            'Return %': r.get('return_pct'),
            'Final Value': r.get('final_value'),
            'Sharpe': r.get('sharpe'),
            'Max Drawdown %': r.get('max_drawdown_pct'),
            'Win Rate %': r.get('win_rate_pct'),
            'Trades Count': len(r.get('trades', []))
        })
    summary_df = pd.DataFrame(summary_rows)
    st.header("Summary Across Tickers")
    st.dataframe(summary_df)

    # CSV export
    csv = summary_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download summary CSV", data=csv, file_name="moon_strategy_summary.csv", mime="text/csv")
