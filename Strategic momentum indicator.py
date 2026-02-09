import yfinance as yf
import pandas as pd
import numpy as np
import urllib.request
import os
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tabulate import tabulate


#                CONFIGURATION
# ============================================================

TIME_WINDOW = 12

USE_SMA_FILTER = False
SHOW_SMAS_ON_CHART = False

# === Revenue filter ===
USE_REVENUE_FILTER = True
MIN_REVENUE_CAGR = 0.14
MIN_YEARS_INCREASING = 0.65
MAX_REVENUE_YEARS = 4

DATA_PERIOD_YEARS = 10
DATA_INTERVAL = "1wk"

RESULTS_FOLDER = "results"


#                TICKER FETCHING FUNCTION
# ============================================================

def get_sp500_tickers(limit=500):
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        html = response.read()
    tables = pd.read_html(html)
    df = tables[0]
    tickers = [t.replace(".", "-") for t in df['Symbol'].tolist()]
    return tickers[:limit]


#                FUNDAMENTAL FILTER
# ============================================================

def passes_revenue_filter(ticker):
    try:
        t = yf.Ticker(ticker)
        df = t.financials.T
        if "Total Revenue" not in df.columns or df["Total Revenue"].dropna().empty:
            print(f"⚠️ {ticker}: No revenue data available.")
            return False, None

        revenues = df["Total Revenue"].dropna().tail(MAX_REVENUE_YEARS)
        if len(revenues) < 3:
            print(f"⚠️ {ticker}: Not enough revenue history ({len(revenues)} years).")
            return False, None

        revenues = revenues.sort_index()
        start_rev, end_rev = revenues.iloc[0], revenues.iloc[-1]
        years = len(revenues) - 1
        cagr = (end_rev / start_rev) ** (1 / years) - 1
        increases = sum(revenues.iloc[i] > revenues.iloc[i - 1] for i in range(1, len(revenues)))
        pct_increasing = increases / (len(revenues) - 1)

        passes = (cagr >= MIN_REVENUE_CAGR) and (pct_increasing >= MIN_YEARS_INCREASING)
        print(f"📊 {ticker}: CAGR={cagr:.2%}, ↑Years={pct_increasing:.0%} → {'✅ Pass' if passes else '❌ Fail'}")
        return passes, cagr

    except Exception as e:
        print(f"⚠️ {ticker}: Revenue filter error - {e}")
        return False, None


#                TECHNICAL ANALYSIS
# ============================================================

def get_stock_data(ticker, years=DATA_PERIOD_YEARS, interval=DATA_INTERVAL):
    data = yf.download(
        tickers=ticker,
        period=f"{years}y",
        interval=interval,
        multi_level_index=False,
        auto_adjust=False
    )
    data.dropna(inplace=True)
    return data

def calculate_sma(data, short_window=50, long_window=100):
    data['SMA50'] = data['Close'].rolling(window=short_window).mean()
    data['SMA100'] = data['Close'].rolling(window=long_window).mean()
    return data

def calculate_stochastic(data, k_length=14, d_smoothing=5, k_smoothing=9):
    data['LowestLow'] = data['Low'].rolling(window=k_length, min_periods=1).min()
    data['HighestHigh'] = data['High'].rolling(window=k_length, min_periods=1).max()
    k_raw = ((data['Close'] - data['LowestLow']) /
             (data['HighestHigh'] - data['LowestLow'])) * 100
    data['%K'] = k_raw.rolling(window=k_smoothing, min_periods=1).mean()
    data['%D'] = data['%K'].rolling(window=d_smoothing, min_periods=1).mean()
    return data

def generate_signals(data, oversold_k=25, oversold_d=25):
    crossover = (data['%K'] > data['%D']) & (data['%K'].shift(1) <= data['%D'].shift(1))
    oversold = (data['%K'] < oversold_k) | (data['%D'] < oversold_d)
    recent_oversold_flag = oversold.rolling(window=5, min_periods=1).max() > 0
    sma_filter = data['SMA100'] < data['SMA50'] if USE_SMA_FILTER else True
    data['Signal'] = crossover & recent_oversold_flag & sma_filter
    latest_cross = (
        len(data) > 1 and crossover.iloc[-1]
        and (oversold.iloc[-1] or recent_oversold_flag.iloc[-1])
        and (sma_filter.iloc[-1] if USE_SMA_FILTER else True)
    )
    data['New_Signal'] = latest_cross
    return data


#     HISTORICAL EDGE CHECK
# ============================================================

def check_historical_edge(data, time_window):
    signal_points = data[data['Signal']]
    num_signals = len(signal_points)
    if num_signals == 0:
        return False, None, None, None, None, num_signals

    gains, losses = [], []
    for signal_date in signal_points.index:
        entry_idx = data.index.get_loc(signal_date)
        entry_price = data.iloc[entry_idx]['Close']
        end_idx = min(entry_idx + time_window, len(data) - 1)
        future_prices = data.iloc[entry_idx + 1:end_idx + 1]['Close']
        if future_prices.empty:
            continue
        max_future_price, min_future_price = future_prices.max(), future_prices.min()
        gains.append((max_future_price / entry_price) - 1)
        losses.append((min_future_price / entry_price) - 1)

    if not gains:
        return False, None, None, None, None, num_signals

    avg_gain, median_gain = np.mean(gains), np.median(gains)
    avg_loss, median_loss = np.mean(losses), np.median(losses)
    return avg_gain, median_gain, avg_loss, median_loss, num_signals


#                VOLATILITY CALCULATION
# ============================================================

def calculate_annualized_volatility(data, window=20):
    interval_factor = {"1d": 252, "1wk": 52, "1mo": 12}
    T = interval_factor.get(DATA_INTERVAL, 52)
    data["LogReturn"] = np.log(data["Close"] / data["Close"].shift(1))
    data["Volatility"] = data["LogReturn"].rolling(window=window).std() * np.sqrt(T)
    return data


#                PLOTTING
# ============================================================

def plot_stochastic_chart(ticker, data):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.05, row_heights=[0.6, 0.25, 0.15],
        subplot_titles=[f"{ticker} Price", "Stochastic Oscillator", "Annualized Historical Volatility"]
    )

    fig.add_trace(go.Candlestick(
        x=data.index, open=data['Open'], high=data['High'],
        low=data['Low'], close=data['Close'], name='Price'
    ), row=1, col=1)

    if SHOW_SMAS_ON_CHART:
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], mode='lines',
                                 name='SMA50', line=dict(color='cyan')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA100'], mode='lines',
                                 name='SMA100', line=dict(color='magenta')), row=1, col=1)

    signal_points = data[data['Signal']]
    fig.add_trace(go.Scatter(
        x=signal_points.index, y=signal_points['Close'],
        mode='markers', marker=dict(symbol='triangle-up', size=12, color='green'),
        name='Buy Signal'
    ), row=1, col=1)

    for date in signal_points.index:
        fig.add_vline(x=date, line=dict(color='green', width=2), opacity=0.6, row=1, col=1)

    fig.add_trace(go.Scatter(x=data.index, y=data['%K'], mode='lines',
                             name='%K', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['%D'], mode='lines',
                             name='%D', line=dict(color='orange')), row=2, col=1)
    fig.add_hline(y=20, line_dash="dot", line_color="gray", row=2, col=1)
    fig.add_hline(y=80, line_dash="dot", line_color="gray", row=2, col=1)

    if "Volatility" in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data["Volatility"], mode="lines",
            name="Volatility", line=dict(color="yellow")
        ), row=3, col=1)

    fig.update_layout(
        title=f"Stochastic Strategy Chart - {ticker}",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Price", row=1, col=1, type="log")
    fig.update_yaxes(title_text="Stochastic", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="Volatility (Ann.)", row=3, col=1)
    fig.show()


#                MAIN SCRIPT
# ============================================================

if __name__ == "__main__":
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    # === Load existing results file ===
    load_choice = input("Do you want to load a saved results file? (y/n): ").strip().lower()
    if load_choice == "y":
        csv_files = [f for f in os.listdir(RESULTS_FOLDER) if f.endswith(".csv")]
        if not csv_files:
            print("⚠️ No saved files found.")
        else:
            print("\n📂 Saved results:")
            for i, file in enumerate(csv_files, start=1):
                print(f"{i}. {file}")
            choice = int(input("\nEnter number of file to load: "))
            if 1 <= choice <= len(csv_files):
                file_path = os.path.join(RESULTS_FOLDER, csv_files[choice - 1])
                df_loaded = pd.read_csv(file_path)

                # Ensure percentage column is correctly formatted
                if "Median Gain - Median Loss" in df_loaded.columns:
                    df_loaded["Median Gain - Median Loss"] = df_loaded["Median Gain - Median Loss"].apply(
                        lambda x: f"{float(x)*100:.2f}%" if str(x).replace('.', '', 1).isdigit() else x
                    )

                print(f"\n✅ Loaded file: {file_path}")
                print(tabulate(df_loaded, headers="keys", tablefmt="fancy_grid", showindex=False))
                exit()
            else:
                print("❌ Invalid choice. Exiting.")
                exit()

    # === Otherwise, run a new analysis ===
    mode = input("Choose mode: (1) Multiple stocks or (2) Single stock: ")
    if mode == "1":
        NUM_TICKERS = int(input("How many S&P 500 tickers to analyze (max 500)? "))
        tickers = get_sp500_tickers(limit=NUM_TICKERS)
    elif mode == "2":
        ticker_input = input("Enter ticker symbol: ").upper().strip()
        tickers = [ticker_input]
    else:
        print("❌ Invalid option. Exiting.")
        exit()

    results = []
    for ticker in tickers:
        try:
            cagr_value = None
            if USE_REVENUE_FILTER:
                passes, cagr_value = passes_revenue_filter(ticker)
                if not passes:
                    print(f"⏩ Skipping {ticker}: Failed revenue filter\n")
                    continue

            data = get_stock_data(ticker)
            if data.empty:
                print(f"⚠️ No data for {ticker}")
                continue

            if USE_SMA_FILTER or SHOW_SMAS_ON_CHART:
                calculate_sma(data)
            calculate_stochastic(data)
            data = generate_signals(data)
            calculate_annualized_volatility(data)

            avg_gain, median_gain, avg_loss, median_loss, num_signals = \
                check_historical_edge(data, time_window=TIME_WINDOW)

            fmt = lambda v: f"{v:.2%}" if isinstance(v, (float, int)) else "N/A"
            diff_gain_loss = (median_gain - abs(median_loss)) if (median_gain and median_loss) else None
            recent_cross_signal = bool(data['New_Signal'].iloc[-1]) if 'New_Signal' in data.columns else False

            results.append({
                "Ticker": ticker,
                "Signals": num_signals,
                "CAGR": fmt(cagr_value),
                "Avg Max Gain": fmt(avg_gain),
                "Median Max Gain": fmt(median_gain),
                "Avg Max Loss": fmt(avg_loss),
                "Median Max Loss": fmt(median_loss),
                "Median Gain - Median Loss": fmt(diff_gain_loss),
                "New Signal": "✅" if recent_cross_signal else "❌",
            })

            print(f"✅ {ticker}: Signals={num_signals}, Median Gain={fmt(median_gain)}, Loss={fmt(median_loss)}")

        except Exception as e:
            print(f"⚠️ Error analyzing {ticker}: {e}")

    # === Save results ===
    if results:
        df_results = pd.DataFrame(results)

        # Sort by numeric value of the difference column
        df_results["SortValue"] = df_results["Median Gain - Median Loss"].str.rstrip("%").astype(float)
        df_results = df_results.sort_values(by="SortValue", ascending=False).drop(columns="SortValue").reset_index(drop=True)

        df_results.insert(0, "Rank", range(1, len(df_results) + 1))

        custom_name = input("\nEnter custom file name (or press Enter to use timestamp): ").strip()
        save_file = f"{custom_name or 'results_table_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv"
        save_path = os.path.join(RESULTS_FOLDER, save_file)
        df_results.to_csv(save_path, index=False)
        print(f"\n💾 Results saved to {save_path}")
        print("\n📊 Final Results:")
        print(tabulate(df_results, headers="keys", tablefmt="fancy_grid", showindex=False))

    # === Optional plotting ===
    choice_plot = input("\nEnter one or more tickers to plot (comma-separated, or press Enter to skip): ").upper()
    if choice_plot:
        for t in [x.strip() for x in choice_plot.split(",")]:
            if t in df_results["Ticker"].values:
                print(f"\n📈 Plotting {t} ...")
                data = get_stock_data(t)
                if USE_SMA_FILTER or SHOW_SMAS_ON_CHART:
                    calculate_sma(data)
                calculate_stochastic(data)
                generate_signals(data)
                calculate_annualized_volatility(data)
                plot_stochastic_chart(t, data)

