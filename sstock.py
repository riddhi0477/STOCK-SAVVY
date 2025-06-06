import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

# --- Data Preparation ---
end_date = date.today().strftime("%Y-%m-%d")
start_date = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")
tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS']
data = yf.download(tickers, start=start_date, end=end_date, progress=False)
data = data.reset_index()
data.columns = ['Date'] + [f"{col[1]}_{col[0]}" for col in data.columns[1:]]
data_melted = data.melt(id_vars=['Date'], var_name='Ticker_Attribute')
data_melted[['Ticker', 'Attribute']] = data_melted['Ticker_Attribute'].str.split('_', expand=True)
data_melted = data_melted.drop(columns='Ticker_Attribute')
data_pivoted = data_melted.pivot_table(index=['Date', 'Ticker'], columns='Attribute', values='value', aggfunc='first')
stock_data = data_pivoted.reset_index()

# Add daily returns
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
stock_data['Daily Return'] = stock_data.groupby('Ticker')['Adj Close'].pct_change()

# --- Plotting Helper Functions ---
def plot_adjusted_close(fig):
    sns.lineplot(data=stock_data, x='Date', y='Adj Close', hue='Ticker', marker='o', ax=fig.gca())
    fig.gca().set_title('Adjusted Close Price Over Time')
    fig.gca().set_xlabel('Date')
    fig.gca().set_ylabel('Adjusted Close Price')
    fig.autofmt_xdate()

def plot_moving_averages(fig):
    short_window, long_window = 50, 200
    unique_tickers = stock_data['Ticker'].unique()
    for ticker in unique_tickers:
        ticker_data = stock_data[stock_data['Ticker'] == ticker].copy()
        ticker_data['50_MA'] = ticker_data['Adj Close'].rolling(window=short_window).mean()
        ticker_data['200_MA'] = ticker_data['Adj Close'].rolling(window=long_window).mean()
        ax = fig.add_subplot(len(unique_tickers), 1, list(unique_tickers).index(ticker) + 1)
        ax.plot(ticker_data['Date'], ticker_data['Adj Close'], label='Adj Close')
        ax.plot(ticker_data['Date'], ticker_data['50_MA'], label='50-Day MA')
        ax.plot(ticker_data['Date'], ticker_data['200_MA'], label='200-Day MA')
        ax.set_title(f'{ticker} - Moving Averages')
        ax.legend()
    fig.tight_layout()

def plot_volume_traded(fig):
    unique_tickers = stock_data['Ticker'].unique()
    for ticker in unique_tickers:
        ticker_data = stock_data[stock_data['Ticker'] == ticker]
        ax = fig.add_subplot(len(unique_tickers), 1, list(unique_tickers).index(ticker) + 1)
        ax.bar(ticker_data['Date'], ticker_data['Volume'], label='Volume Traded', color='orange')
        ax.set_title(f'{ticker} - Volume Traded')
        ax.set_xlabel('Date')
        ax.set_ylabel('Volume')
        ax.legend()
    fig.tight_layout()

def plot_daily_returns_distribution(fig):
    sns.histplot(data=stock_data, x='Daily Return', hue='Ticker', kde=True, bins=50, alpha=0.5, ax=fig.gca())
    fig.gca().set_title('Distribution of Daily Returns')
    fig.gca().set_xlabel('Daily Return')
    fig.gca().set_ylabel('Frequency')

def plot_correlation_matrix(fig):
    daily_returns = stock_data.pivot_table(index='Date', columns='Ticker', values='Daily Return')
    correlation_matrix = daily_returns.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=fig.gca())
    fig.gca().set_title('Correlation Matrix of Daily Returns')

def plot_efficient_frontier(fig):
    daily_returns = stock_data.pivot_table(index='Date', columns='Ticker', values='Daily Return')
    expected_returns = daily_returns.mean() * 252
    cov_matrix = daily_returns.cov() * 252
    num_portfolios = 10000
    results = np.zeros((3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = portfolio_return / portfolio_volatility
    fig.gca().scatter(results[1, :], results[0, :], c=results[2, :], cmap='YlGnBu', marker='o')
    fig.gca().set_title('Efficient Frontier')
    fig.gca().set_xlabel('Volatility')
    fig.gca().set_ylabel('Expected Return')
    fig.gca().grid(True)

# --- Tkinter GUI ---
class StockApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Market Visualization")
        self.root.geometry("800x600")

        self.figures = [plot_adjusted_close, plot_moving_averages, plot_volume_traded,
                        plot_daily_returns_distribution, plot_correlation_matrix, plot_efficient_frontier]
        self.current_index = 0

        self.figure = plt.Figure(figsize=(8, 5))
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.nav_frame = ttk.Frame(root)
        self.nav_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.back_button = ttk.Button(self.nav_frame, text="Back", command=self.previous_plot)
        self.back_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.next_button = ttk.Button(self.nav_frame, text="Next", command=self.next_plot)
        self.next_button.pack(side=tk.RIGHT, padx=5, pady=5)

        self.update_plot()

    def update_plot(self):
        self.figure.clf()
        self.figures[self.current_index](self.figure)
        self.canvas.draw()

    def next_plot(self):
        if self.current_index < len(self.figures) - 1:
            self.current_index += 1
            self.update_plot()

    def previous_plot(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_plot()

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = StockApp(root)
    root.mainloop()
    
    
    
   