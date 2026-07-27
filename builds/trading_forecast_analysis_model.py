#+----------------------------------------------------------------------------+
#|                                         trading_forecast_analysis_model.py |
#|          Copyright 2022-2026 HP Investment Trading and Gambling Strategies |
#|                                                        https://hp-fx-g.com |
#+----------------------------------------------------------------------------+

##--- import modules
import os
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as mtick
from matplotlib.collections import LineCollection
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from adjustText import adjust_text
from matplotlib.patches import Rectangle
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.cluster import KMeans
from scipy.signal import argrelextrema
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D

#+----------------------------------------------------------------------------+
#| @func: get parent directory                                                |
#| @desc: finds and returns the parent directory of the current script        |
#| @params: N/A                                                               |
#| @return: parent dir[ectory] --> parent directory of the current script     |
#+----------------------------------------------------------------------------+
def get_parent_directory():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    return parent_dir


#+----------------------------------------------------------------------------+
#| @class: Trading System Forecaster                                          |
#| @desc: forecasts and analyzes the long-term usage of a trading system      |
#| @params: N/A                                                               |
#| @return: N/A                                                               |
#+----------------------------------------------------------------------------+
class Trading_System_Forecaster():
    ##--- create the initialization method
    def __init__(self, cfd_position_records):
        self.cfd_position_records = cfd_position_records
        self.net_profit_series    = None
        
    #+----------------------------------------------------------------------------+
    #| @func: analyze net profit patterns                                         |
    #| @desc: determines the algorithm that will be used to forecast net profits  |
    #| @params: N/A                                                               |
    #| @return: N/A                                                               |
    #+----------------------------------------------------------------------------+
    def analyze_net_profit_patterns(self):
        # --- extract Net Profit series
        net_profit = self.cfd_position_records['Net Profit'].dropna().values
        n = len(net_profit)

        if n < 30:
            raise ValueError("Not enough trades to perform meaningful forecast analysis.")

        # --- basic statistics
        mean_np = np.mean(net_profit)
        median_np = np.median(net_profit)
        std_np = np.std(net_profit, ddof=1)
        skew_np = pd.Series(net_profit).skew()
        kurt_np = pd.Series(net_profit).kurtosis()

        # --- win / loss encoding
        wins_losses = np.where(net_profit > 0, 1, np.where(net_profit < 0, 0, -1))

        # --- lag-1 autocorrelation
        lag1_corr = (
            np.corrcoef(net_profit[:-1], net_profit[1:])[0, 1]
            if np.std(net_profit) > 0 else 0.0
        )
        
        # --- streak analysis
        streaks = []
        current_streak = 1

        for i in range(1, len(wins_losses)):
            if wins_losses[i] == wins_losses[i - 1]:
                current_streak += 1
            else:
                streaks.append(current_streak)
                current_streak = 1
        streaks.append(current_streak)

        avg_streak = np.mean(streaks)
        max_streak = np.max(streaks)

        # --- determine forecast method
        # thresholds are intentionally conservative
        if abs(lag1_corr) > 0.15 or avg_streak > 2.5:
            forecast_method = 'block_bootstrap'
        else:
            forecast_method = 'bootstrap'

        # --- store analysis results
        self.net_profit_analysis = {
            'count': n,
            'mean': mean_np,
            'median': median_np,
            'std': std_np,
            'skew': skew_np,
            'kurtosis': kurt_np,
            'lag1_autocorrelation': lag1_corr,
            'average_streak_length': avg_streak,
            'max_streak_length': max_streak,
            'forecast_method': forecast_method
        }

        # --- also store raw series for later simulation
        self.net_profit_series = net_profit
        
    #+----------------------------------------------------------------------------+
    #| @func: apply execution noise                                               |
    #| @desc: simulates the imperfection of trade execution and profit taking     |
    #| @params: pnl_array --> pnl forecast                                        |
    #|          noise_std --> amount of noise to apply to each profit             |
    #|     max_distortion --> level of noise tolerance                            |
    #| @return: noisy_pnl --> profit taking imperfection applied data             |
    #+----------------------------------------------------------------------------+
    def apply_execution_noise(self, pnl_array, noise_std = 0.05, max_distortion = 0.15):
        noise = np.random.normal(0, noise_std, size=len(pnl_array))
        noise = np.clip(noise, -max_distortion, max_distortion)

        noisy_pnl = pnl_array * (1 + noise)

        # prevent sign flipping due to noise
        sign_flip = np.sign(noisy_pnl) != np.sign(pnl_array)
        noisy_pnl[sign_flip] = pnl_array[sign_flip]

        return noisy_pnl
    
    #+----------------------------------------------------------------------------+
    #| @func: plot actual vs forecast                                             |
    #| @desc: plots the actual PnL in addition to the forecasted PnL              |
    #| @params: forecast_df --> forecasted trade                                  |
    #|            noise_std --> margin of noise error                             |
    #|       max_distortion --> level of noise tolerance for error level tracking |
    #| @return: N/A                                                               |
    #+----------------------------------------------------------------------------+
    def plot_actual_vs_forecast(self, forecast_df, noise_std = 0.05, max_distortion = 0.15):
        # --- cumulative PnL for actual trades
        actual_pnl = self.cfd_position_records['Net Profit'].dropna()
        cumulative_actual = np.cumsum(actual_pnl.values)

        # --- cumulative PnL for forecast trades
        forecast_pnl = forecast_df['Forecasted_Net_Profit'].values
        # --- apply execution noise
        noise = np.random.normal(0, noise_std, size=len(forecast_pnl))
        noise = np.clip(noise, -max_distortion, max_distortion)
        forecast_noisy = forecast_pnl * (1 + noise)
        forecast_noisy = np.array(forecast_noisy)
        
        # --- cumulative forecast
        cumulative_forecast = np.cumsum(forecast_noisy)
        if len(cumulative_actual) > 0:
            cumulative_forecast += cumulative_actual[-1]

        # --- x-axis for actual and forecast
        x_actual = np.arange(1, len(cumulative_actual) + 1)
        x_forecast = np.arange(len(cumulative_actual) + 1, len(cumulative_actual) + len(cumulative_forecast) + 1)

        # --- uncertainty band (± noise_std)
        lower_forecast = cumulative_forecast * (1 - noise_std)
        upper_forecast = cumulative_forecast * (1 + noise_std)

        # --- plot
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(14, 6))

        # actual and forecast lines
        ax.plot(x_actual, cumulative_actual, lw=2.5, color='deepskyblue', label='Actual')
        ax.plot(x_forecast, cumulative_forecast, lw=2.5, linestyle='--', color='gold', label='Forecast')

        # --- fill under curves
        ax.fill_between(x_actual, cumulative_actual, 0, where=cumulative_actual>=0, color='deepskyblue', alpha=0.2)
        ax.fill_between(x_actual, cumulative_actual, 0, where=cumulative_actual<0, color='deepskyblue', alpha=0.15)
        ax.fill_between(x_forecast, cumulative_forecast, 0, where=cumulative_forecast>=0, color='gold', alpha=0.2)
        ax.fill_between(x_forecast, cumulative_forecast, 0, where=cumulative_forecast<0, color='gold', alpha=0.15)

        # --- uncertainty band around forecast
        ax.fill_between(x_forecast, lower_forecast, upper_forecast, color='orange', alpha=0.1, label='Forecast Uncertainty')

        # --- selective labeling at peaks/troughs per regime
        def label_peaks_troughs(x_vals, y_vals, color_positive='deepskyblue', color_negative='orange'):
            y_array = np.array(y_vals)
            sign = np.sign(y_array)
            sign[sign == 0] = 1  # treat zeros as positive

            regime_start = 0
            for i in range(1, len(y_array)):
                if sign[i] != sign[i - 1]:
                    segment = slice(regime_start, i)
                    if sign[i - 1] > 0:  # positive regime
                        idx = segment.start + np.argmax(y_array[segment])
                        va = 'bottom'
                        label_color = color_positive
                        label_text = f"${y_array[idx]:.2f}"
                    else:  # negative regime
                        idx = segment.start + np.argmin(y_array[segment])
                        va = 'top'
                        label_color = color_negative
                        label_text = f"-${abs(y_array[idx]):.2f}"

                    ax.plot(x_vals[idx], y_array[idx], 'o', color=label_color)
                    ax.text(x_vals[idx], y_array[idx], label_text, color=label_color,
                            fontsize=9, ha='left', va=va, fontweight='bold')

                    regime_start = i

            # --- handle final regime
            segment = slice(regime_start, len(y_array))
            if sign[-1] > 0:
                idx = segment.start + np.argmax(y_array[segment])
                va = 'bottom'
                label_color = color_positive
                label_text = f"${y_array[idx]:.2f}"
            else:
                idx = segment.start + np.argmin(y_array[segment])
                va = 'top'
                label_color = color_negative
                label_text = f"-${abs(y_array[idx]):.2f}"

            ax.plot(x_vals[idx], y_array[idx], 'o', color=label_color)
            ax.text(x_vals[idx], y_array[idx], label_text, color=label_color,
                    fontsize=9, ha='left', va=va, fontweight='bold')

        label_peaks_troughs(x_actual, cumulative_actual, color_positive='deepskyblue', color_negative='deepskyblue')
        label_peaks_troughs(x_forecast, cumulative_forecast, color_positive='gold', color_negative='gold')


        # --- zero line
        ax.axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.6)

        # --- labels and title
        ax.set_xlabel('Trade Number', fontsize=12)
        ax.set_ylabel('Cumulative Net Profit', fontsize=12)
        ax.set_title(f'Cumulative Net Profit: Actual with Forecast ({len(forecast_df)} Trade Forecast)', fontsize=16)

        # --- grid
        ax.grid(True, linestyle=':', linewidth=1, alpha=0.4)

        # --- legend top-left
        ax.legend(loc='lower center', fontsize=11)

        plt.tight_layout()
        plt.show()
        
    #+----------------------------------------------------------------------------+
    #| @func: forecast net profits                                                |
    #| @desc: forecasts net profit from the next x trades                         |
    #| @params: n_future --> number of trades to forecast                         |
    #|        block_size --> trade forecast cluster level                         |
    #| @return: N/A                                                               |
    #+----------------------------------------------------------------------------+
    def forecast_net_profits(self, n_future = 25, block_size = 5):
        if not hasattr(self, 'net_profit_analysis'):
            self.analyze_net_profit_patterns()

        method = self.net_profit_analysis['forecast_method']
        historical = self.net_profit_series
        n_hist = len(historical)

        forecast = []

        if method == 'bootstrap':
            forecast = np.random.choice(
                historical,
                size=n_future,
                replace=True
            )

        elif method == 'block_bootstrap':
            while len(forecast) < n_future:
                start_idx = np.random.randint(0, n_hist - block_size + 1)
                block = historical[start_idx:start_idx + block_size]
                forecast.extend(block)

            forecast = np.array(forecast[:n_future])

        else:
            raise ValueError(f"Unknown forecast method: {method}")
        
        ##--- apply execution noise
        forecast = self.apply_execution_noise(forecast)
        
        # --- round to 2 decimal places
        forecast = np.round(forecast, 2)

        # --- build output dataframe
        forecast_df = pd.DataFrame({
            'Trade_Number': np.arange(1, n_future + 1),
            'Forecasted_Net_Profit': forecast
        })
        
        ##--- plot the forecasted PnL
        self.plot_actual_vs_forecast(forecast_df = forecast_df)

    
##--- execute the main method
if __name__ == '__main__':
    ##--- change directory to parent directory
    os.chdir(get_parent_directory())

    ##--- read the position data
    cfd_position_records = pd.read_csv(r'data/complete_cfd_position_records.csv')
    
    ##--- perform trading systen forecast analysis
    trading_system_forecaster = Trading_System_Forecaster(cfd_position_records = cfd_position_records)
    trading_system_forecaster.forecast_net_profits(n_future = (len(cfd_position_records) * 5), block_size = 5)