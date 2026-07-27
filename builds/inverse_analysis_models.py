#+----------------------------------------------------------------------------+
#|                                                    edge_analysis_models.py |
#|          Copyright 2022-2025 HP Investment Trading and Gambling Strategies |
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
#| @class: Inverse Analysis Modeler                                           |
#| @desc: analyzes a profitable system again an unprofitable one              |
#| @params: N/A                                                               |
#| @return: N/A                                                               |
#+----------------------------------------------------------------------------+
class Inverse_Analysis_Modeler():
    ##--- create the initialization method
    def __init__(self, cfd_position_records, inverse_position_records):
        self.cfd_position_records     = cfd_position_records
        self.inverse_position_records = inverse_position_records
        
    #+----------------------------------------------------------------------------+
    #| @func: base inverse PnL                                                    |
    #| @desc: analysis tool for plotting PnL against the actual and inverse data  |
    #| @params: N/A                                                               |
    #| @return: N/A                                                               |
    #+----------------------------------------------------------------------------+
    def base_inverse_pnl(self):
        actual = self.cfd_position_records.sort_values('Date Open').copy()
        inverse = self.inverse_position_records.sort_values('Date Open').copy()
        n = min(len(actual), len(inverse))
        actual = actual.iloc[:n]
        inverse = inverse.iloc[:n]
        trades = np.arange(1, n+1)

        actual_cum_net = actual['Net Profit'].cumsum().values
        inverse_cum_net = inverse['Net Profit'].cumsum().values
        actual_cum_gross = actual['Gross Profit'].cumsum().values
        inverse_cum_gross = inverse['Gross Profit'].cumsum().values
        
        def _plot_comparison(ax, y_actual, y_inverse, y_comm=None, title='',
                            color_actual='deepskyblue', color_inverse='orange', alpha_fill=0.2):
            # lines
            ax.plot(trades, y_actual, lw=2.5, color=color_actual, label='Actual')
            ax.plot(trades, y_inverse, lw=2.5, linestyle='--', color=color_inverse, label='Inverse')

            # filled areas
            ax.fill_between(trades, y_actual, 0, where=y_actual>=0, color=color_actual, alpha=alpha_fill)
            ax.fill_between(trades, y_actual, 0, where=y_actual<0, color=color_actual, alpha=alpha_fill*0.7)
            ax.fill_between(trades, y_inverse, 0, where=y_inverse>=0, color=color_inverse, alpha=alpha_fill)
            ax.fill_between(trades, y_inverse, 0, where=y_inverse<0, color=color_inverse, alpha=alpha_fill*0.7)

            # max divergence markers
            pnl_diff = y_actual - y_inverse
            max_div_idx = np.argmax(np.abs(pnl_diff))
            ax.axvline(trades[max_div_idx], color='white', linestyle=':', alpha=0.6)
            ax.scatter(trades[max_div_idx], y_actual[max_div_idx], s=120, color=color_actual, edgecolors='black')
            ax.scatter(trades[max_div_idx], y_inverse[max_div_idx], s=120, color=color_inverse, edgecolors='black')

            # zero line
            ax.axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.6)

            # --- regime labeling with minimum separation
            for y, color in zip([y_actual, y_inverse], [color_actual, color_inverse]):
                min_distance = int(0.05 * len(y))
                last_labeled_idx = -min_distance
                sign = np.sign(y)
                sign[sign == 0] = 1
                regime_start = 0
                for i in range(1, len(y)):
                    if sign[i] != sign[i-1]:
                        seg = slice(regime_start, i)
                        if sign[i-1] > 0:
                            idx = seg.start + np.argmax(y[seg])
                            va = 'bottom'
                        else:
                            idx = seg.start + np.argmin(y[seg])
                            va = 'top'
                        if idx - last_labeled_idx >= min_distance:
                            ax.plot(trades[idx], y[idx], 'o', color=color)
                            ax.text(trades[idx], y[idx],
                                    f"-${abs(y[idx]):.2f}" if y[idx] < 0 else f"${y[idx]:.2f}",
                                    color=color, fontsize=9, ha='left', va=va, fontweight='bold')
                            last_labeled_idx = idx
                        regime_start = i
                # final regime
                seg = slice(regime_start, len(y))
                if sign[-1] > 0:
                    idx = seg.start + np.argmax(y[seg])
                    va = 'bottom'
                else:
                    idx = seg.start + np.argmin(y[seg])
                    va = 'top'
                if idx - last_labeled_idx >= min_distance:
                    ax.plot(trades[idx], y[idx], 'o', color=color)
                    ax.text(trades[idx], y[idx], f"{y[idx]:.2f}", color=color,
                            fontsize=9, ha='left', va=va, fontweight='bold')

            # --- move Max ΔPnL label to bottom-left
            ax.text(
                0.01, 1.02,
                f"Max ΔPnL = {pnl_diff[max_div_idx]:.2f}",
                transform=ax.transAxes,
                fontsize=10,
                ha='left',
                va='bottom',
                fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=3)
            )

            ax.set_title(title, fontsize=16)
            ax.set_xlabel('Trade Count', fontsize=12)
            ax.set_ylabel('Cumulative PnL', fontsize=12)
            ax.grid(True, linestyle=':', alpha=0.4)
            ax.legend(loc='lower left')

        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 1, figsize=(16,8), sharex=True)
        _plot_comparison(axes[0], actual_cum_net, inverse_cum_net,
                        title='Cumulative Net Profit: Actual vs Inverse')
        _plot_comparison(axes[1], actual_cum_gross, inverse_cum_gross,
                        title='Cumulative Gross Profit: Actual vs Inverse')
        plt.tight_layout()
        plt.show()
        
    #+----------------------------------------------------------------------------+
    #| @func: trrade cost diagnostic analysis                                     |
    #| @desc: analyzes comissison and swap imbalance on a trading system          |
    #| @params: N/A                                                               |
    #| @return: N/A                                                               |
    #+----------------------------------------------------------------------------+
    def trade_cost_diagnostic_analysis(self):
        actual = self.cfd_position_records.sort_values('Date Open').copy()
        inverse = self.inverse_position_records.sort_values('Date Open').copy()
        n = min(len(actual), len(inverse))
        actual = actual.iloc[:n]
        inverse = inverse.iloc[:n]
        trades = np.arange(1, n+1)

        actual_cum_net = actual['Net Profit'].cumsum().values
        inverse_cum_net = inverse['Net Profit'].cumsum().values
        trade_cost_cum = (actual['Commission'] + actual['Swap']).cumsum().values
        actual_cum_gross = actual['Gross Profit'].cumsum().values
        inverse_cum_gross = inverse['Gross Profit'].cumsum().values
        
        ##--- linear regression of commission
        X = trades.reshape(-1, 1)
        y = trade_cost_cum
        reg = LinearRegression()
        reg.fit(X, y)
        trade_cost_lr = reg.predict(X)  # regression line
        slope = reg.coef_[0]
        
        def _plot_comparison(ax, y_actual, y_inverse, y_comm_lr=None, title='',
            color_actual='deepskyblue', color_inverse='orange', alpha_fill=0.2):
            # lines
            ax.plot(trades, y_actual, lw=2.5, color=color_actual, label='Actual')
            ax.plot(trades, y_inverse, lw=2.5, linestyle='--', color=color_inverse, label='Inverse')
            
            # --- commission regression line
            if y_comm_lr is not None:
                ax.plot(
                    trades,
                    y_comm_lr,
                    color='white',
                    lw=2.5,
                    linestyle='-',
                    alpha=0.9,
                    label='Trade Cost Axis Bias'
                )

                # annotate slope
                ax.text(
                    0.11, 1.02,
                    f" | Trade Cost Slope = {slope:.3f}",
                    transform=ax.transAxes,
                    fontsize=10,
                    ha='left',
                    va='bottom',
                    fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=3)
                )
                
                # --- label start and end of commission regression
                start_idx = 0
                end_idx = -1

                for idx, ha in zip([start_idx, end_idx], ['left', 'right']):
                    value = y_comm_lr[idx]
                    label = f"-${abs(value):.2f}" if value < 0 else f"${value:.2f}"

                    ax.plot(
                        trades[idx],
                        value,
                        'o',
                        color='white',
                        markersize=6
                    )

                    ax.text(
                        trades[idx],
                        value,
                        label,
                        color='white',
                        fontsize=9,
                        ha=ha,
                        va='bottom' if value >= 0 else 'top',
                        fontweight='bold'
                    )

            # filled areas
            ax.fill_between(trades, y_actual, 0, where=y_actual>=0, color=color_actual, alpha=alpha_fill)
            ax.fill_between(trades, y_actual, 0, where=y_actual<0, color=color_actual, alpha=alpha_fill*0.7)
            ax.fill_between(trades, y_inverse, 0, where=y_inverse>=0, color=color_inverse, alpha=alpha_fill)
            ax.fill_between(trades, y_inverse, 0, where=y_inverse<0, color=color_inverse, alpha=alpha_fill*0.7)

            # max divergence markers
            pnl_diff = y_actual - y_inverse
            max_div_idx = np.argmax(np.abs(pnl_diff))
            ax.axvline(trades[max_div_idx], color='white', linestyle=':', alpha=0.6)
            ax.scatter(trades[max_div_idx], y_actual[max_div_idx], s=120, color=color_actual, edgecolors='black')
            ax.scatter(trades[max_div_idx], y_inverse[max_div_idx], s=120, color=color_inverse, edgecolors='black')

            # zero line
            ax.axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.6)

            # --- regime labeling with minimum separation
            for y, color in zip([y_actual, y_inverse], [color_actual, color_inverse]):
                min_distance = int(0.05 * len(y))
                last_labeled_idx = -min_distance
                sign = np.sign(y)
                sign[sign == 0] = 1
                regime_start = 0
                for i in range(1, len(y)):
                    if sign[i] != sign[i-1]:
                        seg = slice(regime_start, i)
                        if sign[i-1] > 0:
                            idx = seg.start + np.argmax(y[seg])
                            va = 'bottom'
                        else:
                            idx = seg.start + np.argmin(y[seg])
                            va = 'top'
                        if idx - last_labeled_idx >= min_distance:
                            ax.plot(trades[idx], y[idx], 'o', color=color)
                            ax.text(trades[idx], y[idx],
                                    f"-${abs(y[idx]):.2f}" if y[idx] < 0 else f"${y[idx]:.2f}",
                                    color=color, fontsize=9, ha='left', va=va, fontweight='bold')
                            last_labeled_idx = idx
                        regime_start = i
                # final regime
                seg = slice(regime_start, len(y))
                if sign[-1] > 0:
                    idx = seg.start + np.argmax(y[seg])
                    va = 'bottom'
                else:
                    idx = seg.start + np.argmin(y[seg])
                    va = 'top'
                if idx - last_labeled_idx >= min_distance:
                    ax.plot(trades[idx], y[idx], 'o', color=color)
                    ax.text(trades[idx], y[idx], f"{y[idx]:.2f}", color=color,
                            fontsize=9, ha='left', va=va, fontweight='bold')

            # --- move Max ΔPnL label to bottom-left
            ax.text(
                0.01, 1.02,
                f"Max ΔPnL = {pnl_diff[max_div_idx]:.2f}",
                transform=ax.transAxes,
                fontsize=10,
                ha='left',
                va='bottom',
                fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=3)
            )

            ax.set_title(title, fontsize=16)
            ax.set_xlabel('Trade Count', fontsize=12)
            ax.set_ylabel('Cumulative PnL', fontsize=12)
            ax.grid(True, linestyle=':', alpha=0.4)
            ax.legend(loc='lower left')

        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 1, figsize=(16,8), sharex=True)
        _plot_comparison(axes[0], actual_cum_net, inverse_cum_net, y_comm_lr=trade_cost_lr,
                        title='Cumulative Net Profit: Actual vs Inverse')
        _plot_comparison(axes[1], actual_cum_gross, inverse_cum_gross,
                        title='Cumulative Gross Profit: Actual vs Inverse')
        plt.tight_layout()
        plt.show()
        
    #+----------------------------------------------------------------------------+
    #| @func: trade cost solution analysis                                        |
    #| @desc: proposes a solution to trade cost imbalance in a trading system     |
    #| @params: N/A                                                               |
    #| @return: N/A                                                               |
    #+----------------------------------------------------------------------------+
    def trade_cost_solution_analysis(self):
        actual = self.cfd_position_records.sort_values('Date Open').copy()
        inverse = self.inverse_position_records.sort_values('Date Open').copy()
        n = min(len(actual), len(inverse))
        actual = actual.iloc[:n]
        inverse = inverse.iloc[:n]
        trades = np.arange(1, n+1)

        ##--- calculate the trade cost sum
        pre_trade_cost_cum = (actual['Commission'] + actual['Swap']).cumsum().values
        
        ##--- linear regression of trade cost
        pre_X = trades.reshape(-1, 1)
        pre_y = pre_trade_cost_cum
        reg = LinearRegression()
        reg.fit(pre_X, pre_y)
        pre_trade_cost_lr = reg.predict(pre_X)  # regression line
        pre_slope = reg.coef_[0]
        
        ##--- total trade cost drag (from regression endpoints)
        pre_total_trade_cost = pre_trade_cost_lr[-1] - pre_trade_cost_lr[0]

        ##--- per-trade edge required to neutralize costs
        raw_per_trade_offset = abs(pre_total_trade_cost) / n
        per_trade_offset = math.ceil(raw_per_trade_offset * 100) / 100  # ceiling to cents
        
        ##--- synthetic refund column (edge compensation)
        actual['Refund'] = per_trade_offset
        inverse['Refund'] = per_trade_offset
        
        # --- apply per-trade edge back into profits (solution scenario)
        actual_adj = actual.copy()
        inverse_adj = inverse.copy()
        actual_adj['Gross Profit'] += per_trade_offset
        actual_adj['Net Profit']   += per_trade_offset
        inverse_adj['Gross Profit'] += per_trade_offset
        inverse_adj['Net Profit']   += per_trade_offset
        
        trade_cost_cum = (
            actual['Commission']
            + actual['Swap']
            + actual['Refund']
        ).cumsum().values
        
        X = trades.reshape(-1, 1)
        y = trade_cost_cum
        reg = LinearRegression()
        reg.fit(X, y)
        trade_cost_lr = reg.predict(X)
        slope = reg.coef_[0]
        
        actual_cum_net = actual_adj['Net Profit'].cumsum().values
        inverse_cum_net = inverse_adj['Net Profit'].cumsum().values
        actual_cum_gross = actual_adj['Gross Profit'].cumsum().values
        inverse_cum_gross = inverse_adj['Gross Profit'].cumsum().values
        
        def _plot_comparison(ax, y_actual, y_inverse, y_comm_lr=None, title='',
            color_actual='deepskyblue', color_inverse='orange', alpha_fill=0.2):
            # lines
            ax.plot(trades, y_actual, lw=2.5, color=color_actual, label='Actual')
            ax.plot(trades, y_inverse, lw=2.5, linestyle='--', color=color_inverse, label='Inverse')
            
            # --- commission regression line
            if y_comm_lr is not None:
                ax.plot(
                    trades,
                    y_comm_lr,
                    color='white',
                    lw=2.5,
                    linestyle='-',
                    alpha=0.9,
                    label='Trade Cost Axis Bias'
                )

                ax.text(
                    0.11, 1.02,
                    f" | Edge Needed to Offset Trade Cost = ${per_trade_offset:.2f}",
                    transform=ax.transAxes,
                    fontsize=10,
                    ha='left',
                    va='bottom',
                    fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=3)
                )
                
                # --- label start and end of commission regression
                start_idx = 0
                end_idx = -1

                for idx, ha in zip([start_idx, end_idx], ['left', 'right']):
                    value = y_comm_lr[idx]
                    label = f"-${abs(value):.2f}" if value < 0 else f"${value:.2f}"

                    ax.plot(
                        trades[idx],
                        value,
                        'o',
                        color='white',
                        markersize=6
                    )

                    ax.text(
                        trades[idx],
                        value,
                        label,
                        color='white',
                        fontsize=9,
                        ha=ha,
                        va='bottom' if value >= 0 else 'top',
                        fontweight='bold'
                    )

            # filled areas
            ax.fill_between(trades, y_actual, 0, where=y_actual>=0, color=color_actual, alpha=alpha_fill)
            ax.fill_between(trades, y_actual, 0, where=y_actual<0, color=color_actual, alpha=alpha_fill*0.7)
            ax.fill_between(trades, y_inverse, 0, where=y_inverse>=0, color=color_inverse, alpha=alpha_fill)
            ax.fill_between(trades, y_inverse, 0, where=y_inverse<0, color=color_inverse, alpha=alpha_fill*0.7)

            # max divergence markers
            pnl_diff = y_actual - y_inverse
            max_div_idx = np.argmax(np.abs(pnl_diff))
            ax.axvline(trades[max_div_idx], color='white', linestyle=':', alpha=0.6)
            ax.scatter(trades[max_div_idx], y_actual[max_div_idx], s=120, color=color_actual, edgecolors='black')
            ax.scatter(trades[max_div_idx], y_inverse[max_div_idx], s=120, color=color_inverse, edgecolors='black')

            # zero line
            ax.axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.6)

            # --- regime labeling with minimum separation
            for y, color in zip([y_actual, y_inverse], [color_actual, color_inverse]):
                min_distance = int(0.05 * len(y))
                last_labeled_idx = -min_distance
                sign = np.sign(y)
                sign[sign == 0] = 1
                regime_start = 0
                for i in range(1, len(y)):
                    if sign[i] != sign[i-1]:
                        seg = slice(regime_start, i)
                        if sign[i-1] > 0:
                            idx = seg.start + np.argmax(y[seg])
                            va = 'bottom'
                        else:
                            idx = seg.start + np.argmin(y[seg])
                            va = 'top'
                        if idx - last_labeled_idx >= min_distance:
                            ax.plot(trades[idx], y[idx], 'o', color=color)
                            ax.text(trades[idx], y[idx],
                                    f"-${abs(y[idx]):.2f}" if y[idx] < 0 else f"${y[idx]:.2f}",
                                    color=color, fontsize=9, ha='left', va=va, fontweight='bold')
                            last_labeled_idx = idx
                        regime_start = i
                # final regime
                seg = slice(regime_start, len(y))
                if sign[-1] > 0:
                    idx = seg.start + np.argmax(y[seg])
                    va = 'bottom'
                else:
                    idx = seg.start + np.argmin(y[seg])
                    va = 'top'
                if idx - last_labeled_idx >= min_distance:
                    ax.plot(trades[idx], y[idx], 'o', color=color)
                    ax.text(trades[idx], y[idx], f"{y[idx]:.2f}", color=color,
                            fontsize=9, ha='left', va=va, fontweight='bold')

            # --- move Max ΔPnL label to bottom-left
            ax.text(
                0.01, 1.02,
                f"Max ΔPnL = {pnl_diff[max_div_idx]:.2f}",
                transform=ax.transAxes,
                fontsize=10,
                ha='left',
                va='bottom',
                fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=3)
            )

            ax.set_title(title, fontsize=16)
            ax.set_xlabel('Trade Count', fontsize=12)
            ax.set_ylabel('Cumulative PnL', fontsize=12)
            ax.grid(True, linestyle=':', alpha=0.4)
            ax.legend(loc='lower left')

        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 1, figsize=(16,8), sharex=True)
        _plot_comparison(axes[0], actual_cum_net, inverse_cum_net, y_comm_lr=trade_cost_lr,
                        title='Cumulative Net Profit: Actual vs Inverse')
        _plot_comparison(axes[1], actual_cum_gross, inverse_cum_gross,
                        title='Cumulative Gross Profit: Actual vs Inverse')
        plt.tight_layout()
        plt.show()
        
    #+----------------------------------------------------------------------------+
    #| @func: symbol actual incompatiability diagnostic analyis                   |
    #| @desc: identifies symbols least compatible with a trading system           |
    #| @params: N/A                                                               |
    #| @return: N/A                                                               |
    #+----------------------------------------------------------------------------+
    def symbol_actual_incompatability_diagnostic_analysis(self): 
        # --- aggregate per-symbol stats
        symbol_stats = self.cfd_position_records.groupby('Symbol').agg(
            trade_count=('Net Profit', 'count'),
            lifetime_profit=('Net Profit', 'sum')
        )

        # --- compute z-score of lifetime profit
        profit_mean = symbol_stats['lifetime_profit'].mean()
        profit_std  = symbol_stats['lifetime_profit'].std()
        symbol_stats['Profit Z'] = (symbol_stats['lifetime_profit'] - profit_mean) / profit_std

        # --- negative-edge symbols (z <= threshold)
        negative_symbols = symbol_stats[symbol_stats['Profit Z'] <= -0.75].copy()
        positive_symbols = symbol_stats[symbol_stats['Profit Z'] >= 0.90].copy()

        # --- sort by negative impact
        negative_symbols.sort_values('Profit Z', inplace=True)
        positive_symbols.sort_values('Profit Z', inplace=True)
        
        # --- get list of symbols to remove
        worst_symbols = negative_symbols.index.tolist()
        best_symbols  = positive_symbols.index.tolist()
        
        # --- filter datasets
        actual = self.cfd_position_records[~self.cfd_position_records['Symbol'].isin(worst_symbols)].copy()
        inverse = self.inverse_position_records[~self.inverse_position_records['Symbol'].isin(worst_symbols)].copy()
        
        n = min(len(actual), len(inverse))
        actual = actual.iloc[:n]
        inverse = inverse.iloc[:n]
        trades = np.arange(1, n+1)

        actual_cum_net = actual['Net Profit'].cumsum().values
        inverse_cum_net = inverse['Net Profit'].cumsum().values
        actual_cum_gross = actual['Gross Profit'].cumsum().values
        inverse_cum_gross = inverse['Gross Profit'].cumsum().values
        
        def _plot_comparison(ax, y_actual, y_inverse, y_comm=None, title='',
                            color_actual='deepskyblue', color_inverse='orange', alpha_fill=0.2):
            # lines
            ax.plot(trades, y_actual, lw=2.5, color=color_actual, label='Actual')
            ax.plot(trades, y_inverse, lw=2.5, linestyle='--', color=color_inverse, label='Inverse')

            # filled areas
            ax.fill_between(trades, y_actual, 0, where=y_actual>=0, color=color_actual, alpha=alpha_fill)
            ax.fill_between(trades, y_actual, 0, where=y_actual<0, color=color_actual, alpha=alpha_fill*0.7)
            ax.fill_between(trades, y_inverse, 0, where=y_inverse>=0, color=color_inverse, alpha=alpha_fill)
            ax.fill_between(trades, y_inverse, 0, where=y_inverse<0, color=color_inverse, alpha=alpha_fill*0.7)

            # max divergence markers
            pnl_diff = y_actual - y_inverse
            max_div_idx = np.argmax(np.abs(pnl_diff))
            ax.axvline(trades[max_div_idx], color='white', linestyle=':', alpha=0.6)
            ax.scatter(trades[max_div_idx], y_actual[max_div_idx], s=120, color=color_actual, edgecolors='black')
            ax.scatter(trades[max_div_idx], y_inverse[max_div_idx], s=120, color=color_inverse, edgecolors='black')

            # zero line
            ax.axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.6)

            # --- regime labeling with minimum separation
            for y, color in zip([y_actual, y_inverse], [color_actual, color_inverse]):
                min_distance = int(0.05 * len(y))
                last_labeled_idx = -min_distance
                sign = np.sign(y)
                sign[sign == 0] = 1
                regime_start = 0
                for i in range(1, len(y)):
                    if sign[i] != sign[i-1]:
                        seg = slice(regime_start, i)
                        if sign[i-1] > 0:
                            idx = seg.start + np.argmax(y[seg])
                            va = 'bottom'
                        else:
                            idx = seg.start + np.argmin(y[seg])
                            va = 'top'
                        if idx - last_labeled_idx >= min_distance:
                            ax.plot(trades[idx], y[idx], 'o', color=color)
                            ax.text(trades[idx], y[idx],
                                    f"-${abs(y[idx]):.2f}" if y[idx] < 0 else f"${y[idx]:.2f}",
                                    color=color, fontsize=9, ha='left', va=va, fontweight='bold')
                            last_labeled_idx = idx
                        regime_start = i
                # final regime
                seg = slice(regime_start, len(y))
                if sign[-1] > 0:
                    idx = seg.start + np.argmax(y[seg])
                    va = 'bottom'
                else:
                    idx = seg.start + np.argmin(y[seg])
                    va = 'top'
                if idx - last_labeled_idx >= min_distance:
                    ax.plot(trades[idx], y[idx], 'o', color=color)
                    ax.text(trades[idx], y[idx], f"{y[idx]:.2f}", color=color,
                            fontsize=9, ha='left', va=va, fontweight='bold')

            # --- move Max ΔPnL label to bottom-left
            ax.text(
                0.01, 1.02,
                f"Max ΔPnL = {pnl_diff[max_div_idx]:.2f}",
                transform=ax.transAxes,
                fontsize=10,
                ha='left',
                va='bottom',
                fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=3)
            )

            ax.set_title(title, fontsize=16)
            ax.set_xlabel('Trade Count', fontsize=12)
            ax.set_ylabel('Cumulative PnL', fontsize=12)
            ax.grid(True, linestyle=':', alpha=0.4)
            ax.legend(loc='upper left')

        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 1, figsize=(16,8), sharex=True)
        _plot_comparison(axes[0], actual_cum_net, inverse_cum_net,
                        title='Cumulative Net Profit: Actual vs Inverse')
        _plot_comparison(axes[1], actual_cum_gross, inverse_cum_gross,
                        title='Cumulative Gross Profit: Actual vs Inverse')
        plt.tight_layout()
        plt.show()
        
    #+----------------------------------------------------------------------------+
    #| @func: symbol inverse incompatiability diagnostic analyis                  |
    #| @desc: identifies symbols least compatible with an inverse trading system  |
    #| @params: N/A                                                               |
    #| @return: N/A                                                               |
    #+----------------------------------------------------------------------------+
    def symbol_inverse_incompatability_diagnostic_analysis(self): 
        # --- aggregate per-symbol stats
        symbol_stats = self.cfd_position_records.groupby('Symbol').agg(
            trade_count=('Net Profit', 'count'),
            lifetime_profit=('Net Profit', 'sum')
        )

        # --- compute z-score of lifetime profit
        profit_mean = symbol_stats['lifetime_profit'].mean()
        profit_std  = symbol_stats['lifetime_profit'].std()
        symbol_stats['Profit Z'] = (symbol_stats['lifetime_profit'] - profit_mean) / profit_std

        # --- negative-edge symbols (z <= threshold)
        negative_symbols = symbol_stats[symbol_stats['Profit Z'] <= -0.75].copy()
        positive_symbols = symbol_stats[symbol_stats['Profit Z'] >= 0.90].copy()

        # --- sort by negative impact
        negative_symbols.sort_values('Profit Z', inplace=True)
        positive_symbols.sort_values('Profit Z', inplace=True)
        
        # --- get list of symbols to remove
        worst_symbols = negative_symbols.index.tolist()
        best_symbols  = positive_symbols.index.tolist()
        
        # --- filter datasets
        actual = self.cfd_position_records[~self.cfd_position_records['Symbol'].isin(best_symbols)].copy()
        inverse = self.inverse_position_records[~self.inverse_position_records['Symbol'].isin(best_symbols)].copy()
        
        n = min(len(actual), len(inverse))
        actual = actual.iloc[:n]
        inverse = inverse.iloc[:n]
        trades = np.arange(1, n+1)

        actual_cum_net = actual['Net Profit'].cumsum().values
        inverse_cum_net = inverse['Net Profit'].cumsum().values
        actual_cum_gross = actual['Gross Profit'].cumsum().values
        inverse_cum_gross = inverse['Gross Profit'].cumsum().values
        
        def _plot_comparison(ax, y_actual, y_inverse, y_comm=None, title='',
                            color_actual='deepskyblue', color_inverse='orange', alpha_fill=0.2):
            # lines
            ax.plot(trades, y_actual, lw=2.5, color=color_actual, label='Actual')
            ax.plot(trades, y_inverse, lw=2.5, linestyle='--', color=color_inverse, label='Inverse')

            # filled areas
            ax.fill_between(trades, y_actual, 0, where=y_actual>=0, color=color_actual, alpha=alpha_fill)
            ax.fill_between(trades, y_actual, 0, where=y_actual<0, color=color_actual, alpha=alpha_fill*0.7)
            ax.fill_between(trades, y_inverse, 0, where=y_inverse>=0, color=color_inverse, alpha=alpha_fill)
            ax.fill_between(trades, y_inverse, 0, where=y_inverse<0, color=color_inverse, alpha=alpha_fill*0.7)

            # max divergence markers
            pnl_diff = y_actual - y_inverse
            max_div_idx = np.argmax(np.abs(pnl_diff))
            ax.axvline(trades[max_div_idx], color='white', linestyle=':', alpha=0.6)
            ax.scatter(trades[max_div_idx], y_actual[max_div_idx], s=120, color=color_actual, edgecolors='black')
            ax.scatter(trades[max_div_idx], y_inverse[max_div_idx], s=120, color=color_inverse, edgecolors='black')

            # zero line
            ax.axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.6)

            # --- regime labeling with minimum separation
            for y, color in zip([y_actual, y_inverse], [color_actual, color_inverse]):
                min_distance = int(0.05 * len(y))
                last_labeled_idx = -min_distance
                sign = np.sign(y)
                sign[sign == 0] = 1
                regime_start = 0
                for i in range(1, len(y)):
                    if sign[i] != sign[i-1]:
                        seg = slice(regime_start, i)
                        if sign[i-1] > 0:
                            idx = seg.start + np.argmax(y[seg])
                            va = 'bottom'
                        else:
                            idx = seg.start + np.argmin(y[seg])
                            va = 'top'
                        if idx - last_labeled_idx >= min_distance:
                            ax.plot(trades[idx], y[idx], 'o', color=color)
                            ax.text(trades[idx], y[idx],
                                    f"-${abs(y[idx]):.2f}" if y[idx] < 0 else f"${y[idx]:.2f}",
                                    color=color, fontsize=9, ha='left', va=va, fontweight='bold')
                            last_labeled_idx = idx
                        regime_start = i
                # final regime
                seg = slice(regime_start, len(y))
                if sign[-1] > 0:
                    idx = seg.start + np.argmax(y[seg])
                    va = 'bottom'
                else:
                    idx = seg.start + np.argmin(y[seg])
                    va = 'top'
                if idx - last_labeled_idx >= min_distance:
                    ax.plot(trades[idx], y[idx], 'o', color=color)
                    ax.text(trades[idx], y[idx], f"{y[idx]:.2f}", color=color,
                            fontsize=9, ha='left', va=va, fontweight='bold')

            # --- move Max ΔPnL label to bottom-left
            ax.text(
                0.01, 1.02,
                f"Max ΔPnL = {pnl_diff[max_div_idx]:.2f}",
                transform=ax.transAxes,
                fontsize=10,
                ha='left',
                va='bottom',
                fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=3)
            )

            ax.set_title(title, fontsize=16)
            ax.set_xlabel('Trade Count', fontsize=12)
            ax.set_ylabel('Cumulative PnL', fontsize=12)
            ax.grid(True, linestyle=':', alpha=0.4)
            ax.legend(loc='upper left')

        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 1, figsize=(16,8), sharex=True)
        _plot_comparison(axes[0], actual_cum_net, inverse_cum_net,
                        title='Cumulative Net Profit: Actual vs Inverse')
        _plot_comparison(axes[1], actual_cum_gross, inverse_cum_gross,
                        title='Cumulative Gross Profit: Actual vs Inverse')
        plt.tight_layout()
        plt.show()
    
    #+----------------------------------------------------------------------------+
    #| @func: symbol comparison incompatiability diagnostic analyis               |
    #| @desc: identifies symbols least compatible with an both trading systems    |
    #| @params: N/A                                                               |
    #| @return: N/A                                                               |
    #+----------------------------------------------------------------------------+  
    def symbol_comparison_incompatability_diagnostic_analysis(self): 
        # --- aggregate per-symbol stats
        symbol_stats = self.cfd_position_records.groupby('Symbol').agg(
            trade_count=('Net Profit', 'count'),
            lifetime_profit=('Net Profit', 'sum')
        )

        # --- compute z-score of lifetime profit
        profit_mean = symbol_stats['lifetime_profit'].mean()
        profit_std  = symbol_stats['lifetime_profit'].std()
        symbol_stats['Profit Z'] = (symbol_stats['lifetime_profit'] - profit_mean) / profit_std

        # --- negative-edge symbols (z <= threshold)
        negative_symbols = symbol_stats[symbol_stats['Profit Z'] <= -0.75].copy()
        positive_symbols = symbol_stats[symbol_stats['Profit Z'] >= 0.90].copy()

        # --- sort by negative impact
        negative_symbols.sort_values('Profit Z', inplace=True)
        positive_symbols.sort_values('Profit Z', inplace=True)
        
        # --- get list of symbols to remove
        worst_symbols = negative_symbols.index.tolist()
        best_symbols  = positive_symbols.index.tolist()
        
        # --- filter datasets
        actual = self.cfd_position_records[~self.cfd_position_records['Symbol'].isin(worst_symbols)].copy()
        inverse = self.inverse_position_records[~self.inverse_position_records['Symbol'].isin(best_symbols)].copy()
        
        n = min(len(actual), len(inverse))
        actual = actual.iloc[:n]
        inverse = inverse.iloc[:n]
        trades = np.arange(1, n+1)

        actual_cum_net = actual['Net Profit'].cumsum().values
        inverse_cum_net = inverse['Net Profit'].cumsum().values
        actual_cum_gross = actual['Gross Profit'].cumsum().values
        inverse_cum_gross = inverse['Gross Profit'].cumsum().values
        
        def _plot_comparison(ax, y_actual, y_inverse, y_comm=None, title='',
                            color_actual='deepskyblue', color_inverse='orange', alpha_fill=0.2):
            # lines
            ax.plot(trades, y_actual, lw=2.5, color=color_actual, label='Actual')
            ax.plot(trades, y_inverse, lw=2.5, linestyle='--', color=color_inverse, label='Inverse')

            # filled areas
            ax.fill_between(trades, y_actual, 0, where=y_actual>=0, color=color_actual, alpha=alpha_fill)
            ax.fill_between(trades, y_actual, 0, where=y_actual<0, color=color_actual, alpha=alpha_fill*0.7)
            ax.fill_between(trades, y_inverse, 0, where=y_inverse>=0, color=color_inverse, alpha=alpha_fill)
            ax.fill_between(trades, y_inverse, 0, where=y_inverse<0, color=color_inverse, alpha=alpha_fill*0.7)

            # max divergence markers
            pnl_diff = y_actual - y_inverse
            max_div_idx = np.argmax(np.abs(pnl_diff))
            ax.axvline(trades[max_div_idx], color='white', linestyle=':', alpha=0.6)
            ax.scatter(trades[max_div_idx], y_actual[max_div_idx], s=120, color=color_actual, edgecolors='black')
            ax.scatter(trades[max_div_idx], y_inverse[max_div_idx], s=120, color=color_inverse, edgecolors='black')

            # zero line
            ax.axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.6)

            # --- regime labeling with minimum separation
            for y, color in zip([y_actual, y_inverse], [color_actual, color_inverse]):
                min_distance = int(0.05 * len(y))
                last_labeled_idx = -min_distance
                sign = np.sign(y)
                sign[sign == 0] = 1
                regime_start = 0
                for i in range(1, len(y)):
                    if sign[i] != sign[i-1]:
                        seg = slice(regime_start, i)
                        if sign[i-1] > 0:
                            idx = seg.start + np.argmax(y[seg])
                            va = 'bottom'
                        else:
                            idx = seg.start + np.argmin(y[seg])
                            va = 'top'
                        if idx - last_labeled_idx >= min_distance:
                            ax.plot(trades[idx], y[idx], 'o', color=color)
                            ax.text(trades[idx], y[idx],
                                    f"-${abs(y[idx]):.2f}" if y[idx] < 0 else f"${y[idx]:.2f}",
                                    color=color, fontsize=9, ha='left', va=va, fontweight='bold')
                            last_labeled_idx = idx
                        regime_start = i
                # final regime
                seg = slice(regime_start, len(y))
                if sign[-1] > 0:
                    idx = seg.start + np.argmax(y[seg])
                    va = 'bottom'
                else:
                    idx = seg.start + np.argmin(y[seg])
                    va = 'top'
                if idx - last_labeled_idx >= min_distance:
                    ax.plot(trades[idx], y[idx], 'o', color=color)
                    ax.text(trades[idx], y[idx], f"{y[idx]:.2f}", color=color,
                            fontsize=9, ha='left', va=va, fontweight='bold')

            # --- move Max ΔPnL label to bottom-left
            ax.text(
                0.01, 1.02,
                f"Max ΔPnL = {pnl_diff[max_div_idx]:.2f}",
                transform=ax.transAxes,
                fontsize=10,
                ha='left',
                va='bottom',
                fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=3)
            )

            ax.set_title(title, fontsize=16)
            ax.set_xlabel('Trade Count', fontsize=12)
            ax.set_ylabel('Cumulative PnL', fontsize=12)
            ax.grid(True, linestyle=':', alpha=0.4)
            ax.legend(loc='upper left')

        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 1, figsize=(16,8), sharex=True)
        _plot_comparison(axes[0], actual_cum_net, inverse_cum_net,
                        title='Cumulative Net Profit: Actual vs Inverse')
        _plot_comparison(axes[1], actual_cum_gross, inverse_cum_gross,
                        title='Cumulative Gross Profit: Actual vs Inverse')
        plt.tight_layout()
        plt.show()

    #+----------------------------------------------------------------------------+
    #| @func: trade cost symbol analysis                                          |
    #| @desc: proposes a solution to trade cost imbalance in a trading system     |
    #| @params: N/A                                                               |
    #| @return: N/A                                                               |
    #+----------------------------------------------------------------------------+
    def trade_cost_symbol_analysis(self):
        # --- aggregate per-symbol stats
        symbol_stats = self.cfd_position_records.groupby('Symbol').agg(
            trade_count=('Net Profit', 'count'),
            lifetime_profit=('Net Profit', 'sum')
        )

        # --- compute z-score of lifetime profit
        profit_mean = symbol_stats['lifetime_profit'].mean()
        profit_std  = symbol_stats['lifetime_profit'].std()
        symbol_stats['Profit Z'] = (symbol_stats['lifetime_profit'] - profit_mean) / profit_std

        # --- negative-edge symbols (z <= threshold)
        negative_symbols = symbol_stats[symbol_stats['Profit Z'] <= -0.75].copy()
        positive_symbols = symbol_stats[symbol_stats['Profit Z'] >= 0.90].copy()

        # --- sort by negative impact
        negative_symbols.sort_values('Profit Z', inplace=True)
        positive_symbols.sort_values('Profit Z', inplace=True)
        
        # --- get list of symbols to remove
        worst_symbols = negative_symbols.index.tolist()
        best_symbols  = positive_symbols.index.tolist()
        
        # --- filter datasets
        actual = self.cfd_position_records[~self.cfd_position_records['Symbol'].isin(worst_symbols)].copy()
        inverse = self.inverse_position_records[~self.inverse_position_records['Symbol'].isin(best_symbols)].copy()
        
        n = min(len(actual), len(inverse))
        actual = actual.iloc[:n]
        inverse = inverse.iloc[:n]
        trades = np.arange(1, n+1)

        ##--- calculate the trade cost sum
        pre_trade_cost_cum = (actual['Commission'] + actual['Swap']).cumsum().values
        
        ##--- linear regression of trade cost
        pre_X = trades.reshape(-1, 1)
        pre_y = pre_trade_cost_cum
        reg = LinearRegression()
        reg.fit(pre_X, pre_y)
        pre_trade_cost_lr = reg.predict(pre_X)  # regression line
        pre_slope = reg.coef_[0]
        
        ##--- total trade cost drag (from regression endpoints)
        pre_total_trade_cost = pre_trade_cost_lr[-1] - pre_trade_cost_lr[0]

        ##--- per-trade edge required to neutralize costs
        raw_per_trade_offset = abs(pre_total_trade_cost) / n
        per_trade_offset = math.ceil(raw_per_trade_offset * 100) / 100  # ceiling to cents
        
        ##--- synthetic refund column (edge compensation)
        actual['Refund'] = per_trade_offset
        inverse['Refund'] = per_trade_offset
        
        # --- apply per-trade edge back into profits (solution scenario)
        actual_adj = actual.copy()
        inverse_adj = inverse.copy()
        actual_adj['Gross Profit'] += per_trade_offset
        actual_adj['Net Profit']   += per_trade_offset
        inverse_adj['Gross Profit'] += per_trade_offset
        inverse_adj['Net Profit']   += per_trade_offset
        
        trade_cost_cum = (
            actual['Commission']
            + actual['Swap']
            + actual['Refund']
        ).cumsum().values
        
        X = trades.reshape(-1, 1)
        y = trade_cost_cum
        reg = LinearRegression()
        reg.fit(X, y)
        trade_cost_lr = reg.predict(X)
        slope = reg.coef_[0]
        
        actual_cum_net = actual_adj['Net Profit'].cumsum().values
        inverse_cum_net = inverse_adj['Net Profit'].cumsum().values
        actual_cum_gross = actual_adj['Gross Profit'].cumsum().values
        inverse_cum_gross = inverse_adj['Gross Profit'].cumsum().values
        
        def _plot_comparison(ax, y_actual, y_inverse, y_comm_lr=None, title='',
            color_actual='deepskyblue', color_inverse='orange', alpha_fill=0.2):
            # lines
            ax.plot(trades, y_actual, lw=2.5, color=color_actual, label='Actual')
            ax.plot(trades, y_inverse, lw=2.5, linestyle='--', color=color_inverse, label='Inverse')
            
            # --- commission regression line
            if y_comm_lr is not None:
                ax.plot(
                    trades,
                    y_comm_lr,
                    color='white',
                    lw=2.5,
                    linestyle='-',
                    alpha=0.9,
                    label='Trade Cost Axis Bias'
                )

                ax.text(
                    0.11, 1.02,
                    f" | Edge Needed to Offset Trade Cost = ${per_trade_offset:.2f}",
                    transform=ax.transAxes,
                    fontsize=10,
                    ha='left',
                    va='bottom',
                    fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=3)
                )
                
                # --- label start and end of commission regression
                start_idx = 0
                end_idx = -1

                for idx, ha in zip([start_idx, end_idx], ['left', 'right']):
                    value = y_comm_lr[idx]
                    label = f"-${abs(value):.2f}" if value < 0 else f"${value:.2f}"

                    ax.plot(
                        trades[idx],
                        value,
                        'o',
                        color='white',
                        markersize=6
                    )

                    ax.text(
                        trades[idx],
                        value,
                        label,
                        color='white',
                        fontsize=9,
                        ha=ha,
                        va='bottom' if value >= 0 else 'top',
                        fontweight='bold'
                    )

            # filled areas
            ax.fill_between(trades, y_actual, 0, where=y_actual>=0, color=color_actual, alpha=alpha_fill)
            ax.fill_between(trades, y_actual, 0, where=y_actual<0, color=color_actual, alpha=alpha_fill*0.7)
            ax.fill_between(trades, y_inverse, 0, where=y_inverse>=0, color=color_inverse, alpha=alpha_fill)
            ax.fill_between(trades, y_inverse, 0, where=y_inverse<0, color=color_inverse, alpha=alpha_fill*0.7)

            # max divergence markers
            pnl_diff = y_actual - y_inverse
            max_div_idx = np.argmax(np.abs(pnl_diff))
            ax.axvline(trades[max_div_idx], color='white', linestyle=':', alpha=0.6)
            ax.scatter(trades[max_div_idx], y_actual[max_div_idx], s=120, color=color_actual, edgecolors='black')
            ax.scatter(trades[max_div_idx], y_inverse[max_div_idx], s=120, color=color_inverse, edgecolors='black')

            # zero line
            ax.axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.6)

            # --- regime labeling with minimum separation
            for y, color in zip([y_actual, y_inverse], [color_actual, color_inverse]):
                min_distance = int(0.05 * len(y))
                last_labeled_idx = -min_distance
                sign = np.sign(y)
                sign[sign == 0] = 1
                regime_start = 0
                for i in range(1, len(y)):
                    if sign[i] != sign[i-1]:
                        seg = slice(regime_start, i)
                        if sign[i-1] > 0:
                            idx = seg.start + np.argmax(y[seg])
                            va = 'bottom'
                        else:
                            idx = seg.start + np.argmin(y[seg])
                            va = 'top'
                        if idx - last_labeled_idx >= min_distance:
                            ax.plot(trades[idx], y[idx], 'o', color=color)
                            ax.text(trades[idx], y[idx],
                                    f"-${abs(y[idx]):.2f}" if y[idx] < 0 else f"${y[idx]:.2f}",
                                    color=color, fontsize=9, ha='left', va=va, fontweight='bold')
                            last_labeled_idx = idx
                        regime_start = i
                # final regime
                seg = slice(regime_start, len(y))
                if sign[-1] > 0:
                    idx = seg.start + np.argmax(y[seg])
                    va = 'bottom'
                else:
                    idx = seg.start + np.argmin(y[seg])
                    va = 'top'
                if idx - last_labeled_idx >= min_distance:
                    ax.plot(trades[idx], y[idx], 'o', color=color)
                    ax.text(trades[idx], y[idx], f"{y[idx]:.2f}", color=color,
                            fontsize=9, ha='left', va=va, fontweight='bold')

            # --- move Max ΔPnL label to bottom-left
            ax.text(
                0.01, 1.02,
                f"Max ΔPnL = {pnl_diff[max_div_idx]:.2f}",
                transform=ax.transAxes,
                fontsize=10,
                ha='left',
                va='bottom',
                fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=3)
            )

            ax.set_title(title, fontsize=16)
            ax.set_xlabel('Trade Count', fontsize=12)
            ax.set_ylabel('Cumulative PnL', fontsize=12)
            ax.grid(True, linestyle=':', alpha=0.4)
            ax.legend(loc='upper left')

        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 1, figsize=(16,8), sharex=True)
        _plot_comparison(axes[0], actual_cum_net, inverse_cum_net, y_comm_lr=trade_cost_lr,
                        title='Cumulative Net Profit: Actual vs Inverse')
        _plot_comparison(axes[1], actual_cum_gross, inverse_cum_gross,
                        title='Cumulative Gross Profit: Actual vs Inverse')
        plt.tight_layout()
        plt.show()

    #+----------------------------------------------------------------------------+
    #| @func: regression analysis diagnostic                                      |
    #| @desc: regression analysis to provide information on system growth model   |
    #| @params: N/A                                                               |
    #| @return: N/A                                                               |
    #+----------------------------------------------------------------------------+
    def regression_analysis_diagnostic(self):
        actual = self.cfd_position_records.sort_values('Date Open').copy()
        inverse = self.inverse_position_records.sort_values('Date Open').copy()
        n = min(len(actual), len(inverse))
        actual = actual.iloc[:n]
        inverse = inverse.iloc[:n]
        trades = np.arange(1, n+1)
        
        def fit_multiple_regressions(trades, pnl_values):
            results = {}
            X = trades.reshape(-1, 1)
            
            # --- Linear
            lin = LinearRegression().fit(X, pnl_values)
            lin_pred = lin.predict(X)
            results['linear'] = {'prediction': lin_pred, 'r2': r2_score(pnl_values, lin_pred)}
            
            # --- Quadratic
            poly = PolynomialFeatures(degree=2)
            X_quad = poly.fit_transform(X)
            quad = LinearRegression().fit(X_quad, pnl_values)
            quad_pred = quad.predict(X_quad)
            results['quadratic'] = {'prediction': quad_pred, 'r2': r2_score(pnl_values, quad_pred)}
            
            # --- Logarithmic
            X_log = np.log(X + 1e-6)
            log_reg = LinearRegression().fit(X_log, pnl_values)
            log_pred = log_reg.predict(X_log)
            results['logarithmic'] = {'prediction': log_pred, 'r2': r2_score(pnl_values, log_pred)}
            
            # --- Exponential
            mask = pnl_values > 0
            if np.any(mask):
                X_exp = X[mask]
                y_exp = np.log(pnl_values[mask])
                exp_reg = LinearRegression().fit(X_exp, y_exp)
                exp_pred_full = np.exp(exp_reg.predict(X))
                results['exponential'] = {'prediction': exp_pred_full, 'r2': r2_score(pnl_values, exp_pred_full)}
            else:
                results['exponential'] = {'prediction': np.zeros_like(pnl_values), 'r2': -np.inf}
            
            best_model = max(results.items(), key=lambda kv: kv[1]['r2'])[0]
            return results, best_model

        actual_cum_net = actual['Net Profit'].cumsum().values
        inverse_cum_net = inverse['Net Profit'].cumsum().values
        actual_cum_gross = actual['Gross Profit'].cumsum().values
        inverse_cum_gross = inverse['Gross Profit'].cumsum().values

        # --- Fit regressions
        actual_net_res, actual_net_best       = fit_multiple_regressions(trades, actual_cum_net)
        actual_gross_res, actual_gross_best   = fit_multiple_regressions(trades, actual_cum_gross)
        inverse_net_res, inverse_net_best     = fit_multiple_regressions(trades, inverse_cum_net)
        inverse_gross_res, inverse_gross_best = fit_multiple_regressions(trades, inverse_cum_gross)
        
        def regression_growth_rate(prediction, trades):
                return (prediction[-1] - prediction[0]) / (trades[-1] - trades[0])
        
        actual_net_growth = regression_growth_rate(
            actual_net_res[actual_net_best]['prediction'], trades
        )

        inverse_net_growth = regression_growth_rate(
            inverse_net_res[inverse_net_best]['prediction'], trades
        )
        
        actual_gross_growth = regression_growth_rate(
            actual_gross_res[actual_gross_best]['prediction'], trades
        )

        inverse_gross_growth = regression_growth_rate(
            inverse_gross_res[inverse_gross_best]['prediction'], trades
        )

        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 1, figsize=(16,8), sharex=True)

        def _plot_with_regression(ax, y_actual, y_inverse, actual_res, inverse_res, actual_best, inverse_best,
                                actual_growth, inverse_growth, title, color_actual='deepskyblue', color_inverse='orange', alpha_fill=0.2):
            # --- Raw PnL
            ax.plot(trades, y_actual, lw=2.5, color=color_actual, label='Actual')
            ax.plot(trades, y_inverse, lw=2.5, linestyle='--', color=color_inverse, label='Inverse')
            
            # --- max divergence markers
            pnl_diff = y_actual - y_inverse
            max_div_idx = np.argmax(np.abs(pnl_diff))

            ax.axvline(trades[max_div_idx], color='white', linestyle=':', alpha=0.6)
            ax.scatter(trades[max_div_idx], y_actual[max_div_idx],
                    s=120, color=color_actual, edgecolors='black')
            ax.scatter(trades[max_div_idx], y_inverse[max_div_idx],
                    s=120, color=color_inverse, edgecolors='black')
            
            # --- Best-fit regression overlay
            ax.plot(trades, actual_res[actual_best]['prediction'], lw=2.5, color='lime', linestyle=':', label=f'Actual {actual_best.title()} Fit')
            ax.plot(trades, inverse_res[inverse_best]['prediction'], lw=2.5, color='red', linestyle=':', label=f'Inverse {inverse_best.title()} Fit')

            # --- Filled area under curves
            ax.fill_between(trades, y_actual, 0, where=y_actual>=0, color=color_actual, alpha=alpha_fill)
            ax.fill_between(trades, y_actual, 0, where=y_actual<0, color=color_actual, alpha=alpha_fill*0.7)
            ax.fill_between(trades, y_inverse, 0, where=y_inverse>=0, color=color_inverse, alpha=alpha_fill)
            ax.fill_between(trades, y_inverse, 0, where=y_inverse<0, color=color_inverse, alpha=alpha_fill*0.7)
            
            # --- regime labeling with minimum separation
            for y, color in zip([y_actual, y_inverse], [color_actual, color_inverse]):
                min_distance = int(0.05 * len(y))
                last_labeled_idx = -min_distance
                sign = np.sign(y)
                sign[sign == 0] = 1
                regime_start = 0
                for i in range(1, len(y)):
                    if sign[i] != sign[i-1]:
                        seg = slice(regime_start, i)
                        if sign[i-1] > 0:
                            idx = seg.start + np.argmax(y[seg])
                            va = 'bottom'
                        else:
                            idx = seg.start + np.argmin(y[seg])
                            va = 'top'
                        if idx - last_labeled_idx >= min_distance:
                            ax.plot(trades[idx], y[idx], 'o', color=color)
                            ax.text(trades[idx], y[idx],
                                    f"-${abs(y[idx]):.2f}" if y[idx] < 0 else f"${y[idx]:.2f}",
                                    color=color, fontsize=9, ha='left', va=va, fontweight='bold')
                            last_labeled_idx = idx
                        regime_start = i
                # final regime
                seg = slice(regime_start, len(y))
                if sign[-1] > 0:
                    idx = seg.start + np.argmax(y[seg])
                    va = 'bottom'
                else:
                    idx = seg.start + np.argmin(y[seg])
                    va = 'top'
                if idx - last_labeled_idx >= min_distance:
                    ax.plot(trades[idx], y[idx], 'o', color=color)
                    ax.text(trades[idx], y[idx], f"{y[idx]:.2f}", color=color,
                            fontsize=9, ha='left', va=va, fontweight='bold')

            # --- move Max ΔPnL label to bottom-left
            ax.text(
                0.01, 1.02,
                f"Max ΔPnL = {pnl_diff[max_div_idx]:.2f}",
                transform=ax.transAxes,
                fontsize=10,
                ha='left',
                va='bottom',
                fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=3)
            )
            
            growth_ratio = actual_growth / inverse_growth if inverse_growth != 0 else np.nan
            ax.text(
                0.11, 1.02,
                (
                    f" |  Actual ΔPnL / ΔTrades:  ${actual_growth:.4f}\n"
                    f" |  Inverse ΔPnL / ΔTrades: ${inverse_growth:.4f}\n"
                    f" |  Ratio (A / I): {growth_ratio:.2f}"
                ),
                transform=ax.transAxes,
                fontsize=10,
                ha='left',
                va='bottom',
                fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=4)
            )

            ax.axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.6)
            ax.set_title(title, fontsize=16)
            ax.set_xlabel('Trade Count', fontsize=12)
            ax.set_ylabel('Cumulative PnL', fontsize=12)
            ax.grid(True, linestyle=':', alpha=0.4)
            ax.legend(loc='lower left')

        # --- Plot Net
        _plot_with_regression(axes[0], actual_cum_net, inverse_cum_net,
                            actual_net_res, inverse_net_res,
                            actual_net_best, inverse_net_best,
                            actual_net_growth, inverse_net_growth,
                            title='Cumulative Net Profit: Actual vs Inverse')

        # --- Plot Gross
        _plot_with_regression(axes[1], actual_cum_gross, inverse_cum_gross,
                            actual_gross_res, inverse_gross_res,
                            actual_gross_best, inverse_gross_best,
                            actual_gross_growth, inverse_gross_growth,
                            title='Cumulative Gross Profit: Actual vs Inverse')

        plt.tight_layout()
        plt.show()
        
    
    #+----------------------------------------------------------------------------+
    #| @func: regression advantage_analysis diagnostic                            |
    #| @desc: regression analysis to provide information on system growth model   |
    #| @params: N/A                                                               |
    #| @return: N/A                                                               |
    #+----------------------------------------------------------------------------+
    def regression_advantage_analysis_diagnostic(self):
        # --- aggregate per-symbol stats
        symbol_stats = self.cfd_position_records.groupby('Symbol').agg(
            trade_count=('Net Profit', 'count'),
            lifetime_profit=('Net Profit', 'sum')
        )

        # --- compute z-score of lifetime profit
        profit_mean = symbol_stats['lifetime_profit'].mean()
        profit_std  = symbol_stats['lifetime_profit'].std()
        symbol_stats['Profit Z'] = (symbol_stats['lifetime_profit'] - profit_mean) / profit_std

        # --- negative-edge symbols (z <= threshold)
        negative_symbols = symbol_stats[symbol_stats['Profit Z'] <= -0.75].copy()
        positive_symbols = symbol_stats[symbol_stats['Profit Z'] >= 0.90].copy()

        # --- sort by negative impact
        negative_symbols.sort_values('Profit Z', inplace=True)
        positive_symbols.sort_values('Profit Z', inplace=True)
        
        # --- get list of symbols to remove
        worst_symbols = negative_symbols.index.tolist()
        best_symbols  = positive_symbols.index.tolist()
        
        # --- filter datasets
        actual = self.cfd_position_records[~self.cfd_position_records['Symbol'].isin(worst_symbols)].copy()
        inverse = self.inverse_position_records[~self.inverse_position_records['Symbol'].isin(best_symbols)].copy()
        
        n = min(len(actual), len(inverse))
        actual = actual.iloc[:n]
        inverse = inverse.iloc[:n]
        trades = np.arange(1, n+1)
        
        ##--- calculate the trade cost sum
        pre_trade_cost_cum = (actual['Commission'] + actual['Swap']).cumsum().values
        
        ##--- linear regression of trade cost
        pre_X = trades.reshape(-1, 1)
        pre_y = pre_trade_cost_cum
        reg = LinearRegression()
        reg.fit(pre_X, pre_y)
        pre_trade_cost_lr = reg.predict(pre_X)  # regression line
        pre_slope = reg.coef_[0]
        
        ##--- total trade cost drag (from regression endpoints)
        pre_total_trade_cost = pre_trade_cost_lr[-1] - pre_trade_cost_lr[0]

        ##--- per-trade edge required to neutralize costs
        raw_per_trade_offset = abs(pre_total_trade_cost) / n
        per_trade_offset = math.ceil(raw_per_trade_offset * 100) / 100  # ceiling to cents
        
        ##--- synthetic refund column (edge compensation)
        actual['Refund'] = per_trade_offset
        inverse['Refund'] = per_trade_offset
        
        # --- apply per-trade edge back into profits (solution scenario)
        actual['Gross Profit'] += per_trade_offset
        actual['Net Profit']   += per_trade_offset
        inverse['Gross Profit'] += per_trade_offset
        inverse['Net Profit']   += per_trade_offset
        
        trade_cost_cum = (
            actual['Commission']
            + actual['Swap']
            + actual['Refund']
        ).cumsum().values
        
        post_X = trades.reshape(-1, 1)
        post_y = trade_cost_cum
        reg = LinearRegression()
        reg.fit(post_X, post_y)
        trade_cost_lr = reg.predict(post_X)
        slope = reg.coef_[0]
        
        def fit_multiple_regressions(trades, pnl_values):
            results = {}
            X = trades.reshape(-1, 1)
            
            # --- Linear
            lin = LinearRegression().fit(X, pnl_values)
            lin_pred = lin.predict(X)
            results['linear'] = {'prediction': lin_pred, 'r2': r2_score(pnl_values, lin_pred)}
            
            # --- Quadratic
            poly = PolynomialFeatures(degree=2)
            X_quad = poly.fit_transform(X)
            quad = LinearRegression().fit(X_quad, pnl_values)
            quad_pred = quad.predict(X_quad)
            results['quadratic'] = {'prediction': quad_pred, 'r2': r2_score(pnl_values, quad_pred)}
            
            # --- Logarithmic
            X_log = np.log(X + 1e-6)
            log_reg = LinearRegression().fit(X_log, pnl_values)
            log_pred = log_reg.predict(X_log)
            results['logarithmic'] = {'prediction': log_pred, 'r2': r2_score(pnl_values, log_pred)}
            
            # --- Exponential
            mask = pnl_values > 0
            if np.any(mask):
                X_exp = X[mask]
                y_exp = np.log(pnl_values[mask])
                exp_reg = LinearRegression().fit(X_exp, y_exp)
                exp_pred_full = np.exp(exp_reg.predict(X))
                results['exponential'] = {'prediction': exp_pred_full, 'r2': r2_score(pnl_values, exp_pred_full)}
            else:
                results['exponential'] = {'prediction': np.zeros_like(pnl_values), 'r2': -np.inf}
            
            best_model = max(results.items(), key=lambda kv: kv[1]['r2'])[0]
            return results, best_model

        actual_cum_net = actual['Net Profit'].cumsum().values
        inverse_cum_net = inverse['Net Profit'].cumsum().values
        actual_cum_gross = actual['Gross Profit'].cumsum().values
        inverse_cum_gross = inverse['Gross Profit'].cumsum().values

        # --- Fit regressions
        actual_net_res, actual_net_best       = fit_multiple_regressions(trades, actual_cum_net)
        actual_gross_res, actual_gross_best   = fit_multiple_regressions(trades, actual_cum_gross)
        inverse_net_res, inverse_net_best     = fit_multiple_regressions(trades, inverse_cum_net)
        inverse_gross_res, inverse_gross_best = fit_multiple_regressions(trades, inverse_cum_gross)
        
        def regression_growth_rate(prediction, trades):
                return (prediction[-1] - prediction[0]) / (trades[-1] - trades[0])
        
        actual_net_growth = regression_growth_rate(
            actual_net_res[actual_net_best]['prediction'], trades
        )

        inverse_net_growth = regression_growth_rate(
            inverse_net_res[inverse_net_best]['prediction'], trades
        )
        
        actual_gross_growth = regression_growth_rate(
            actual_gross_res[actual_gross_best]['prediction'], trades
        )

        inverse_gross_growth = regression_growth_rate(
            inverse_gross_res[inverse_gross_best]['prediction'], trades
        )

        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 1, figsize=(16,8), sharex=True)

        def _plot_with_regression(ax, y_actual, y_inverse, actual_res, inverse_res, actual_best, inverse_best,
                                actual_growth, inverse_growth, title, y_comm_lr=None,  color_actual='deepskyblue', color_inverse='orange', alpha_fill=0.2):
            # --- Raw PnL
            ax.plot(trades, y_actual, lw=2.5, color=color_actual, label='Actual')
            ax.plot(trades, y_inverse, lw=2.5, linestyle='--', color=color_inverse, label='Inverse')
            
            # --- max divergence markers
            pnl_diff = y_actual - y_inverse
            max_div_idx = np.argmax(np.abs(pnl_diff))

            ax.axvline(trades[max_div_idx], color='white', linestyle=':', alpha=0.6)
            ax.scatter(trades[max_div_idx], y_actual[max_div_idx],
                    s=120, color=color_actual, edgecolors='black')
            ax.scatter(trades[max_div_idx], y_inverse[max_div_idx],
                    s=120, color=color_inverse, edgecolors='black')
            
            # --- commission regression line
            if y_comm_lr is not None:
                ax.plot(
                    trades,
                    y_comm_lr,
                    color='white',
                    lw=2.5,
                    linestyle='-',
                    alpha=0.9,
                    label='Trade Cost Axis Bias'
                )
                
            # --- label start and end of commission regression
            start_idx = 0
            end_idx = -1

            for idx, ha in zip([start_idx, end_idx], ['left', 'right']):
                value = y_comm_lr[idx]
                label = f"-${abs(value):.2f}" if value < 0 else f"${value:.2f}"

                ax.plot(
                    trades[idx],
                    value,
                    'o',
                    color='white',
                    markersize=6
                )

                ax.text(
                    trades[idx],
                    value,
                    label,
                    color='white',
                    fontsize=9,
                    ha=ha,
                    va='bottom' if value >= 0 else 'top',
                    fontweight='bold'
                )
            
            # --- Best-fit regression overlay
            ax.plot(trades, actual_res[actual_best]['prediction'], lw=2.5, color='lime', linestyle=':', label=f'Actual {actual_best.title()} Fit')
            ax.plot(trades, inverse_res[inverse_best]['prediction'], lw=2.5, color='red', linestyle=':', label=f'Inverse {inverse_best.title()} Fit')

            # --- Filled area under curves
            ax.fill_between(trades, y_actual, 0, where=y_actual>=0, color=color_actual, alpha=alpha_fill)
            ax.fill_between(trades, y_actual, 0, where=y_actual<0, color=color_actual, alpha=alpha_fill*0.7)
            ax.fill_between(trades, y_inverse, 0, where=y_inverse>=0, color=color_inverse, alpha=alpha_fill)
            ax.fill_between(trades, y_inverse, 0, where=y_inverse<0, color=color_inverse, alpha=alpha_fill*0.7)
            
            # --- regime labeling with minimum separation
            for y, color in zip([y_actual, y_inverse], [color_actual, color_inverse]):
                min_distance = int(0.05 * len(y))
                last_labeled_idx = -min_distance
                sign = np.sign(y)
                sign[sign == 0] = 1
                regime_start = 0
                for i in range(1, len(y)):
                    if sign[i] != sign[i-1]:
                        seg = slice(regime_start, i)
                        if sign[i-1] > 0:
                            idx = seg.start + np.argmax(y[seg])
                            va = 'bottom'
                        else:
                            idx = seg.start + np.argmin(y[seg])
                            va = 'top'
                        if idx - last_labeled_idx >= min_distance:
                            ax.plot(trades[idx], y[idx], 'o', color=color)
                            ax.text(trades[idx], y[idx],
                                    f"-${abs(y[idx]):.2f}" if y[idx] < 0 else f"${y[idx]:.2f}",
                                    color=color, fontsize=9, ha='left', va=va, fontweight='bold')
                            last_labeled_idx = idx
                        regime_start = i
                # final regime
                seg = slice(regime_start, len(y))
                if sign[-1] > 0:
                    idx = seg.start + np.argmax(y[seg])
                    va = 'bottom'
                else:
                    idx = seg.start + np.argmin(y[seg])
                    va = 'top'
                if idx - last_labeled_idx >= min_distance:
                    ax.plot(trades[idx], y[idx], 'o', color=color)
                    ax.text(trades[idx], y[idx], f"{y[idx]:.2f}", color=color,
                            fontsize=9, ha='left', va=va, fontweight='bold')

            # --- move Max ΔPnL label to bottom-left
            ax.text(
                0.01, 1.02,
                f"Max ΔPnL = {pnl_diff[max_div_idx]:.2f}",
                transform=ax.transAxes,
                fontsize=10,
                ha='left',
                va='bottom',
                fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=3)
            )
            
            growth_ratio = actual_growth / inverse_growth if inverse_growth != 0 else np.nan
            ax.text(
                0.11, 1.02,
                (
                    f" |  Actual ΔPnL / ΔTrades:  ${actual_growth:.4f}\n"
                    f" |  Inverse ΔPnL / ΔTrades: ${inverse_growth:.4f}\n"
                    f" |  Ratio (A / I): {growth_ratio:.2f}"
                ),
                transform=ax.transAxes,
                fontsize=10,
                ha='left',
                va='bottom',
                fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=4)
            )
            
            ax.text(
                    0.99, 1.02,
                    f"Edge Needed to Offset Trade Cost = ${per_trade_offset:.2f}",
                    transform=ax.transAxes,
                    fontsize=10,
                    ha='right',
                    va='bottom',
                    fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=3)
            )

            ax.axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.6)
            ax.set_title(title, fontsize=16)
            ax.set_xlabel('Trade Count', fontsize=12)
            ax.set_ylabel('Cumulative PnL', fontsize=12)
            ax.grid(True, linestyle=':', alpha=0.4)
            ax.legend(loc='upper left')

        # --- Plot Net
        _plot_with_regression(axes[0], actual_cum_net, inverse_cum_net,
                            actual_net_res, inverse_net_res,
                            actual_net_best, inverse_net_best,
                            actual_net_growth, inverse_net_growth,
                            title='Cumulative Net Profit: Actual vs Inverse', y_comm_lr=trade_cost_lr)

        # --- Plot Gross
        _plot_with_regression(axes[1], actual_cum_gross, inverse_cum_gross,
                            actual_gross_res, inverse_gross_res,
                            actual_gross_best, inverse_gross_best,
                            actual_gross_growth, inverse_gross_growth,
                            title='Cumulative Gross Profit: Actual vs Inverse', y_comm_lr=trade_cost_lr)

        plt.tight_layout()
        plt.show()

    #+----------------------------------------------------------------------------+
    #| @func: regression derivative diagnostic                                    |
    #| @desc: regression analysis to provide information on system growth model   |
    #| @params: N/A                                                               |
    #| @return: N/A                                                               |
    #+----------------------------------------------------------------------------+
    def regression_derivative_diagnostic(self):
        actual = self.cfd_position_records.sort_values('Date Open').copy()
        inverse = self.inverse_position_records.sort_values('Date Open').copy()
        n = min(len(actual), len(inverse))
        actual = actual.iloc[:n]
        inverse = inverse.iloc[:n]
        trades = np.arange(1, n + 1)

        def fit_multiple_regressions(trades, pnl_values):
            results = {}
            X = trades.reshape(-1, 1)

            # Linear
            lin = LinearRegression().fit(X, pnl_values)
            lin_pred = lin.predict(X)
            results['linear'] = {'prediction': lin_pred, 'r2': r2_score(pnl_values, lin_pred)}

            # Quadratic
            poly = PolynomialFeatures(degree=2)
            X_quad = poly.fit_transform(X)
            quad = LinearRegression().fit(X_quad, pnl_values)
            quad_pred = quad.predict(X_quad)
            results['quadratic'] = {'prediction': quad_pred, 'r2': r2_score(pnl_values, quad_pred)}

            # Logarithmic
            X_log = np.log(X + 1e-6)
            log_reg = LinearRegression().fit(X_log, pnl_values)
            log_pred = log_reg.predict(X_log)
            results['logarithmic'] = {'prediction': log_pred, 'r2': r2_score(pnl_values, log_pred)}

            # Exponential
            mask = pnl_values > 0
            if np.any(mask):
                X_exp = X[mask]
                y_exp = np.log(pnl_values[mask])
                exp_reg = LinearRegression().fit(X_exp, y_exp)
                exp_pred = np.exp(exp_reg.predict(X))
                results['exponential'] = {'prediction': exp_pred, 'r2': r2_score(pnl_values, exp_pred)}
            else:
                results['exponential'] = {'prediction': np.zeros_like(pnl_values), 'r2': -np.inf}

            best = max(results.items(), key=lambda kv: kv[1]['r2'])[0]
            return results, best

        # --- cumulative net only (keep it focused)
        actual_cum = actual['Net Profit'].cumsum().values
        inverse_cum = inverse['Net Profit'].cumsum().values

        actual_res, actual_best = fit_multiple_regressions(trades, actual_cum)
        inverse_res, inverse_best = fit_multiple_regressions(trades, inverse_cum)

        actual_pred = actual_res[actual_best]['prediction']
        inverse_pred = inverse_res[inverse_best]['prediction']

        # --- derivatives
        actual_d1 = np.gradient(actual_pred)
        actual_d2 = np.gradient(actual_d1)

        inverse_d1 = np.gradient(inverse_pred)
        inverse_d2 = np.gradient(inverse_d1)

        # --- periapsis-style diagnostics
        actual_peak_vel_idx = np.argmax(actual_d1)
        inverse_peak_vel_idx = np.argmax(inverse_d1)

        print("\n--- Regression Derivative Diagnostics ---")
        print(f"Actual best model:  {actual_best}")
        print(f"Inverse best model: {inverse_best}")
        print(f"Actual peak velocity trade:  {trades[actual_peak_vel_idx]}")
        print(f"Inverse peak velocity trade: {trades[inverse_peak_vel_idx]}")

        # --- plotting
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

        # --- First derivative (velocity)
        axes[0].plot(trades, actual_d1, lw=2.5, color='deepskyblue', label='Actual dPnL/dTrade')
        axes[0].plot(trades, inverse_d1, lw=2.5, linestyle='--', color='orange', label='Inverse dPnL/dTrade')

        axes[0].axvline(trades[actual_peak_vel_idx], color='deepskyblue', linestyle=':', alpha=0.6)
        axes[0].axvline(trades[inverse_peak_vel_idx], color='orange', linestyle=':', alpha=0.6)

        axes[0].axhline(0, color='white', linestyle='--', alpha=0.6)
        axes[0].set_title('First Derivative (PnL Velocity)', fontsize=15)
        axes[0].set_ylabel('ΔPnL / ΔTrades')
        axes[0].grid(True, linestyle=':', alpha=0.4)
        axes[0].legend(loc='upper right')

        # --- Second derivative (curvature)
        axes[1].plot(trades, actual_d2, lw=2.5, color='lime', label='Actual d²PnL/dTrade²')
        axes[1].plot(trades, inverse_d2, lw=2.5, linestyle='--', color='red', label='Inverse d²PnL/dTrade²')

        axes[1].axhline(0, color='white', linestyle='--', alpha=0.6)
        axes[1].set_title('Second Derivative (Edge Acceleration / Decay)', fontsize=15)
        axes[1].set_xlabel('Trade Count')
        axes[1].set_ylabel('Δ²PnL / ΔTrades²')
        axes[1].grid(True, linestyle=':', alpha=0.4)
        axes[1].legend(loc='upper right')

        plt.tight_layout()
        plt.show()

    
    def periapsis_identification_diagnostic(self):
        actual = self.cfd_position_records.sort_values('Date Open').copy()
        inverse = self.inverse_position_records.sort_values('Date Open').copy()
        n = min(len(actual), len(inverse))
        actual = actual.iloc[:n]
        inverse = inverse.iloc[:n]
        trades = np.arange(1, n + 1)

        def fit_multiple_regressions(trades, pnl_values):
            results = {}
            X = trades.reshape(-1, 1)

            # Linear
            lin = LinearRegression().fit(X, pnl_values)
            lin_pred = lin.predict(X)
            results['linear'] = {
                'prediction': lin_pred,
                'model': lin,
                'r2': r2_score(pnl_values, lin_pred)
            }

            # Quadratic
            poly = PolynomialFeatures(degree=2)
            X_quad = poly.fit_transform(X)
            quad = LinearRegression().fit(X_quad, pnl_values)
            quad_pred = quad.predict(X_quad)
            results['quadratic'] = {'prediction': quad_pred, 'model': quad, 'poly': poly,
                                    'r2': r2_score(pnl_values, quad_pred)}

            # Logarithmic
            X_log = np.log(X + 1e-6)
            log_reg = LinearRegression().fit(X_log, pnl_values)
            log_pred = log_reg.predict(X_log)
            results['logarithmic'] = {
                'prediction': log_pred,
                'model': log_reg,
                'r2': r2_score(pnl_values, log_pred)
            }

            # Exponential
            mask = pnl_values > 0
            if np.any(mask):
                X_exp = X[mask]
                y_exp = np.log(pnl_values[mask])
                exp_reg = LinearRegression().fit(X_exp, y_exp)
                exp_pred = np.exp(exp_reg.predict(X))
                results['exponential'] = {
                    'prediction': exp_pred,
                    'model': exp_reg,
                    'r2': r2_score(pnl_values, exp_pred)
                }
            else:
                results['exponential'] = {
                    'prediction': np.zeros_like(pnl_values),
                    'model': None,
                    'r2': -np.inf
                }

            ##--- return best fit regression
            best = max(results.items(), key=lambda kv: kv[1]['r2'])[0]
            return results, best
        
        def compute_periapsis(res, best, trades):
            """
            Returns the index (1-based) of periapsis if the MOST RECENT trade
            is the point of maximum velocity so far. Otherwise returns None.
            """

            x = trades.astype(float)
            i = x[-1]

            # ---------- LINEAR ----------
            if best == 'linear':
                return None  # constant velocity → no periapsis

            # ---------- QUADRATIC ----------
            elif best == 'quadratic':
                model = res['quadratic']['model']
                a = model.coef_[2]
                b = model.coef_[1]

                velocities = 2 * a * x + b

            # ---------- LOGARITHMIC ----------
            elif best == 'logarithmic':
                model = res['logarithmic']['model']
                a = model.coef_[0]

                velocities = a / x

            # ---------- EXPONENTIAL ----------
            elif best == 'exponential':
                model = res['exponential']['model']
                a = model.coef_[0]
                b = model.intercept_

                velocities = a * np.exp(a * x + b)

            else:
                return None

            # 🚫 Reject negative or zero velocity
            if velocities[-1] <= 0:
                return None
            
            # Compute velocity at current trade
            velocity_now = velocities[-1]

            # Only allow periapsis if slope is positive (moving upward)
            if velocity_now <= 0:
                return None

            # ✅ Periapsis condition: current trade is fastest so far
            if velocities[-1] == np.max(velocities):
                return int(i)

            return None
            
        ##--- perpare storage for periapsis calculations
        actual_net_periapsis_list = []
        acutal_gross_periapsis_list = []
        inverse_net_periapsis_list = []
        inverse_gross_periapsis_list = []
        
        actual['Date Open'] = pd.to_datetime(actual['Date Open'])
        inverse['Date Open'] = pd.to_datetime(inverse['Date Open'])

        ##--- get actual dataset periapses
        for i in range(2, len(actual) + 1):
            ##--- subset cumulative PnL
            actual_net_subset = actual['Net Profit'].cumsum().values[:i]
            actual_gross_subset = actual['Gross Profit'].cumsum().values[:i]
            
            trades_subset = trades[:i]

            ##--- fit regressions
            actual_net_res, actual_net_best = fit_multiple_regressions(trades_subset, actual_net_subset)
            actual_gross_res, actual_gross_best = fit_multiple_regressions(trades_subset, actual_gross_subset)
            
            ##--- determine if there is a periapsis for this iteration
            y_pred_actual_net = actual_net_res[actual_net_best]['prediction']
            y_pred_actual_gross = actual_gross_res[actual_gross_best]['prediction']
            
            ##--- positive movement requirement for actual net
            if y_pred_actual_net[-1] < y_pred_actual_net[-2]:
                # regression is moving downward → reject this periapsis
                actual_net_periapsis = None
            else:
                actual_net_periapsis = compute_periapsis(actual_net_res, actual_net_best, trades_subset)
            
            ##--- postitive movement requirement for actual gross
            if y_pred_actual_gross[-1] < y_pred_actual_gross[-2]:
                actual_gross_periapsis = None
            else:
                actual_gross_periapsis = compute_periapsis(actual_gross_res, actual_gross_best, trades_subset)
            
            ##--- add valid actual net periapsis to data
            if actual_net_periapsis is not None:
                actual_net_periapsis_list.append(actual_net_periapsis)
                
            ##--- add valid actual gross periapsis to data
            if actual_gross_periapsis is not None:
                acutal_gross_periapsis_list.append(actual_gross_periapsis)
                
        ##--- get inverse dataset periapses
        for i in range(2, len(inverse) + 1):
            ##--- subset cumulative PnL
            inverse_net_subset = inverse['Net Profit'].cumsum().values[:i]
            inverse_gross_subset = inverse['Gross Profit'].cumsum().values[:i]
            
            trades_subset = trades[:i]

            ##--- fit regressions
            inverse_net_res, inverse_net_best = fit_multiple_regressions(trades_subset, inverse_net_subset)
            inverse_gross_res, inverse_gross_best = fit_multiple_regressions(trades_subset, inverse_gross_subset)
            
            ##--- determine if there is a periapsis for this iteration
            y_pred_inverse_net = inverse_net_res[inverse_net_best]['prediction']
            y_pred_inverse_gross = inverse_gross_res[inverse_gross_best]['prediction']
            
            ##--- positive movement requirement for inverse net
            if y_pred_inverse_net[-1] < y_pred_inverse_net[-2]:
                # regression is moving downward → reject this periapsis
                inverse_net_periapsis = None
            else:
                inverse_net_periapsis = compute_periapsis(inverse_net_res, inverse_net_best, trades_subset)
            
            ##--- postitive movement requirement for inverse gross
            if y_pred_inverse_gross[-1] < y_pred_inverse_gross[-2]:
                inverse_gross_periapsis = None
            else:
                inverse_gross_periapsis = compute_periapsis(inverse_gross_res, inverse_gross_best, trades_subset)
            
            ##--- add valid inverse net periapsis to data
            if inverse_net_periapsis is not None:
                inverse_net_periapsis_list.append(inverse_net_periapsis)
                
            ##--- add valid inverse gross periapsis to data
            if inverse_gross_periapsis is not None:
                inverse_gross_periapsis_list.append(inverse_gross_periapsis)

        def apply_consecutive_day_rule(df, periapsis_list):
            filtered = []
            last_peri_date = None

            for trade_idx in periapsis_list:
                trade_date = pd.to_datetime(df.iloc[trade_idx - 1]['Date Open']).date()
                if last_peri_date is None or trade_date > last_peri_date:
                    filtered.append(trade_idx)
                    last_peri_date = trade_date
                # else: skip trades on the same day

            return filtered
        
        actual_net_periapsis_list = apply_consecutive_day_rule(actual, actual_net_periapsis_list)
        acutal_gross_periapsis_list = apply_consecutive_day_rule(actual, acutal_gross_periapsis_list)
        inverse_net_periapsis_list = apply_consecutive_day_rule(inverse, inverse_net_periapsis_list)
        inverse_gross_periapsis_list = apply_consecutive_day_rule(inverse, inverse_gross_periapsis_list)
        
        # --- Step 1: Make copies of the datasets
        actual_net_transformed = actual.copy()
        actual_gross_transformed = actual.copy()
        inverse_net_transformed = inverse.copy()
        inverse_gross_transformed = inverse.copy()

        def apply_volume_increase(df, periapsis_list, factor=1.5):
            # Sort periapses just in case
            periapsis_list = sorted(periapsis_list)
            
            for peri in periapsis_list:
                start_idx = peri  # periapsis + 1 (1-based index)
                if start_idx < len(df):
                    df.loc[start_idx:, 'Volume'] *= factor
                    df.loc[start_idx:, 'Gross Profit'] *= factor
                    df.loc[start_idx:, 'Commission'] *= factor
                    df.loc[start_idx:, 'Swap'] *= factor
                    df.loc[start_idx:, 'Net Profit'] = (
                        df.loc[start_idx:, 'Gross Profit'] +
                        df.loc[start_idx:, 'Commission'] +
                        df.loc[start_idx:, 'Swap']
                    )

        # --- Step 2: Apply transformations
        apply_volume_increase(actual_net_transformed, actual_net_periapsis_list)
        apply_volume_increase(actual_gross_transformed, acutal_gross_periapsis_list)
        apply_volume_increase(inverse_net_transformed, inverse_net_periapsis_list)
        apply_volume_increase(inverse_gross_transformed, inverse_gross_periapsis_list)

        ##--- now you have all periapsis trades stored
        print("Actual Net Periapses:", actual_net_periapsis_list)
        print("Actual Gross Periapses:", acutal_gross_periapsis_list)
        print("Inverse Net Periapses:", inverse_net_periapsis_list)
        print("Inverse Gross Periapses:", inverse_gross_periapsis_list)
        
        actual_cum_net = actual_net_transformed['Net Profit'].cumsum().values
        inverse_cum_net = inverse_net_transformed['Net Profit'].cumsum().values
        actual_cum_gross = actual_gross_transformed['Gross Profit'].cumsum().values
        inverse_cum_gross = inverse_gross_transformed['Gross Profit'].cumsum().values
        
        def _plot_comparison(ax, y_actual, y_inverse, y_comm=None, title='',
                            color_actual='deepskyblue', color_inverse='orange', alpha_fill=0.2):
            # lines
            ax.plot(trades, y_actual, lw=2.5, color=color_actual, label='Actual')
            ax.plot(trades, y_inverse, lw=2.5, linestyle='--', color=color_inverse, label='Inverse')

            # filled areas
            ax.fill_between(trades, y_actual, 0, where=y_actual>=0, color=color_actual, alpha=alpha_fill)
            ax.fill_between(trades, y_actual, 0, where=y_actual<0, color=color_actual, alpha=alpha_fill*0.7)
            ax.fill_between(trades, y_inverse, 0, where=y_inverse>=0, color=color_inverse, alpha=alpha_fill)
            ax.fill_between(trades, y_inverse, 0, where=y_inverse<0, color=color_inverse, alpha=alpha_fill*0.7)

            # max divergence markers
            pnl_diff = y_actual - y_inverse
            max_div_idx = np.argmax(np.abs(pnl_diff))
            ax.axvline(trades[max_div_idx], color='white', linestyle=':', alpha=0.6)
            ax.scatter(trades[max_div_idx], y_actual[max_div_idx], s=120, color=color_actual, edgecolors='black')
            ax.scatter(trades[max_div_idx], y_inverse[max_div_idx], s=120, color=color_inverse, edgecolors='black')

            # zero line
            ax.axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.6)

            # --- regime labeling with minimum separation
            for y, color in zip([y_actual, y_inverse], [color_actual, color_inverse]):
                min_distance = int(0.05 * len(y))
                last_labeled_idx = -min_distance
                sign = np.sign(y)
                sign[sign == 0] = 1
                regime_start = 0
                for i in range(1, len(y)):
                    if sign[i] != sign[i-1]:
                        seg = slice(regime_start, i)
                        if sign[i-1] > 0:
                            idx = seg.start + np.argmax(y[seg])
                            va = 'bottom'
                        else:
                            idx = seg.start + np.argmin(y[seg])
                            va = 'top'
                        if idx - last_labeled_idx >= min_distance:
                            ax.plot(trades[idx], y[idx], 'o', color=color)
                            ax.text(trades[idx], y[idx],
                                    f"-${abs(y[idx]):.2f}" if y[idx] < 0 else f"${y[idx]:.2f}",
                                    color=color, fontsize=9, ha='left', va=va, fontweight='bold')
                            last_labeled_idx = idx
                        regime_start = i
                # final regime
                seg = slice(regime_start, len(y))
                if sign[-1] > 0:
                    idx = seg.start + np.argmax(y[seg])
                    va = 'bottom'
                else:
                    idx = seg.start + np.argmin(y[seg])
                    va = 'top'
                if idx - last_labeled_idx >= min_distance:
                    ax.plot(trades[idx], y[idx], 'o', color=color)
                    ax.text(trades[idx], y[idx], f"{y[idx]:.2f}", color=color,
                            fontsize=9, ha='left', va=va, fontweight='bold')

            # --- move Max ΔPnL label to bottom-left
            ax.text(
                0.01, 1.02,
                f"Max ΔPnL = {pnl_diff[max_div_idx]:.2f}",
                transform=ax.transAxes,
                fontsize=10,
                ha='left',
                va='bottom',
                fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=3)
            )

            ax.set_title(title, fontsize=16)
            ax.set_xlabel('Trade Count', fontsize=12)
            ax.set_ylabel('Cumulative PnL', fontsize=12)
            ax.grid(True, linestyle=':', alpha=0.4)
            ax.legend(loc='lower left')

        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 1, figsize=(16,8), sharex=True)
        _plot_comparison(axes[0], actual_cum_net, inverse_cum_net,
                        title='Cumulative Net Profit: Actual vs Inverse')
        _plot_comparison(axes[1], actual_cum_gross, inverse_cum_gross,
                        title='Cumulative Gross Profit: Actual vs Inverse')
        plt.tight_layout()
        plt.show()
    
            
            
        


##--- execute the main method
if __name__ == '__main__':
    ##--- change directory to parent directory
    os.chdir(get_parent_directory())

    ##--- read the position data
    cfd_position_records     = pd.read_csv(r'data/complete_cfd_position_records.csv')
    inverse_position_records = pd.read_csv(r'data/inverse_cfd_position_records.csv')
    
    ##--- initialize the edge evaluator class
    inverse_analysis_modeler = Inverse_Analysis_Modeler(cfd_position_records = cfd_position_records, inverse_position_records = inverse_position_records)
    
    ##--- perform edge analysis
    # inverse_analysis_modeler.base_inverse_pnl()
    # inverse_analysis_modeler.trade_cost_diagnostic_analysis()
    # inverse_analysis_modeler.trade_cost_solution_analysis()
    # inverse_analysis_modeler.symbol_actual_incompatability_diagnostic_analysis()
    # inverse_analysis_modeler.symbol_inverse_incompatability_diagnostic_analysis()
    # inverse_analysis_modeler.symbol_comparison_incompatability_diagnostic_analysis()
    # inverse_analysis_modeler.trade_cost_symbol_analysis()
    # inverse_analysis_modeler.regression_analysis_diagnostic()
    # inverse_analysis_modeler.regression_advantage_analysis_diagnostic()
    # inverse_analysis_modeler.regression_derivative_diagnostic()
    inverse_analysis_modeler.periapsis_identification_diagnostic()