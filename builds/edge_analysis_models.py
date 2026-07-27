#+----------------------------------------------------------------------------+
#|                                                    edge_analysis_models.py |
#|          Copyright 2022-2025 HP Investment Trading and Gambling Strategies |
#|                                                        https://hp-fx-g.com |
#+----------------------------------------------------------------------------+

##--- import modules
import os
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
#| @class: Edge Evaluator                                                     |
#| @desc: defnies functions that determine if a trader has a market edge      |
#| @params: N/A                                                               |
#| @return: N/A                                                               |
#+----------------------------------------------------------------------------+
class Edge_Evaluator():
    ##--- create the initialization method
    def __init__(self, cfd_position_records):
        self.cfd_position_records = cfd_position_records
        
    #+----------------------------------------------------------------------------+
    #| @func: get expectancy distribution                                         |
    #| @desc: calculates the expectancy distribution and computes statistics      |
    #| @params: N/A                                                               |
    #| @return: N/A                                                               |
    #+----------------------------------------------------------------------------+
    def get_expectancy_distribution(self):
        ##--- ensure numeric and drop invalid rows
        self.cfd_position_records['Net Profit'] = pd.to_numeric(
            self.cfd_position_records['Net Profit'], errors='coerce'
        )
        returns = self.cfd_position_records['Net Profit'].dropna()

        ##--- core statistics
        mean_return   = returns.mean()
        median_return = returns.median()
        std_return    = returns.std()
        win_rate      = (returns > 0).mean()
        n_trades      = len(returns)
        
        ##--- compute histogram manually
        counts, bins = np.histogram(returns, bins=40, density=True)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])

        ##--- create colormap based on bin centers
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=bins.min(), vmax=bins.max())
        colors = cmap(norm(bin_centers))

        ##--- plot bars
        plt.style.use('dark_background')
        plt.figure(figsize=(14, 8))
        plt.bar(bin_centers, counts, width=bins[1]-bins[0], color=colors, alpha=0.8, edgecolor='black')

        # --- reference lines
        plt.axvline(mean_return,   color='tab:blue',   linewidth=2, label='Mean')
        plt.axvline(median_return, color='tab:orange', linewidth=2, label='Median')
        plt.axvline(0,             color='white',      linestyle='--', linewidth=2, label='Break-even')
        
        ##--- add stats as text on the plot
        stats_text = (
            f"{'Trades:':<16}{n_trades:>10}\n"
            f"{'Mean Return:':<15}{mean_return:>10.4f}\n"
            f"{'Median Return:':<10}{median_return:>10.4f}\n"
            f"{'Std Deviation:':<15}{std_return:>10.4f}\n"
            f"{'Win Rate:':<17}{win_rate:>10.2%}"
        )
        plt.text(
            0.01, 0.98, stats_text,
            transform=plt.gca().transAxes,
            fontsize=12,
            color='white',
            ha='left',
            va='top',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='white')
        )

        plt.title('Expectancy Distribution (Net Profit per Trade)', fontsize=20)
        plt.xlabel('Net Profit', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.4)
        plt.tight_layout()
        plt.savefig(
            r'documentation/edge_analysis/expectancy_distribution.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.show()
    
    #+----------------------------------------------------------------------------+
    #| @func: get cumulitive PnL by trade count                                   |
    #| @desc: interprets the way profits accumulate as trades are executed        |
    #| @params: N/A                                                               |
    #| @return: N/A                                                               |
    #+----------------------------------------------------------------------------+
    def get_cumulitive_pnl_by_trade_count(self):
        ##--- sort trades by Date Open to maintain execution order
        trades = self.cfd_position_records.sort_values('Date Open').copy()

        ##--- calculate cumulative PnL
        trades['Cumulative PnL'] = trades['Net Profit'].cumsum()
        
        ##--- create the figure
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(14, 8))
        
        ##--- line plot for cumulative PnL
        ax.plot(trades.index + 1, trades['Cumulative PnL'], color='lime', lw=2)
        
        ##--- scatter points colored by trade outcome
        colors = np.where(trades['Net Profit'] > 0, 'tab:blue',
                        np.where(trades['Net Profit'] < 0, 'tab:orange', 'white'))
        ax.scatter(trades.index + 1, trades['Cumulative PnL'], c=colors, s=80, alpha=0.8, edgecolors='black')
        
        ##--- highlight max drawdown and max peak
        max_peak = trades['Cumulative PnL'].max()
        max_peak_idx = trades['Cumulative PnL'].idxmax()
        max_drawdown = trades['Cumulative PnL'].min()
        max_drawdown_idx = trades['Cumulative PnL'].idxmin()
        
        ax.scatter(max_peak_idx + 1, max_peak, color='gold', s=150, marker='*', label='Max Peak')
        ax.scatter(max_drawdown_idx + 1, max_drawdown, color='red', s=150, marker='X', label='Max Drawdown')
        
        ##--- axis labels and title
        ax.set_xlabel('Trade Count', fontsize=14)
        ax.set_ylabel('Cumulative PnL', fontsize=14)
        ax.set_title('Cumulative PnL by Trade Count', fontsize=20)
        
        ##--- grid and legend
        ax.grid(True, linestyle=':', color='grey', alpha=0.5)
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', markersize=10, label='Winning Trade'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:orange', markersize=10, label='Losing Trade'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markersize=10, label='Breakeven Trade'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markersize=15, label='Max Peak'),
            Line2D([0], [0], marker='X', color='w', markerfacecolor='red', markersize=15, label='Max Drawdown')
        ]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ##--- annotate stats in top-left corner
        n_trades = len(trades)
        mean_return = trades['Net Profit'].mean()
        median_return = trades['Net Profit'].median()
        std_return = trades['Net Profit'].std()
        win_rate = (trades['Net Profit'] > 0).mean()
        
        stats_text = (
            f"{'Trades:':<16}{n_trades:>10}\n"
            f"{'Mean Return:':<15}{mean_return:>10.4f}\n"
            f"{'Median Return:':<10}{median_return:>10.4f}\n"
            f"{'Std Deviation:':<15}{std_return:>10.4f}\n"
            f"{'Win Rate:':<17}{win_rate:>10.2%}"
        )
        ax.text(1.02, 0.67, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
        
        ##--- save and show
        plt.tight_layout()
        plt.savefig(r'documentation/edge_analysis/cumulative_pnl_by_trade_count.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    #+----------------------------------------------------------------------------+
    #| @func: plot lorenz curve on profit contribution                            |
    #| @desc: evaluates the consistency of a trading system                       |
    #| @params: N/A                                                               |
    #| @return: N/A                                                               |
    #+----------------------------------------------------------------------------+
    def plot_lorenz_curve_on_profit_contribution(self):
        profits = self.cfd_position_records['Net Profit'].values
        if len(profits) == 0:
            print("No trades available.")
            return

        # Sort profits ascending by absolute value
        sorted_indices = np.argsort(np.abs(profits))
        sorted_profits = profits[sorted_indices]

        # Cumulative share of trades
        n_trades = len(sorted_profits)
        cumulative_trades = np.arange(1, n_trades + 1) / n_trades

        # Cumulative share of profit by absolute value
        total_abs_profit = np.sum(np.abs(sorted_profits))
        cumulative_profit = np.cumsum(sorted_profits) / total_abs_profit

        # Clamp cumulative profit to [0,1] for plotting
        cumulative_profit = np.clip(cumulative_profit, 0, 1)

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 8))

        # Equality line
        ax.plot([0, 1], [0, 1], color='white', linestyle='--', lw=1.5, label='Equality Line')

        # Prepare segments for coloring
        points = np.array([cumulative_trades, cumulative_profit]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Define color based on net profit sign for each segment
        segment_colors = []
        for i in range(len(sorted_profits)):
            if sorted_profits[i] > 0:
                segment_colors.append('dodgerblue')  # positive profit
            elif sorted_profits[i] < 0:
                segment_colors.append('orange')      # negative profit
            else:
                segment_colors.append('white')       # breakeven

        # Create a LineCollection with colors
        lc = LineCollection(segments, colors=segment_colors, linewidths=3)
        ax.add_collection(lc)
        proxy_line_profitable = Line2D([0], [0], color='dodgerblue', lw=3, label='Lorenz Curve (Profitable)')
        proxy_line_negative   = Line2D([0], [0], color='orange', lw=3, label='Lorenz Curve (Negative)')
        proxy_line_breakeven  = Line2D([0], [0], color='white', lw=3, label='Lorenz Curve (Breakeven)')

        # Correct Gini using absolute profits
        gini = 1 - 2 * np.trapz(cumulative_profit, cumulative_trades)

        # Annotate Gini
        ax.text(
            0.02, 0.95,
            f"Gini Coefficient: {gini:.3f}",
            transform=ax.transAxes,
            fontsize=14,
            fontweight='bold',
            color='lime',
            va='top'
        )

        ax.set_xlabel('Cumulative Share of Trades', fontsize=14)
        ax.set_ylabel('Cumulative Share of Profit', fontsize=14)
        ax.set_title('Lorenz Curve: Profit Contribution by Trade', fontsize=20)
        ax.grid(True, linestyle=':', color='grey', alpha=0.5)
        ax.legend(handles=[proxy_line_profitable, proxy_line_negative, proxy_line_breakeven] + ax.get_legend_handles_labels()[0], loc='upper right')

        # Save and show
        plt.tight_layout()
        plt.savefig(r'documentation/edge_analysis/lorenz_curve_profit_contribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    #+----------------------------------------------------------------------------+
    #| @func: bootstrap edge score distribution                                   |
    #| @desc: evaluates the stability of a trading system                         |
    #| @params: N/A                                                               |
    #| @return: N/A                                                               |
    #+----------------------------------------------------------------------------+
    def bootstrap_edge_score_distribution(self, n_bootstrap = 10000):
        # --- extract edge scores per trade
        edge_scores = self.cfd_position_records['Net Profit'].values
        if len(edge_scores) == 0:
            print("No trades available.")
            return

        # --- bootstrap resampling
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(edge_scores, size=len(edge_scores), replace=True)
            bootstrap_means.append(np.mean(sample))
        bootstrap_means = np.array(bootstrap_means)

        # --- compute statistics
        observed_mean = np.mean(edge_scores)
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)

        # --- plot
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 8))

        # --- histogram
        counts, bins, patches = ax.hist(bootstrap_means, bins=50, edgecolor='grey')

        # --- apply viridis colormap based on bin midpoint
        cmap = cm.get_cmap('viridis')
        norm = mcolors.Normalize(vmin=bins.min(), vmax=bins.max())
        for patch, left, right in zip(patches, bins[:-1], bins[1:]):
            midpoint = (left + right) / 2
            patch.set_facecolor(cmap(norm(midpoint)))

        # --- annotate observed mean and CI
        ax.axvline(observed_mean, color='lime', linestyle='--', lw=2, label=f'Observed Mean: {observed_mean:.2f}')
        ax.axvline(ci_lower, color='orange', linestyle=':', lw=2, label=f'95% CI Lower: {ci_lower:.2f}')
        ax.axvline(ci_upper, color='orange', linestyle=':', lw=2, label=f'95% CI Upper: {ci_upper:.2f}')

        # --- cosmetics
        ax.set_title('Bootstrap Distribution of Edge Score', fontsize=20)
        ax.set_xlabel('Edge Score', fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='upper right')

        # --- save & show
        plt.tight_layout()
        plt.savefig(r'documentation/edge_analysis/bootstrap_edge_score_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    #+----------------------------------------------------------------------------+
    #| @func: get real edge trades                                                |
    #| @desc: separates data into high edge contributing and baseline             |
    #| @params: N/A                                                               |
    #| @return: real_edge_trades --> trades with the highest contribution to edge |
    #            baseline_trades --> trades that don't fit into real edge trades  |
    #+----------------------------------------------------------------------------+
    def get_real_edge_trades(self):
        # --- extract edge scores per trade
        edge_scores = self.cfd_position_records['Net Profit'].values
        if len(edge_scores) == 0:
            print("No trades available.")
            return
        
        # --- trade-level distribution stats
        trade_mean = np.mean(edge_scores)
        trade_std  = np.std(edge_scores)

        # --- trade-level z-score
        self.cfd_position_records['Edge Z'] = (
            (self.cfd_position_records['Net Profit'] - trade_mean) / trade_std
        )

        # --- real positive-edge trades (2σ rule)
        real_edge_trades = self.cfd_position_records[
            self.cfd_position_records['Edge Z'] > 2
        ].copy()
        baseline_trades = self.cfd_position_records[
            self.cfd_position_records['Edge Z'] <= 2
        ].copy()

        # --- sort by edge strength
        real_edge_trades.sort_values('Edge Z', ascending=False, inplace=True)
        baseline_trades.sort_values('Edge Z', ascending=False, inplace=True)
        
        ##--- return the edge based datasets
        return real_edge_trades, baseline_trades
        
    def get_position_holding_time_vs_profitability_by_edge_case(self, real_edge_trades, baseline_trades):
        # --- convert holding times to total hours
        for df in [real_edge_trades, baseline_trades]:
            df['Date Open']  = pd.to_datetime(df['Date Open'])
            df['Date Close'] = pd.to_datetime(df['Date Close'])
            df['Time Held'] = (df['Date Close'] - df['Date Open']).dt.total_seconds() / 3600

        # --- set plot style
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 1, figsize=(16, 16), sharex=True)  # stacked plots

        # --- Real edge trades density
        sns.kdeplot(
            x    = real_edge_trades['Time Held'],
            y    = real_edge_trades['Net Profit'],
            fill = True,
            cmap = "viridis",
            thresh = 0.05,
            ax   = axes[0]
        )
        axes[0].set_title('Real Edge Trades: Holding Time vs Profitability', fontsize=18)
        axes[0].set_ylabel('Profit (USD)', fontsize=14)
        axes[0].axhline(0, color='white', linestyle='dashed', alpha=0.6)
        axes[0].grid(True, linestyle=':', color='grey', linewidth=0.5)
        axes[0].set_xlim(left=0)

        # --- Baseline trades density
        sns.kdeplot(
            x    = baseline_trades['Time Held'],
            y    = baseline_trades['Net Profit'],
            fill = True,
            cmap = "viridis",
            thresh = 0.05,
            ax   = axes[1]
        )
        axes[1].set_title('Baseline Trades: Holding Time vs Profitability', fontsize=18)
        axes[1].set_xlabel('Time Held (hrs)', fontsize=14)
        axes[1].set_ylabel('Profit (USD)', fontsize=14)
        axes[1].axhline(0, color='white', linestyle='dashed', alpha=0.6)
        axes[1].grid(True, linestyle=':', color='grey', linewidth=0.5)
        axes[1].set_xlim(left=0)

        # --- finalize
        plt.tight_layout()
        plt.savefig(r'documentation/edge_analysis/position_holding_time_vs_profitability_by_edge_case.png', dpi = 300, bbox_inches = 'tight')

    def plot_holding_time_distribution_by_edge_case(self, real_edge_trades, baseline_trades, bins=20):
        
    
        # --- convert holding times to total hours
        for df in [real_edge_trades, baseline_trades]:
            df['Date Open']  = pd.to_datetime(df['Date Open'])
            df['Date Close'] = pd.to_datetime(df['Date Close'])
            df['Time Held'] = (df['Date Close'] - df['Date Open']).dt.total_seconds() / 3600

        # --- set plot style
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # --- real edge trades histogram
        axes[0].hist(real_edge_trades['Time Held'], bins=bins, color='dodgerblue', alpha=0.8, edgecolor='white')
        axes[0].set_title('Real Edge Trades: Holding Time Distribution', fontsize=16)
        axes[0].set_ylabel('Count', fontsize=14)
        axes[0].grid(True, linestyle=':', color='grey', linewidth=0.5)

        # --- baseline trades histogram
        axes[1].hist(baseline_trades['Time Held'], bins=bins, color='orange', alpha=0.8, edgecolor='white')
        axes[1].set_title('Baseline Trades: Holding Time Distribution', fontsize=16)
        axes[1].set_xlabel('Time Held (hrs)', fontsize=14)
        axes[1].set_ylabel('Count', fontsize=14)
        axes[1].grid(True, linestyle=':', color='grey', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(r'documentation/edge_analysis/position_holding_time_distribution_by_edge_case.png', dpi = 300, bbox_inches = 'tight')
        
    def plot_category_distribution_by_edge_case(self, real_edge_trades, baseline_trades):
        # --- compute percentages for each category
        real_perc = real_edge_trades['Category'].value_counts(normalize=True) * 100
        baseline_perc = baseline_trades['Category'].value_counts(normalize=True) * 100

        # --- combine into a single DataFrame for plotting
        categories = list(set(real_perc.index).union(baseline_perc.index))
        data = {
            'Category': categories,
            'Real Edge (%)': [real_perc.get(cat, 0) for cat in categories],
            'Baseline (%)': [baseline_perc.get(cat, 0) for cat in categories]
        }
        df_plot = pd.DataFrame(data).sort_values('Real Edge (%)', ascending=False)

        # --- plot
        plt.style.use('dark_background')
        x = np.arange(len(df_plot))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, df_plot['Real Edge (%)'], width, color='dodgerblue', label='Real Edge Trades')
        ax.bar(x + width/2, df_plot['Baseline (%)'], width, color='orange', label='Baseline Trades')

        # --- cosmetics
        ax.set_xticks(x)
        ax.set_xticklabels(df_plot['Category'], rotation=45, fontsize=12)
        ax.set_ylabel('Percentage of Trades (%)', fontsize=14)
        ax.set_title('Category Distribution: Real Edge vs Baseline Trades', fontsize=18)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.grid(True, linestyle=':', color='grey', alpha=0.5, axis='y')
        ax.legend()

        plt.tight_layout()
        plt.savefig(r'documentation/edge_analysis/category_distribution_by_edge_case.png', dpi = 300, bbox_inches = 'tight')
    
    
##--- execute the main method
if __name__ == '__main__':
    ##--- change directory to parent directory
    os.chdir(get_parent_directory())

    ##--- read the position data
    cfd_position_records = pd.read_csv(r'data/complete_cfd_position_records.csv')
    
    ##--- initialize the edge evaluator class
    edge_evaluator = Edge_Evaluator(cfd_position_records = cfd_position_records)
    
    ##--- perform edge analysis
    edge_evaluator.get_expectancy_distribution()
    # evaluator.get_cumulitive_pnl_by_trade_count()
    # evaluator.plot_lorenz_curve_on_profit_contribution()
    # evaluator.bootstrap_edge_score_distribution()
    real_edge_trades, baseline_trades = edge_evaluator.get_real_edge_trades()
    # evaluator.get_position_holding_time_vs_profitability_by_edge_case(real_edge_trades = real_edge_trades, baseline_trades = baseline_trades)
    # evaluator.plot_holding_time_distribution_by_edge_case(real_edge_trades = real_edge_trades, baseline_trades = baseline_trades)
    # edge_evaluator.plot_category_distribution_by_edge_case(real_edge_trades = real_edge_trades, baseline_trades = baseline_trades)