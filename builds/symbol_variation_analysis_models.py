#+----------------------------------------------------------------------------+
#|                                        symbol_variation_analysis_models.py |
#|          Copyright 2022-2025 HP Investment Trading and Gambling Strategies |
#|                                                        https://hp-fx-g.com |
#+----------------------------------------------------------------------------+

##--- import modules
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from adjustText import adjust_text
from matplotlib.patches import Rectangle

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


class Symbol_Variation_Analysis_Modeler():
    ##--- create the initialization method
    def __init__(self, cfd_position_records):
        self.cfd_position_records = cfd_position_records

    #+----------------------------------------------------------------------------+
    #| @func: get profitability of top symbols by trade count                     |
    #| @desc: identifies top traded symbols by position to measures profitability |
    #| @params: N/A                                                               |
    #| @return: N/A                                                               |
    #+----------------------------------------------------------------------------+
    def get_profitability_of_top_symbols_by_trade_count(self, TOP_N = 5):
        ##--- ensure data is sorted chronologically if possible
        cfd_position_records_sorted = self.cfd_position_records

        ##--- compute per-symbol cumulative PnL stats
        symbol_ranges = []

        for symbol, df in cfd_position_records_sorted.groupby('Symbol'):
            df = df.copy()
            df['cumulative_profit'] = df['Gross Profit'].cumsum()

            symbol_ranges.append({
                'Symbol': symbol,
                'trade_count': len(df),
                'open': 0,
                'close': df['cumulative_profit'].iloc[-1],
                'low': df['cumulative_profit'].min(),
                'high': df['cumulative_profit'].max()
            })

        symbol_ranges = pd.DataFrame(symbol_ranges)

        ##--- select top N by trade count
        top_symbols = (
            symbol_ranges
            .sort_values('trade_count', ascending=False)
            .head(TOP_N)
            .reset_index(drop=True)
        )

        ##--- apply plot dark theme
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 8))

        x = np.arange(len(top_symbols))
        bar_width = 0.5
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_symbols)))

        ##===========================================================
        ## WICKS: Low → High
        ##===========================================================
        for i, row in top_symbols.iterrows():
            ax.vlines(
                i,
                row['low'],
                row['high'],
                color=colors[i],
                linewidth=2,
                alpha=0.9,
                zorder=1
            )

        ##===========================================================
        ## BODIES: Open (0) → Close
        ##===========================================================
        for i, row in top_symbols.iterrows():
            open_ = row['open']
            close = row['close']

            bottom = min(open_, close)
            height = abs(close - open_)

            color = colors[i]

            rect = Rectangle(
                (i - bar_width / 2, bottom),
                bar_width,
                height if height != 0 else 0.001,
                facecolor=color,
                edgecolor=color,
                linewidth=1.2,
                alpha=1.0,     # 👈 CHANGE from 0.9 → 1.0
                zorder=3       # 👈 ADD THIS
            )
            ax.add_patch(rect)
            
            label_y = bottom + height / 2  # center of candle body
            
            ##--- inverse candle color for text
            r, g, b, a = color
            text_color = (1 - r, 1 - g, 1 - b)

            ##--- annotation inside body
            value = row['close']
            label = f"-${abs(value):,.2f} USD" if value < 0 else f"${value:,.2f} USD"
            label += f"\n({row['trade_count']} trades)"  # add trade count on second line

            txt = ax.text(
                i,
                label_y,
                label,                # pass the precomputed label here
                ha='center',
                va='center',
                fontsize=10,
                fontweight='bold',
                color=text_color,
                zorder=4
            )
            txt.set_path_effects([path_effects.withStroke(linewidth=0.75, foreground='black')])

        ##--- gridlines (both directions)
        ax.grid(
            axis='both',
            linestyle='--',
            linewidth=0.5,
            color='grey',
            alpha=0.5
        )

        ##--- labels
        ax.set_xticks(x)
        ax.set_xticklabels(top_symbols['Symbol'])

        ax.set_title(
            f'Profitability for Top {TOP_N} Most Traded Symbols',
            fontsize=14
        )
        ax.set_xlabel('Symbol', fontsize=12)
        ax.set_ylabel('Profit (USD)', fontsize=12)

        plt.tight_layout()
        plt.savefig(
            r'documentation/symbol_variation_analysis/profitability_of_symbols_by_trade_count.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.show()

    #+----------------------------------------------------------------------------+
    #| @func: get profitability of top symbols by performance                     |
    #| @desc: identifies best symbols by proftitability and showcases them        |
    #| @params: N/A                                                               |
    #| @return: N/A                                                               |
    #+----------------------------------------------------------------------------+
    def get_profitability_of_top_symbols_by_performance(self, TOP_N = 10):
        ##--- aggregate stats
        symbol_stats = (
            self.cfd_position_records
            .groupby('Symbol')
            .agg(
                trade_count=('Symbol', 'count'),
                lifetime_profit=('Gross Profit', 'sum')
            )
        )

        ##--- worst (reverse so the worst symbol is at the top)
        worst = (
            symbol_stats
            .sort_values('lifetime_profit', ascending=True)
            .head(TOP_N)
            .iloc[::-1]  # flip upside down
        )

        ##--- best (most positive last so it aligns visually)
        best = (
        symbol_stats
        .sort_values('lifetime_profit', ascending=False)
        .head(TOP_N)
    )
        best = best.iloc[::-1] # flip so most profitable is at top

        y_worst = np.arange(len(worst))
        y_best = np.arange(len(best))

        plt.style.use('dark_background')
        fig, ax_left = plt.subplots(figsize=(11, 7))
        ax_right = ax_left.twinx()

        ##--- LEFT: worst symbols (negative bars)
        ax_left.barh(
            y_worst,
            -worst['lifetime_profit'].abs(),  # flip so bars grow leftward
            color='tab:orange'
        )

        ##--- RIGHT: best symbols
        ax_right.barh(
            y_best,
            best['lifetime_profit'],
            color='tab:blue'
        )

        ##--- y-axis labels
        ax_left.set_yticks(y_worst)
        ax_left.set_yticklabels(worst.index)

        ax_right.set_yticks(y_best)
        ax_right.set_yticklabels(best.index)

        ##--- center line
        ax_left.axvline(0, color='grey', linewidth=1)

        ##--- grid
        ax_left.grid(
            linestyle='--',
            linewidth=0.5,
            color='grey',
            alpha=0.5
        )

        ##--- annotations (labels inside bars)
        for i, row in enumerate(worst.itertuples()):
            value = row.lifetime_profit
            label = f"-${abs(value):,.2f} USD" if value < 0 else f"${value:,.2f} USD"
            label += f"\n({row.trade_count} trades)"

            ax_left.text(
                -abs(row.lifetime_profit) + abs(row.lifetime_profit)*0.02,  # small offset from bar start
                i,
                label,
                va='center',
                ha='left',   # inside the bar
                fontsize=9,
                fontweight='bold',
                color='white'
            )

        for i, row in enumerate(best.itertuples()):
            value = row.lifetime_profit
            label = f"-${abs(value):,.2f} USD" if value < 0 else f"${value:,.2f} USD"
            label += f"\n({row.trade_count} trades)"

            ax_right.text(
                row.lifetime_profit - row.lifetime_profit*0.02,  # small offset from bar end
                i,
                label,
                va='center',
                ha='right',  # inside the bar
                fontsize=9,
                fontweight='bold',
                color='white'
            )

        ##--- cosmetics
        ax_left.set_title(f'Profitability for Top {TOP_N} Best and Worst Performing Symbols')
        ax_left.set_xlabel('Lifetime Profit (USD)')

        ax_left.spines['right'].set_visible(False)
        ax_right.spines['left'].set_visible(False)

        plt.tight_layout()
        plt.savefig(r'documentation/symbol_variation_analysis/profitability_of_symbols_by_performance.png', dpi = 300, bbox_inches = 'tight')
        plt.show()
        
    #+----------------------------------------------------------------------------+
    #| @func: get undervalued symbols                                             |
    #| @desc: identifies undervalued symbols by profitability and usage           |
    #| @params: N/A                                                               |
    #| @return: N/A                                                               |
    #+----------------------------------------------------------------------------+
    def get_undervalued_symbols(self, TOP_N = 8):
        ##--- aggregate stats
        symbol_summary = (
            self.cfd_position_records
            .groupby('Symbol')
            .agg(
                trade_count=('Symbol', 'count'),
                lifetime_profit=('Gross Profit', 'sum')
            )
        )

        ##--- calculate performance metric: profit per trade
        symbol_summary['profit_per_trade'] = symbol_summary['lifetime_profit'] / symbol_summary['trade_count']

        ##--- determine undervalued: high profit per trade with lower trade count
        symbol_summary['undervaluation_score'] = symbol_summary['profit_per_trade'] / np.sqrt(symbol_summary['trade_count'])

        ##--- get top N undervalued symbols
        undervalued_symbols = symbol_summary.sort_values('undervaluation_score', ascending=False).head(TOP_N)

        ##--- apply plot dark theme
        plt.style.use('dark_background')
        plt.figure(figsize=(10, 7))

        ##--- scatter all symbols
        plt.scatter(
            symbol_summary['trade_count'],
            symbol_summary['lifetime_profit'],
            s=80,
            color='grey',
            alpha=0.6,
            label='All Symbols'
        )

        ##--- highlight undervalued symbols
        plt.scatter(
            undervalued_symbols['trade_count'],
            undervalued_symbols['lifetime_profit'],
            s=120,
            color='lime',
            edgecolor='white',
            label='Undervalued Symbols'
        )

        ##--- annotate undervalued symbols using adjustText
        texts = []
        for symbol, row in undervalued_symbols.iterrows():
            texts.append(
                plt.text(
                    row['trade_count'],
                    row['lifetime_profit'],
                    symbol,
                    fontsize=10,
                    color='white',
                    fontweight='bold'
                )
            )

        ##--- automatically adjust labels to prevent overlaps
        adjust_text(
            texts,
            arrowprops=dict(arrowstyle='-', color='white', lw=0.5, shrinkA=5),
            expand_text=(1.1, 1.2)
        )

        ##--- reference lines
        plt.axhline(0, color='grey', linewidth=1)
        plt.axvline(0, color='grey', linewidth=1)

        ##--- labels and title
        plt.xlabel('Trade Count')
        plt.ylabel('Lifetime Profit (USD)')
        plt.title(f'Top {TOP_N} Most Undervalued Symbols')
        plt.legend()
        plt.grid(axis='both', linestyle='--', linewidth=0.5, color='grey', alpha=0.5)
        plt.tight_layout()
        plt.savefig(r'documentation/symbol_variation_analysis/undervalued_symbols.png', dpi = 300, bbox_inches = 'tight')
        plt.show()


##--- execute the main method
if __name__ == '__main__':
    ##--- change directory to parent directory
    os.chdir(get_parent_directory())

    ##--- cfd_position_records = pd.read_csv(r'data/complete_cfd_position_records.csv')
    cfd_position_records = pd.read_csv(r'data/complete_cfd_position_records.csv')
    
    ##--- initialize the class
    modeler = Symbol_Variation_Analysis_Modeler(cfd_position_records = cfd_position_records)
    
    ##--- measure profitability of the most used symbols
    modeler.get_profitability_of_top_symbols_by_trade_count()
    modeler.get_profitability_of_top_symbols_by_performance()
    modeler.get_undervalued_symbols()