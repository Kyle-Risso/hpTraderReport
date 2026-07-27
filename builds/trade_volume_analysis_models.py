#+----------------------------------------------------------------------------+
#|                                            trade_volume_analysis_models.py |
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from adjustText import adjust_text
from matplotlib.patches import Rectangle
import matplotlib.patheffects as path_effects

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
#| @class: Calculate Trade Volume Metrics                                     |
#| @desc: calculates the total, mean, and median trade volume                 |
#| @params: N/A                                                               |
#| @return: N/A                                                               |
#+----------------------------------------------------------------------------+
class Calculate_Trade_Volume_Metrics():
    ##--- create the class initialization method
    def __init__(self):
        pass
    
    #+----------------------------------------------------------------------------+
    #| @func: Calculate Total Trade Volume                                        |
    #| @desc: finds the total trading volume in units the trader has made         |
    #| @params: account records --> dataframe with the cleaned account records    |
    #| @return: total volume --> the total amount of units the trader has traded  |
    #+----------------------------------------------------------------------------+
    def calculate_total_trade_volume(self, cfd_position_records):
        ##--- create the value columns
        cfd_position_records['Value'] = cfd_position_records['Volume'] * cfd_position_records['Contract Size'] # calculate the unit value
        total_volume = cfd_position_records['Value'].sum()                                                     # get the total sum of trade volume in units
        
        ##--- return the total volume
        total_volume = f'{total_volume:,.2f}' # format the total volume
        return total_volume

    #+----------------------------------------------------------------------------+
    #| @func: Calculate Average Trade Volume                                      |
    #| @desc: finds the trader's average trading volume in units                  |
    #| @params: account records --> dataframe with the cleaned account records    |
    #| @return: average volume --> the mean amount of units the trader has traded |
    #+----------------------------------------------------------------------------+
    def calculate_average_trade_volume(self, cfd_position_records):
        ##--- convert trades to units
        cfd_position_records['Value'] = cfd_position_records['Volume'] * cfd_position_records['Contract Size'] # calculate the unit value
        average_volume = cfd_position_records['Value'].mean()                                                  # get the mean of trade volume in units
        
        ##--- return the average volume
        average_volume = f'{average_volume:,.2f}' # format the mean volume
        return average_volume

    #+----------------------------------------------------------------------------+
    #| @func: Calculate Median Trade Volume                                       |
    #| @desc: finds the trader's median trading volume in units                   |
    #| @params: account records --> dataframe with the cleaned account records    |
    #| @return: median volume --> the median unit volume the trader has traded    |
    #+----------------------------------------------------------------------------+
    def calculate_median_trade_volume(self, cfd_position_records):
        ##--- convert trades to units
        cfd_position_records['Value'] = cfd_position_records['Volume'] * cfd_position_records['Contract Size'] # calculate the unit value
        median_volume = cfd_position_records['Value'].median()                                                 # get the median of trade volume in units
        
        ##--- return the median volume
        median_volume = f'{median_volume:,.2f}' # format the median volume
        return median_volume

class Trade_Volume_Analysis_Modeler():
    ##--- create the initialization method
    def __init__(self, cfd_position_records):
        self.cfd_position_records = cfd_position_records

    def get_trade_volume_vs_symbol(self):
        ##--- set records to local variable
        cfd_position_records = self.cfd_position_records
        
        ##--- compute unit volume per trade
        cfd_position_records['Points to Close'] = (cfd_position_records['Price Close'] - cfd_position_records['Price Open']) * cfd_position_records['Contract Size']
        cfd_position_records['Point Value']     = abs(cfd_position_records['Gross Profit'] / cfd_position_records['Points to Close'])
        cfd_position_records['Unit Volume']     = cfd_position_records['Volume'] * cfd_position_records['Contract Size']
        cfd_position_records['USD Volume']      = round(((cfd_position_records['Price Open'] * cfd_position_records['Point Value']) * cfd_position_records['Contract Size']), 2)

        ##--- lifetime profitability by symbol (full dataset)
        symbol_profit = cfd_position_records.groupby('Symbol')['Gross Profit'].sum()

        ##--- build color map
        symbol_colors = {}
        for symbol, profit in symbol_profit.items():
            if profit > 0:
                symbol_colors[symbol] = 'tab:blue'
            elif profit < 0:
                symbol_colors[symbol] = 'tab:orange'
            else:
                symbol_colors[symbol] = 'white'

        ##--- compute stats for ordering
        symbol_stats = (
            cfd_position_records
            .groupby('Symbol')['USD Volume']
            .agg(['median', 'count'])
            .reset_index()
            .sort_values('median')
        )
        symbol_order = symbol_stats['Symbol'].tolist()
        palette = {s: symbol_colors.get(s, 'grey') for s in symbol_order}

        ##--- plot
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(16, 8))

        sns.boxplot(
            x='Symbol',
            y='USD Volume',
            data=cfd_position_records,
            order=symbol_order,
            palette=palette,
            ax=ax,
            boxprops=dict(
                edgecolor='#808080',
                linewidth=1.3
            ),
            whiskerprops=dict(
                color='lightgrey',
                linewidth=1.5
            ),
            capprops=dict(
                color='lightgrey',
                linewidth=1.5
            ),
            medianprops=dict(
                color='lightgrey',
                linewidth=2
            ),
            flierprops=dict(
                marker='o',
                markerfacecolor='lightgrey',
                markeredgecolor='none',
                alpha=0.6,
                markersize=4
            )
        )

        ##--- make box edges match face color
        for patch in ax.patches:
            facecolor = patch.get_facecolor()
            patch.set_edgecolor(facecolor)
            patch.set_linewidth(1.3)

        ##--- cosmetics
        ax.set_title(
            'Symbol vs Trade Volume',
            fontsize=20
        )
        ax.set_xlabel('Symbol', fontsize=14)
        ax.set_ylabel('Trade Volume (USD)', fontsize=14)
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
        plt.xticks(rotation=45)

        ##--- save figure
        plt.tight_layout()
        plt.savefig(
            r'documentation/trade_volume_analysis/trade_volume_vs_symbol.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.show()

    def get_trade_volume_vs_commission(self):
        ##--- set records to local variable
        cfd_position_records = self.cfd_position_records
        
        ##--- compute USD volume per trade
        cfd_position_records['Points to Close'] = (cfd_position_records['Price Close'] - cfd_position_records['Price Open']) * cfd_position_records['Contract Size']
        cfd_position_records['Point Value']     = abs(cfd_position_records['Gross Profit'] / cfd_position_records['Points to Close'])
        cfd_position_records['USD Volume']      = round(((cfd_position_records['Price Open'] * cfd_position_records['Point Value']) * cfd_position_records['Contract Size']), 2)
        
        ##--- lifetime profitability by symbol (full dataset)
        symbol_profit = cfd_position_records.groupby('Symbol')['Gross Profit'].sum()
        
        ##--- build color map by lifetime profitability
        symbol_colors = {}
        for symbol, profit in symbol_profit.items():
            if profit > 0:
                symbol_colors[symbol] = 'tab:blue'
            elif profit < 0:
                symbol_colors[symbol] = 'tab:orange'
            else:
                symbol_colors[symbol] = 'white'
        
        ##--- aggregate per symbol
        symbol_stats = (
            cfd_position_records
            .groupby('Symbol')
            .agg(
                avg_usd_volume=('USD Volume', 'mean'),
                avg_commission=('Commission', 'mean')
            )
            .reset_index()
            .sort_values('avg_usd_volume')
        )
        
        ##--- assign colors
        colors = [symbol_colors.get(sym, 'grey') for sym in symbol_stats['Symbol']]
        
        ##--- plot
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(14, 8))
        
        ax.scatter(
            x=symbol_stats['avg_usd_volume'],
            y=symbol_stats['avg_commission'],
            s=120,
            c=colors,
            alpha=0.8,
            edgecolors='white'
        )
        
        ##--- annotate each point
        texts = []
        for i, row in symbol_stats.iterrows():
            texts.append(
                ax.text(
                    row['avg_usd_volume'],
                    row['avg_commission'],
                    row['Symbol'],
                    fontsize=10,
                    color='white',
                    fontweight='bold'
                )
            )
        
        ##--- adjust text to avoid overlaps
        adjust_text(
            texts,
            arrowprops=dict(arrowstyle='-', color='white', lw=0.5, shrinkA=5),
            expand_text=(1.1, 1.2)
        )
        
        ##--- cosmetics
        ax.set_title('Average Trade Volume vs Average Commission per Symbol', fontsize=18)
        ax.set_xlabel('Average Trade Volume (USD)', fontsize=14)
        ax.set_ylabel('Average Commission (USD)', fontsize=14)
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
        
        ##--- save figure
        plt.tight_layout()
        plt.savefig(
            r'documentation/trade_volume_analysis/trade_volume_vs_commission.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.show()

    def get_trade_volume_distribution(self):
        ##--- set records to local variable
        cfd_position_records = self.cfd_position_records
        
        ##--- compute USD volume per trade
        cfd_position_records['Points to Close'] = (cfd_position_records['Price Close'] - cfd_position_records['Price Open']) * cfd_position_records['Contract Size']
        cfd_position_records['Point Value']     = abs(cfd_position_records['Gross Profit'] / cfd_position_records['Points to Close'])
        cfd_position_records['USD Volume']      = round(((cfd_position_records['Price Open'] * cfd_position_records['Point Value']) * cfd_position_records['Contract Size']), 2)
        
        ##--- get mean and median trade volume
        mean_trade_volume   = cfd_position_records['USD Volume'].mean()
        median_trade_volume = cfd_position_records['USD Volume'].median()

        ##--- plot
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(14, 7))

        sns.histplot(
            cfd_position_records['USD Volume'],
            bins=20,
            kde=True,
            color='gold',
            alpha=0.6,
            ax=ax
        )

        ##--- mean & median reference lines
        ax.axvline(mean_trade_volume, color='cyan', linestyle='--', linewidth=2, label=f'Average: ${mean_trade_volume:,.2f}')
        ax.axvline(median_trade_volume, color='maroon', linestyle='--', linewidth=2, label=f'Median: ${median_trade_volume:,.2f}')

        ##--- annotate mean and median
        ax.text(mean_trade_volume, ax.get_ylim()[1]*0.9, f'${mean_trade_volume:,.2f}', color='cyan', fontsize=10, rotation=90, va='top', ha='right')
        ax.text(median_trade_volume, ax.get_ylim()[1]*0.9, f'${median_trade_volume:,.2f}', color='maroon', fontsize=10, rotation=90, va='top', ha='right')

        ax.set_title('Distribution of Trade Volume (USD)', fontsize=18)
        ax.set_xlabel('Trade Volume (USD)', fontsize=14)
        ax.set_ylabel('Number of Trades', fontsize=14)
        ax.legend()
        ax.grid(True, linestyle=':', color='grey', alpha=0.5)

        plt.tight_layout()
        plt.savefig(
            r'documentation/trade_volume_analysis/trade_volume_distribution.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.show()


##--- execute the main method
if __name__ == '__main__':
    ##--- change directory to parent directory
    os.chdir(get_parent_directory())

    ##--- read the position data
    cfd_position_records = pd.read_csv(r'data/complete_cfd_position_records.csv')
    
    ##--- perform position holding time analysis
    calculator          = Calculate_Trade_Volume_Metrics()
    total_trade_volume  = calculator.calculate_total_trade_volume(cfd_position_records = cfd_position_records)
    mean_trade_volume   = calculator.calculate_average_trade_volume(cfd_position_records = cfd_position_records)
    median_trade_volume = calculator.calculate_median_trade_volume(cfd_position_records = cfd_position_records)
    
    ##--- initialize the class
    modeler = Trade_Volume_Analysis_Modeler(cfd_position_records = cfd_position_records)
    
    ##--- get the models
    modeler.get_trade_volume_vs_symbol()
    modeler.get_trade_volume_vs_commission()
    modeler.get_trade_volume_distribution()