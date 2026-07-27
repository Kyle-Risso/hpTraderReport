#+----------------------------------------------------------------------------+
#|                                   position_holding_time_analysis_models.py |
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
#| @class: Calculate Position Holding Time Metrics                            |
#| @desc: calculates the total, mean, and median position holding time        |
#| @params: N/A                                                               |
#| @return: N/A                                                               |
#+----------------------------------------------------------------------------+
class Calculate_Position_Holding_Time_Metrics():
    ##--- create the class initialization method
    def __init__(self):
        pass
    
    #+----------------------------------------------------------------------------+
    #| @func: Calculate Total Time Held                                           |
    #| @desc: finds the total time each position was held and returns statistics  |
    #| @params: account records --> dataframe with the cleaned account records    |
    #| @return: total time held --> total time the trader was holding positions   |
    #+----------------------------------------------------------------------------+
    def calculate_total_time_held(self, cfd_position_records):
        ##--- calculate the amount of time each position was held
        cfd_position_records['Date Open'] = pd.to_datetime(cfd_position_records['Date Open'])                      # convert open date to datetime format
        cfd_position_records['Date Close'] = pd.to_datetime(cfd_position_records['Date Close'])                    # convert close date to datetime format
        cfd_position_records['Time Held'] = cfd_position_records['Date Close'] - cfd_position_records['Date Open'] # find the amount of time a position was held
        cfd_position_records['Time Held - Seconds'] = cfd_position_records['Time Held'].dt.total_seconds()         # convert the time a position was held to seconds
        
        ##--- calculate the total amount of time each position was held
        total_time_held_seconds         = int(cfd_position_records['Time Held - Seconds'].sum()) # sum each positions time held value to get total time held in seconds
        total_time_held_hours           = total_time_held_seconds // 3600                        # find the number of hours all positions were held for from total seconds
        total_time_held_hours_remainder = total_time_held_seconds % 3600                         # find the leftover seconds that couldn't fit into the total hours
        total_time_held_minutes         = total_time_held_hours_remainder // 60                  # use the leftover seconds from the hours remainder to calculate the total minutes
        remaining_time_held_seconds     = total_time_held_hours_remainder % 60                   # find the leftover seconds that couldn't fit into the total minutes to get remaining seconds
        total_time_held                 = f'{total_time_held_hours}:{total_time_held_minutes:02}:{remaining_time_held_seconds:02}'
        
        ##--- return the total time held
        return total_time_held

    #+----------------------------------------------------------------------------+
    #| @func: Calculate Average Time Held                                         |
    #| @desc: gets the average time each position was held and returns statistics |
    #| @params: account records --> dataframe with the cleaned account records    |
    #| @return: average time held --> mean time the trader held positions         |
    #+----------------------------------------------------------------------------+
    def calculate_average_time_held(self, cfd_position_records):
        ##--- calculate the amount of time each position was held
        cfd_position_records['Time Held'] = cfd_position_records['Date Close'] - cfd_position_records['Date Open']
        cfd_position_records['Time Held - Seconds'] = cfd_position_records['Time Held'].dt.total_seconds()
        
        ##--- calculate the average amount of time each position was held
        average_time_held_seconds         = int(cfd_position_records['Time Held - Seconds'].mean())
        average_time_held_hours           = average_time_held_seconds // 3600
        average_time_held_hours_remainder = average_time_held_seconds % 3600
        average_time_held_minutes         = average_time_held_hours_remainder // 60
        remaining_time_held_seconds       = average_time_held_hours_remainder % 60
        average_time_held                 = f'{average_time_held_hours}:{average_time_held_minutes:02}:{remaining_time_held_seconds:02}'
        
        ##--- return the average time held
        return average_time_held

    #+----------------------------------------------------------------------------+
    #| @func: Calculate Median Time Held                                          |
    #| @desc: gets the median time each position was held and returns statistics  |
    #| @params: account records: dataframe with the cleaned account records       |
    #| @return: median time held --> median time the trader held positions        |
    #+----------------------------------------------------------------------------+
    def calculate_median_time_held(self, cfd_position_records):
        ##--- calculate the amount of time each position was held
        cfd_position_records['Time Held'] = cfd_position_records['Date Close'] - cfd_position_records['Date Open']
        cfd_position_records['Time Held - Seconds'] = cfd_position_records['Time Held'].dt.total_seconds()
        
        ##--- get the median value of time the trader held a position
        median_time_held_seconds         = int(cfd_position_records['Time Held - Seconds'].median())
        median_time_held_hours           = median_time_held_seconds // 3600
        median_time_held_hours_remainder = median_time_held_seconds % 3600
        median_time_held_minutes         = median_time_held_hours_remainder // 60
        remaining_time_held_seconds      = median_time_held_hours_remainder % 60
        median_time_held                 = f'{median_time_held_hours}:{median_time_held_minutes:02}:{remaining_time_held_seconds:02}'
        
        ##--- return the median time held
        return median_time_held


class Position_Holding_Time_Analysis_Modeler():
    ##--- create the initialization method
    def __init__(self, cfd_position_records):
        self.cfd_position_records = cfd_position_records
    
    #+----------------------------------------------------------------------------+
    #| @func: Get Position Holding Time vs Profitability                          |
    #| @desc: measures profitability of trades by how long they were open for     |
    #| @params: account records --> dataframe with the cleaned account records    |
    #| @return: holding times vs profitability density plot                       |
    #+----------------------------------------------------------------------------+
    def get_position_holding_time_vs_profitability(self):
        ##--- set records to local variable
        cfd_position_records = self.cfd_position_records
        
        ##--- convert holding times to total hours
        cfd_position_records['Time Held'] = (cfd_position_records['Date Close'] - cfd_position_records['Date Open']).dt.total_seconds() / 3600
        
        ##--- set plot style
        plt.style.use('dark_background')          # use dark background
        fig, ax = plt.subplots(figsize = (16, 8)) # set figure size
        
        ##--- create a 2D density plot
        sns.kdeplot(
            x    = cfd_position_records['Time Held'],    # set x-axis
            y    = cfd_position_records['Gross Profit'], # set y-axis
            fill = True, cmap = "viridis", thresh = 0.05 # set density plot color specifications
        )
        
        ##--- set plot properties
        ax.set_title('Position Holding Time vs Profitability', fontsize = 21) # set chart title
        ax.set_xlabel('Time Held (hrs)', fontsize = 14)                 # set x-axis label
        ax.set_ylabel('Profit (USD)', fontsize = 14)                    # set y-axis label
        ax.axhline(0, color='white', linestyle='dashed', alpha=0.6)     # line to separate profit/loss
        ax.grid(True, linestyle = ':', color = 'grey', linewidth = 0.5) # define plot grid
        ax.set_xlim(left = 0)                                           # ensure plot doesn't show negative holding times
        
        ##--- save the model
        plt.tight_layout()
        plt.savefig(r'documentation/position_holding_time_analysis/position_holding_time_vs_profitability.png', dpi = 300, bbox_inches = 'tight')
        plt.show()

    #+----------------------------------------------------------------------------+
    #| @func: Position Holding Time Distribution                                  |
    #| @desc: visualizes distribution of holding times with mean & median markers |
    #| @params: account records --> dataframe with cleaned account records        |
    #| @return: N/A                                                               |
    #+----------------------------------------------------------------------------+
    def get_position_holding_time_distribution(self):
        ##--- set records to local variable
        cfd_position_records = self.cfd_position_records
        
        ##--- convert holding times to total hours
        cfd_position_records['Time Held'] = (cfd_position_records['Date Close'] - cfd_position_records['Date Open']).dt.total_seconds() / 3600
        mean_time_held                    = float(cfd_position_records['Time Held'].mean())
        median_time_held                  = float(cfd_position_records['Time Held'].median())
        
        ##--- plot
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(14, 7))
        
        ##--- calculate number of trades held under an hour
        under_60_min = (cfd_position_records['Time Held'] < 1.0).sum()
        total_trades = len(cfd_position_records)

        sns.histplot(
            cfd_position_records['Time Held'],
            bins=total_trades,
            kde=True,
            color='cyan',
            alpha=0.6,
            ax=ax
        )

        ax.text(
            0.02, 1.03,
            f'Trades Held < 1 hr: {under_60_min} of {total_trades}',
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom'
        )
        
        ax.text(
            mean_time_held,
            ax.get_ylim()[1] * 0.95,
            f'Average: {mean_time_held:.2f} hrs',
            color='orange',
            fontsize=10,
            ha='right',
            va='top',
            rotation=90
        )

        ax.text(
            median_time_held,
            ax.get_ylim()[1] * 0.95,
            f'Median: {median_time_held:.2f} hrs',
            color='lime',
            fontsize=10,
            ha='right',
            va='top',
            rotation=90
        )

        ##--- mean & median reference lines
        ax.axvline(mean_time_held, color='orange', linestyle='--', linewidth=2, label='Average Holding Time')
        ax.axvline(median_time_held, color='lime', linestyle='--', linewidth=2, label='Median Holding Time')

        ##--- labels
        ax.set_title('Distribution of Position Holding Times', fontsize=18)
        ax.set_xlabel('Time Held (hrs)', fontsize=14)
        ax.set_ylabel('Number of Trades', fontsize=14)
        ax.legend()

        ax.grid(True, linestyle=':', color='grey', alpha=0.5)

        plt.tight_layout()
        plt.savefig(
            r'documentation/position_holding_time_analysis/position_holding_time_distribution.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.show()

    #+----------------------------------------------------------------------------+
    #| @func: Get Position Holding Time vs Profitability by Stop Trigger          |
    #| @desc: analyzes complete trade outcomes by position holding time per trade |
    #| @params: cfd_position_records --> dataframe with cleaned account records   |
    #| @return: N/A                                                               |
    #+----------------------------------------------------------------------------+
    def get_position_holding_time_vs_profitability_by_stop_trigger(self):
        ##--- set records to local variable
        cfd_position_records = self.cfd_position_records
        
        ##--- calculate holding time in hours
        cfd_position_records['Time Held'] = (cfd_position_records['Date Close'] - cfd_position_records['Date Open']).dt.total_seconds() / 3600
        
        ##--- calculate stop level profit
        cfd_position_records['Points to S/L']   = (cfd_position_records['Price Open'] - cfd_position_records['S/L']) * cfd_position_records['Contract Size']
        cfd_position_records['Points to T/P']   = (cfd_position_records['Price Open'] - cfd_position_records['T/P']) * cfd_position_records['Contract Size']
        cfd_position_records['Points to Close'] = (cfd_position_records['Price Close'] - cfd_position_records['Price Open']) * cfd_position_records['Contract Size']
        cfd_position_records['Point Value']     = (cfd_position_records['Gross Profit'] / cfd_position_records['Points to Close'])
        cfd_position_records['Point Value']     = cfd_position_records['Point Value'].fillna(0)
        cfd_position_records['Max Loss']        = (cfd_position_records['Points to S/L'] * cfd_position_records['Point Value']) * -1
        cfd_position_records['Max Gain']        = (cfd_position_records['Points to T/P'] * cfd_position_records['Point Value']) * -1

        ##--- evaluate whether or not trade hit a stop level (vectorized)
        cfd_position_records['Triggered Stop Level'] = np.where(
            (cfd_position_records['Gross Profit'] <= cfd_position_records['Max Loss'] * 0.9) |
            (cfd_position_records['Gross Profit'] >= cfd_position_records['Max Gain'] * 0.9),
            True,
            False
        )
        
        ##--- add rows where Price Close exactly hit S/L or T/P
        exact_hit_mask = (
            (cfd_position_records['Price Close'] == cfd_position_records['S/L']) |
            (cfd_position_records['Price Close'] == cfd_position_records['T/P'])
        )
        cfd_position_records.loc[exact_hit_mask, 'Triggered Stop Level'] = True

        ##--- filter only stop-level trades
        stop_trigger_data = cfd_position_records[cfd_position_records['Triggered Stop Level'] == True].copy()
        
        ##--- drop intermediate calculation columns
        drop_cols = ['Points to S/L', 'Points to T/P', 'Points to Close', 'Point Value', 'Triggered Stop Level', 'Max Loss', 'Max Gain']
        stop_trigger_data.drop(columns=drop_cols, inplace=True)
        stop_trigger_data = stop_trigger_data[~((stop_trigger_data['S/L'] == 0) & (stop_trigger_data['T/P'] == 0))].copy()
        
        ##--- set plot style
        plt.style.use('dark_background')          # use dark background
        fig, ax = plt.subplots(figsize = (16, 8)) # set figure size
        
        ##--- create a 2D density plot
        sns.kdeplot(
            x    = stop_trigger_data['Time Held'],      # set x-axis
            y    = stop_trigger_data['Gross Profit'],   # set y-axis
            fill = True, 
            cmap = "plasma", 
            thresh = 0.05 # set density plot color specifications
        )
        
        ##--- set plot properties
        ax.set_title('Position Holding Time vs Profitability by Stop Trigger', fontsize = 21) # set chart title
        ax.set_xlabel('Time Held (hrs)', fontsize = 14)                                       # set x-axis label
        ax.set_ylabel('Profit (USD)', fontsize = 14)                                          # set y-axis label
        ax.axhline(0, color='white', linestyle='dashed', alpha=0.6)                           # line to separate profit/loss
        ax.grid(True, linestyle = ':', color = 'grey', linewidth = 0.5)                       # define plot grid
        ax.set_xlim(left = 0)                                                                 # ensure plot doesn't show negative holding times
        
        ##--- save the model
        plt.tight_layout()
        plt.savefig(r'documentation/position_holding_time_analysis/position_holding_time_vs_profitability_by_stop_trigger.png', dpi = 300, bbox_inches = 'tight')
        plt.show()
        
    #+----------------------------------------------------------------------------+
    #| @func: Get Position Holding Time vs Symbol by Stop Trigger                 |
    #| @desc: creates a model of which symbols take the longest to close          |
    #| @params: cfd_position_records --> dataframe with cleaned account records   |
    #| @return: N/A                                                               |
    #+----------------------------------------------------------------------------+
    def get_position_holding_time_vs_symbol_by_stop_trigger(self):
        ##--- set records to local variable
        cfd_position_records = self.cfd_position_records
        
        ##--- calculate holding time in hours
        cfd_position_records['Time Held'] = (cfd_position_records['Date Close'] - cfd_position_records['Date Open']).dt.total_seconds() / 3600
        
        ##--- calculate stop level profit
        cfd_position_records['Points to S/L']   = (cfd_position_records['Price Open'] - cfd_position_records['S/L']) * cfd_position_records['Contract Size']
        cfd_position_records['Points to T/P']   = (cfd_position_records['Price Open'] - cfd_position_records['T/P']) * cfd_position_records['Contract Size']
        cfd_position_records['Points to Close'] = (cfd_position_records['Price Close'] - cfd_position_records['Price Open']) * cfd_position_records['Contract Size']
        cfd_position_records['Point Value']     = (cfd_position_records['Gross Profit'] / cfd_position_records['Points to Close'])
        cfd_position_records['Max Loss']        = (cfd_position_records['Points to S/L'] * cfd_position_records['Point Value']) * -1
        cfd_position_records['Max Gain']        = (cfd_position_records['Points to T/P'] * cfd_position_records['Point Value']) * -1

        ##--- evaluate whether or not trade hit a stop level (vectorized)
        cfd_position_records['Triggered Stop Level'] = np.where(
            (cfd_position_records['Gross Profit'] <= cfd_position_records['Max Loss'] * 0.9) |
            (cfd_position_records['Gross Profit'] >= cfd_position_records['Max Gain'] * 0.9),
            True,
            False
        )
        
        ##--- add rows where Price Close exactly hit S/L or T/P
        exact_hit_mask = (
            (cfd_position_records['Price Close'] == cfd_position_records['S/L']) |
            (cfd_position_records['Price Close'] == cfd_position_records['T/P'])
        )
        cfd_position_records.loc[exact_hit_mask, 'Triggered Stop Level'] = True

        ##--- filter only stop-level trades
        stop_trigger_data = cfd_position_records[cfd_position_records['Triggered Stop Level'] == True].copy()
        
        ##--- drop intermediate calculation columns
        drop_cols = ['Points to S/L', 'Points to T/P', 'Points to Close', 'Point Value', 'Triggered Stop Level', 'Max Loss', 'Max Gain']
        stop_trigger_data.drop(columns=drop_cols, inplace=True)
        stop_trigger_data = stop_trigger_data[~((stop_trigger_data['S/L'] == 0) & (stop_trigger_data['T/P'] == 0))].copy()
        
            ##--- set plot style
        plt.style.use('dark_background')

        ##--- lifetime profitability by symbol (from FULL dataset, not filtered)
        symbol_profit = (
            cfd_position_records
            .groupby('Symbol')['Gross Profit']
            .sum()
        )

        ##--- build color map: blue = profitable, orange = losing, white = neutral
        symbol_colors = {}
        for symbol, profit in symbol_profit.items():
            if profit > 0:
                symbol_colors[symbol] = 'tab:blue'
            elif profit < 0:
                symbol_colors[symbol] = 'tab:orange'
            else:
                symbol_colors[symbol] = 'white'

        ##--- compute holding-time stats (stop-triggered only)
        symbol_stats = (
            stop_trigger_data
            .groupby('Symbol')['Time Held']
            .agg(['mean', 'median', 'std', 'count'])
            .reset_index()
            .sort_values('median')
        )

        symbol_order = symbol_stats['Symbol'].tolist()

        ##--- restrict palette to symbols present in plot
        palette = {s: symbol_colors.get(s, 'grey') for s in symbol_order}

        ##--- plot
        fig, ax = plt.subplots(figsize=(16, 8))

        sns.boxplot(
            x='Symbol',
            y='Time Held',
            data=stop_trigger_data,
            order=symbol_order,
            palette=palette,
            ax=ax,

            ##--- box styling
            boxprops=dict(
                edgecolor='#808080',   # neutral mid-grey
                linewidth=1.3
            ),

            ##--- whiskers & caps (more visible)
            whiskerprops=dict(
                color='lightgrey',
                linewidth=1.5
            ),
            capprops=dict(
                color='lightgrey',
                linewidth=1.5
            ),

            ##--- median line
            medianprops=dict(
                color='lightgrey',
                linewidth=2
            ),

            ##--- outliers
            flierprops=dict(
                marker='o',
                markerfacecolor='lightgrey',
                markeredgecolor='none',
                alpha=0.6,
                markersize=4
            )
        )
        
        ##--- make box edges match their face color (seaborn >= 0.12)
        for patch in ax.patches:
            facecolor = patch.get_facecolor()
            patch.set_edgecolor(facecolor)
            patch.set_linewidth(1.3)

        ##--- cosmetics
        ax.set_title(
            'Position Holding Time vs Symbol by Stop Trigger',
            fontsize=20
        )
        ax.set_xlabel('Symbol', fontsize=14)
        ax.set_ylabel('Time Held (Hours)', fontsize=14)
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
        plt.xticks(rotation=45)

        ##--- save
        plt.tight_layout()
        plt.savefig(
            r'documentation/position_holding_time_analysis/position_holding_time_vs_symbol_by_stop_trigger.png',
            dpi=300,
            bbox_inches='tight'
        )
        
    #+----------------------------------------------------------------------------+
    #| @func: Get Position Holding Time vs Symbol                                 |
    #| @desc: creates a model of which symbols take the longest to close overall  |
    #| @params: cfd_position_records --> dataframe with cleaned account records   |
    #| @return: N/A                                                               |
    #+----------------------------------------------------------------------------+
    def get_position_holding_time_vs_symbol(self):
        ##--- set records to local variable
        cfd_position_records = self.cfd_position_records

        ##--- calculate holding time in hours
        cfd_position_records = cfd_position_records.copy()
        cfd_position_records['Time Held'] = (
            cfd_position_records['Date Close'] - cfd_position_records['Date Open']
        ).dt.total_seconds() / 3600

        ##--- set plot style
        plt.style.use('dark_background')

        ##--- lifetime profitability by symbol (FULL dataset)
        symbol_profit = (
            cfd_position_records
            .groupby('Symbol')['Gross Profit']
            .sum()
        )

        ##--- build color map: blue = profitable, orange = losing, white = neutral
        symbol_colors = {}
        for symbol, profit in symbol_profit.items():
            if profit > 0:
                symbol_colors[symbol] = 'tab:blue'
            elif profit < 0:
                symbol_colors[symbol] = 'tab:orange'
            else:
                symbol_colors[symbol] = 'white'

        ##--- compute holding-time stats (ALL trades)
        symbol_stats = (
            cfd_position_records
            .groupby('Symbol')['Time Held']
            .agg(['mean', 'median', 'std', 'count'])
            .reset_index()
            .sort_values('median')
        )

        symbol_order = symbol_stats['Symbol'].tolist()

        ##--- restrict palette to symbols present in plot
        palette = {s: symbol_colors.get(s, 'grey') for s in symbol_order}

        ##--- plot
        fig, ax = plt.subplots(figsize=(16, 8))

        sns.boxplot(
            x='Symbol',
            y='Time Held',
            data=cfd_position_records,
            order=symbol_order,
            palette=palette,
            ax=ax,

            ##--- box styling
            boxprops=dict(
                edgecolor='#808080',
                linewidth=1.3
            ),

            ##--- whiskers & caps
            whiskerprops=dict(
                color='lightgrey',
                linewidth=1.5
            ),
            capprops=dict(
                color='lightgrey',
                linewidth=1.5
            ),

            ##--- median line
            medianprops=dict(
                color='lightgrey',
                linewidth=2
            ),

            ##--- outliers
            flierprops=dict(
                marker='o',
                markerfacecolor='lightgrey',
                markeredgecolor='none',
                alpha=0.6,
                markersize=4
            )
        )

        ##--- make box edges match their face color
        for patch in ax.patches:
            facecolor = patch.get_facecolor()
            patch.set_edgecolor(facecolor)
            patch.set_linewidth(1.3)

        ##--- cosmetics
        ax.set_title(
            'Position Holding Time vs Symbol',
            fontsize=20
        )
        ax.set_xlabel('Symbol', fontsize=14)
        ax.set_ylabel('Time Held (Hours)', fontsize=14)
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
        plt.xticks(rotation=45)

        ##--- save
        plt.tight_layout()
        plt.savefig(
            r'documentation/position_holding_time_analysis/position_holding_time_vs_symbol.png',
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
    calculator = Calculate_Position_Holding_Time_Metrics()
    total_time_held  = calculator.calculate_total_time_held(cfd_position_records = cfd_position_records)
    mean_time_held   = calculator.calculate_average_time_held(cfd_position_records = cfd_position_records)
    median_time_held = calculator.calculate_median_time_held(cfd_position_records = cfd_position_records)
    
    ##--- initialize the class
    modeler = Position_Holding_Time_Analysis_Modeler(cfd_position_records = cfd_position_records)
    
    ##--- get the models
    modeler.get_position_holding_time_vs_profitability()
    modeler.get_position_holding_time_distribution()
    modeler.get_position_holding_time_vs_profitability_by_stop_trigger()
    modeler.get_position_holding_time_vs_symbol_by_stop_trigger()
    modeler.get_position_holding_time_vs_symbol()