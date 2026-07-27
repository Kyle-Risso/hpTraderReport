#+----------------------------------------------------------------------------+
#|                                            self.cfd_position_records_mutator.py |
#|          Copyright 2022-2025 HP Investment Trading and Gambling Strategies |
#|                                                        https://hp-fx-g.com |
#+----------------------------------------------------------------------------+

##--- import headers
import os
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
#| @class: Mutate CFD Position Records                                        |
#| @desc: defines a sequence of methods that can clean raw position history   |
#| @params: self.cfd_position_records --> the cleaned trade records dataframe      |
#| @return: N/A                                                               |
#+----------------------------------------------------------------------------+
class Mutate_CFD_Position_Records():
    ##--- definite a constructor method
    def __init__(self, cfd_position_records):
        self.cfd_position_records = cfd_position_records
        
    #+----------------------------------------------------------------------------+
    #| @func: remove unwanted data                                                |
    #| @desc: removes any data that jeopardizes the ability to analyze accurately |
    #| @params: N/A                                                               |
    #| @return: cfd_position_records --> position records with all the data       |
    #+----------------------------------------------------------------------------+
    def remove_unwanted_data(self):
        ##--- remove unwanted values
        self.cfd_position_records = self.cfd_position_records[~((self.cfd_position_records['Category'] == 'Forex') & (self.cfd_position_records['Volume'] > 0.10))]
        self.cfd_position_records = self.cfd_position_records[self.cfd_position_records['Category'] != 'Metals']
        self.cfd_position_records = self.cfd_position_records[
            (self.cfd_position_records['Gross Profit'] <= 7) & 
            (self.cfd_position_records['Gross Profit'] >= -2.2)
        ]
        self.cfd_position_records = self.cfd_position_records[~((self.cfd_position_records['Symbol'] == 'BTCUSD') & (self.cfd_position_records['Volume'] > 0.01))]
        
        ##--- return the cfd position records
        self.cfd_position_records.reset_index(drop = True, inplace = True)
        return self.cfd_position_records
    
    #+----------------------------------------------------------------------------+
    #| @func: fill dummy values for journal data                                  |
    #| @desc: fills in missing values for journal data with fabricated values     |
    #| @params: N/A                                                               |
    #| @return: cfd_position_records --> position records with all the data       |
    #+----------------------------------------------------------------------------+
    def fill_dummy_values_for_journal_data(self):
        ##--- fill in Outcome column
        gp = self.cfd_position_records['Gross Profit']
        conditions = [
            gp >= 0.9 * 4,
            (gp >= 2) & (gp < 0.9 * 4),
            (gp >= 0.5) & (gp < 2),
            (gp > -0.5) & (gp < 0.5),
            (gp >= -1) & (gp <= -0.5),
            (gp >= -2 * 0.9) & (gp < -1),
            gp < -2 * 0.9
        ]
        choices = [
            'Full Win', 'Partial Win', 'Small Win', 
            'Breakeven', 'Small Loss', 'Partial Loss', 'Full Loss'
        ]
        self.cfd_position_records['Outcome'] = np.select(conditions, choices, default='Unknown')
        
        ##--- fill in the Macro Strategy column
        self.cfd_position_records['Trade Date'] = pd.to_datetime(self.cfd_position_records['Date Open']).dt.date
        unique_dates = self.cfd_position_records['Trade Date'].unique()
        daily_macro_strategy = {}
        for date in unique_dates:
            # Silence FutureWarning locally for the 'any' call
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                has_macro = self.cfd_position_records[
                    (self.cfd_position_records['Trade Date'] == date) &
                    self.cfd_position_records['Macro Strategy'].notna()
                ].any(axis=None)
            if has_macro:
                value = self.cfd_position_records[
                    (self.cfd_position_records['Trade Date'] == date) &
                    self.cfd_position_records['Macro Strategy'].notna()
                ]['Macro Strategy'].iloc[0]
            else:
                value = np.random.choice(['Growth', 'Retreat'], p=[0.5, 0.5])
            daily_macro_strategy[date] = value
        self.cfd_position_records['Macro Strategy'] = self.cfd_position_records['Trade Date'].map(daily_macro_strategy)
        self.cfd_position_records.drop(columns=['Trade Date'], inplace=True)
        
        ##--- fill in the Strategy column
        def assign_strategy(row):
            if pd.notna(row['Strategy']) and row['Strategy'] != '':
                return row['Strategy']
            macro = row['Macro Strategy']
            if macro == 'Growth':
                return np.random.choice(['Bandwagon', 'Expansion'], p=[0.8, 0.2])
            elif macro == 'Retreat':
                return np.random.choice(['Mean-Reversion', 'Retracement'], p=[0.7, 0.3])
            else:
                return np.nan
        self.cfd_position_records['Strategy'] = self.cfd_position_records.apply(assign_strategy, axis=1)
        
        ##--- fill Setup column, preserving existing except Expansion overwrite
        strategy_setup_map = {
            'Bandwagon': (['Basing', 'Multi-Touch', 'Active Movement'], [0.4, 0.4, 0.2]),
            'Mean-Reversion': (['Basing', 'Multi-Touch', 'Rule of 2'], [0.25, 0.45, 0.3]),
            'Retracement': (['Basing', 'Multi-Touch'], [0.7, 0.3]),
            'Expansion': (['Condensed Bollinger Bands'], [1.0])
        }
        def assign_setup(row):
            strat = row['Strategy']
            setup = row.get('Setup', np.nan)
            if strat not in strategy_setup_map:
                return setup
            if strat == 'Expansion':
                return 'Condensed Bollinger Bands'
            if pd.isna(setup) or setup == '':
                choices, probs = strategy_setup_map[strat]
                return np.random.choice(choices, p=probs)
            return setup
        self.cfd_position_records['Setup'] = self.cfd_position_records.apply(assign_setup, axis=1)
        
        ##--- fill in MACD Proximity to 0 only where missing
        def assign_macd(row):
            if pd.notna(row['MACD Proximity to 0']):
                return row['MACD Proximity to 0']
            return np.random.choice([True, False], p=[0.3, 0.7])
        self.cfd_position_records['MACD Proximity to 0'] = self.cfd_position_records.apply(assign_macd, axis=1)
        
        ##--- fill in Star Rating only where missing
        def assign_star(row):
            if pd.notna(row['Star Rating']):
                return row['Star Rating']
            strat = row['Strategy']
            setup = row['Setup']
            stars = [3, 4, 5]
            probs = [0.75, 0.2, 0.05]
            if strat == 'Mean-Reversion' and setup == 'Multi-Touch':
                stars = [2, 3, 4, 5]
                probs = [0.15, 0.70, 0.10, 0.05]
            return np.random.choice(stars, p=probs)
        self.cfd_position_records['Star Rating'] = self.cfd_position_records.apply(assign_star, axis=1)
        
        ##--- fill in Reason for Entry only where missing
        self.cfd_position_records['Date Close'] = pd.to_datetime(
            self.cfd_position_records['Date Close']
        )
        BASE_REASONS = [
            "Breakout",
            "Strong Trend",
            "News Event",
            "Revisited Previous Level",
            "Biased Basing Pattern",
            "Volatility Spike",
            "Fakeout",
            "Liquidity Grab",
            "Broke Support/Resistance",
            "Buy/Sell Wall",
            "Chop Zone Expansion",
            "Noise"
        ]
        SPECIAL_REASONS = [
            "Spread/Witching Hour",
            "Hedge Out"
        ]
        def is_witching_hour(row):
            hour = row['Date Close'].hour
            return hour == 22
        def is_hedge_out(row, df):
            trade_date = row['Date Close'].date()
            opposite_type = 'buy' if row['Type'] == 'sell' else 'sell'
            concurrent = df[
                (df['Date Close'].dt.date == trade_date) &
                (df['Volume'] == row['Volume']) &
                (df['Type'] == opposite_type) &
                (df['Symbol'] != row['Symbol'])
            ]
            return len(concurrent) > 0
        def assign_entry(row):
            if pd.notna(row['Reason for Entry']) and row['Reason for Entry'] != '':
                return row['Reason for Entry']
            valid = set(BASE_REASONS)
            if row['Outcome'] == 'Full Win':
                valid.discard('Liquidity Grab')
                valid.discard('Revisited Previous Level')
            if is_witching_hour(row):
                valid.add("Spread/Witching Hour")
            if is_hedge_out(row, self.cfd_position_records):
                valid.add("Hedge Out")
            if not valid:
                valid = {"Noise"}
            return np.random.choice(list(valid))
        self.cfd_position_records['Reason for Entry'] = (
            self.cfd_position_records.apply(assign_entry, axis=1)
        )
        
        ##--- fill in Reason for Outcome only where missing
        self.cfd_position_records['Time Held'] = (
            pd.to_datetime(self.cfd_position_records['Date Close']) -
            pd.to_datetime(self.cfd_position_records['Date Open'])
        ).dt.total_seconds() / 3600
        ALL_REASONS = [
            "Breakout",
            "Strong Trend",
            "Hard Reversal",
            "Perfect Entry",
            "Fakeout",
            "Consolidation",
            "Buy/Sell Wall",
            "Noise",
            "Bad Entry/Bad Exit/Slippage",
            "Mistaken Entry/User Error",
            "Early Entry",
            "Late Entry",
            "Time Decay",
            "Spread-Induced Stop Out",
            "Stops Too Tight",
            "Stops Too Loose",
            "Target Miss",
            "Fear of Loss"
        ]
        def assign_outcome(row):
            # Preserve existing
            if pd.notna(row['Reason for Outcome']) and row['Reason for Outcome'] != '':
                return row['Reason for Outcome']
            # Instant close rule
            if row['Time Held'] == 0:
                return 'Spread-Induced Stop Out'
            outcome = row['Outcome']
            time_held = row['Time Held']
            valid = set(ALL_REASONS)
            # A) Strong Trend / Perfect Entry require Partial Win or better
            if outcome not in ['Partial Win', 'Full Win']:
                valid -= {'Strong Trend', 'Perfect Entry'}
            # B) Full Win restrictions
            if outcome == 'Full Win':
                valid -= {
                    'Hard Reversal', 'Fakeout', 'Consolidation',
                    'Bad Entry/Bad Exit/Slippage', 'Mistaken Entry/User Error',
                    'Early Entry', 'Late Entry', 'Time Decay',
                    'Stops Too Tight', 'Stops Too Loose',
                    'Target Miss', 'Fear of Loss'
                }
            # C) Small Loss or worse restrictions
            if outcome in ['Small Loss', 'Partial Loss', 'Full Loss']:
                valid -= {
                    'Strong Trend', 'Perfect Entry',
                    'Mistaken Entry/User Error', 'Target Miss'
                }
            # D) Stops Too Tight only on Full Loss
            if outcome != 'Full Loss':
                valid -= {'Stops Too Tight'}
            # E) Time Decay constraints
            if not (
                outcome in ['Partial Loss', 'Small Loss', 'Breakeven', 'Small Win', 'Partial Win']
                and time_held > 1
            ):
                valid -= {'Time Decay'}
            # F) Mistaken Entry only on Breakeven
            if outcome != 'Breakeven':
                valid -= {'Mistaken Entry/User Error'}
            # Safety fallback
            if len(valid) == 0:
                valid = {'Noise'}
            return np.random.choice(list(valid))
        self.cfd_position_records['Reason for Outcome'] = (
            self.cfd_position_records.apply(assign_outcome, axis=1)
        )
        self.cfd_position_records.drop(columns=['Time Held'], inplace=True)
        
        ##--- assign Left Money on the Table column
        def assign_left_money(row):
            if pd.notna(row['Left Money on the Table?']):
                return row['Left Money on the Table?']
            
            outcome = row['Outcome']
            if outcome in ['Full Win', 'Full Loss']:
                return False
            return np.random.choice([True, False], p=[0.7, 0.3])
        self.cfd_position_records['Left Money on the Table?'] = self.cfd_position_records.apply(assign_left_money, axis=1)
        
        ##--- assign Followed Trading Plan?
        self.cfd_position_records['Time Held'] = (
            pd.to_datetime(self.cfd_position_records['Date Close']) -
            pd.to_datetime(self.cfd_position_records['Date Open'])
        ).dt.total_seconds() / 3600
        def assign_followed_trading_plan(row):
            if pd.notna(row['Followed Trading Plan?']):
                return row['Followed Trading Plan?']
            time_held = row['Time Held']
            outcome = row['Outcome']
            reason  = row['Reason for Outcome']
            if outcome in ['Full Win', 'Full Loss'] and time_held < 1:
                return True
            if (
                outcome == 'Breakeven' and
                reason in ['Target Miss', 'Hard Reversal'] and
                time_held < 1
            ):
                return True
            return False
        self.cfd_position_records['Followed Trading Plan?'] = (
            self.cfd_position_records.apply(assign_followed_trading_plan, axis=1)
        )
        self.cfd_position_records.drop(columns=['Time Held'], inplace=True)
        
        ##--- assign Trading Plan Helped column missing values
        def assign_trading_plan_helped(row):
            if pd.notna(row['Trading Plan Helped?']):
                return row['Trading Plan Helped?']
            if row['Outcome'] in ['Full Win', 'Partial Win', 'Small Win', 'Breakeven']:
                return np.random.choice([True, False], p=[0.8, 0.2])
            else:
                return np.random.choice([True, False], p=[0.3, 0.7])
        self.cfd_position_records['Trading Plan Helped?'] = self.cfd_position_records.apply(assign_trading_plan_helped, axis=1)
        
        ##--- return the refined dataset
        self.cfd_position_records.reset_index(drop = True, inplace = True)
        return self.cfd_position_records

    #+----------------------------------------------------------------------------+
    #| @func: invert position data                                                |
    #| @desc: inverts the dataset to create a trade system with opposite outcome  |
    #| @params: N/A                                                               |
    #| @return: inverse_position_records --> position records with inverse data   |
    #+----------------------------------------------------------------------------+
    def invert_position_data(self):
        ##--- create a new dataset
        inverse_position_records = self.cfd_position_records.copy()

        ##--- invert the Type column
        inverse_position_records['Type'] = inverse_position_records['Type'].map({
            'buy': 'sell',
            'sell': 'buy'
        }).fillna(inverse_position_records['Type'])

        ##--- swap S/L and T/P
        inverse_position_records[['S/L', 'T/P']] = inverse_position_records[['T/P', 'S/L']]

        ##--- flip sign on Gros Profit
        inverse_position_records['Gross Profit'] = inverse_position_records['Gross Profit'] * -1

        ##--- drop and recalculate Net Profit
        if 'Net Profit' in inverse_position_records.columns:
            inverse_position_records.drop(columns=['Net Profit'], inplace=True)
        inverse_position_records['Net Profit'] = (
            inverse_position_records['Gross Profit']
            + inverse_position_records.get('Commission', 0)
            + inverse_position_records.get('Swap', 0)
        )
        
         ##--- fix floating-point precision (2 decimals)
        inverse_position_records['Net Profit'] = inverse_position_records['Net Profit'].round(2)

        ##--- Eeforce column order
        COLUMN_ORDER = [
            'Account',
            'Position',
            'Date Open',
            'Symbol',
            'Category',
            'Type',
            'Volume',
            'Contract Size',
            'Price Open',
            'S/L',
            'T/P',
            'Date Close',
            'Price Close',
            'Gross Profit',
            'Commission',
            'Swap',
            'Net Profit',
            'Outcome',
            'Macro Strategy',
            'Strategy',
            'Setup',
            'MACD Proximity to 0',
            'Star Rating',
            'Reason for Entry',
            'Reason for Outcome',
            'Left Money on the Table?',
            'Followed Trading Plan?',
            'Trading Plan Helped?'
        ]
        
        ##--- swap Outcome values
        OUTCOME_SWAP = {
            'Full Win': 'Full Loss',
            'Full Loss': 'Full Win',
            'Partial Win': 'Partial Loss',
            'Partial Loss': 'Partial Win',
            'Small Win': 'Small Loss',
            'Small Loss': 'Small Win',
            'Breakeven': 'Breakeven'
        }
        inverse_position_records['Outcome'] = inverse_position_records['Outcome'].map(OUTCOME_SWAP).fillna(inverse_position_records['Outcome'])

        ##--- keep only columns that actually exist
        ordered_cols = [c for c in COLUMN_ORDER if c in inverse_position_records.columns]
        inverse_position_records = inverse_position_records[ordered_cols]

        ##--- reset the index and return inverse datasete
        inverse_position_records.reset_index(drop=True, inplace=True)
        return inverse_position_records

##--- execute the main method
if __name__ == '__main__':
    ##--- change directory to parent directory
    os.chdir(get_parent_directory())
    
    ##--- read the position data
    cfd_position_records = pd.read_csv(r'data/complete_cfd_position_records.csv')
    
    ##--- initialize the class
    mutator = Mutate_CFD_Position_Records(cfd_position_records = cfd_position_records)
    
    ##--- remove unwated values if desired
    cfd_position_records = mutator.remove_unwanted_data()
    
    ##--- fill in missing journal data with fabricated values for testing if desired
    cfd_position_records = mutator.fill_dummy_values_for_journal_data()
    
    ##--- export and print new data
    cfd_position_records = cfd_position_records.sort_values(by = 'Date Close')
    cfd_position_records.to_csv(r'data/complete_cfd_position_records.csv', index = False)
    print(cfd_position_records, "\n")
    
    ##--- obatin the inverse dataset
    inverse_position_records = mutator.invert_position_data()
    inverse_position_records.to_csv(r'data/inverse_cfd_position_records.csv', index = False)
    print(inverse_position_records, "\n")