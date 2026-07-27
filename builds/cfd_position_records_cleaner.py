#+----------------------------------------------------------------------------+
#|                                            cfd_position_records_cleaner.py |
#|          Copyright 2022-2025 HP Investment Trading and Gambling Strategies |
#|                                                        https://hp-fx-g.com |
#+----------------------------------------------------------------------------+

##--- import headers
import os
import pandas as pd
import numpy as np
import seaborn as sns
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
#| @class: Clean CFD Position Records                                         |
#| @desc: defines a sequence of methods that can clean raw position history   |
#| @params: N/A                                                               |
#| @return: N/A                                                               |
#+----------------------------------------------------------------------------+
class Clean_CFD_Position_Records():
    ##--- definite a constructor method
    def __init__(self):
        pass

    #+----------------------------------------------------------------------------+
    #| @func: Read All Account Records                                            |
    #| @desc: reads all forex records for an account and combines as a DataFrame  |
    #| @params: account records folder --> the directory containing the records   |
    #| @return: raw account records --> data table containing all records         |
    #+----------------------------------------------------------------------------+
    def read_trade_history_dataframes(self, trade_history_folder):
        ##--- define a list to hold all account history files
        trade_history_df_list = []
        
        ##--- loop through each file in the account positions history folder
        for filename in os.listdir(trade_history_folder):
            ##--- verify the record is in CSV format and UTF-16 encoding
            if filename.endswith('.csv'):
                account_history_file = os.path.join(trade_history_folder, filename)                     # identify each account positions history file
                account_history_df = pd.read_csv(account_history_file, sep = '\t', encoding = 'utf-16') # read the account history files as a dataframe
                trade_history_df_list.append(account_history_df)                                        # append the dataframe to the list
        
        ##--- concatenate all dataframes in the list into a single dataframe
        if trade_history_df_list:
            account_records = pd.concat(trade_history_df_list, ignore_index = True)
        else:
            account_records = pd.DataFrame() # return an empty DataFrame if no CSV files are found
        
        ##--- return the cocatenated account records dataframe
        return account_records

    #+----------------------------------------------------------------------------+
    #| @func: Clean Symbols Data                                                  |
    #| @desc: cleans the symbol data dataframe for further use                    |
    #| @params: symbol data folder --> the directory containing all symbol data   |
    #| @return: symbol data --> data table containing all symbol data             |
    #+----------------------------------------------------------------------------+
    def clean_symbols_data(self, symbols_data_folder):
        ##--- define data structures
        account_symbols_data_list = [] # list to hold all dataframes
        
        ##--- loop through each record in the symbols data folder
        for filename in os.listdir(symbols_data_folder):
            ##--- verify the record is in CSV format
            if filename.endswith('.csv'):
                account_symbols_data    = os.path.join(symbols_data_folder, filename)                        # identify each symbol data history file
                account_symbols_data_df = pd.read_csv(account_symbols_data, sep = '\t', encoding = 'utf-16') # read the csv record as a DataFrame
                account_symbols_data_list.append(account_symbols_data_df)                                    # append the dataframe to the list
        
        ##--- concatenate all dataframes in the list into a single dataframe
        if account_symbols_data_list:
            symbols_data = pd.concat(account_symbols_data_list, ignore_index = True)
        else:
            symbols_data = pd.DataFrame() # return an empty DataFrame if no CSV files are found
        
        ##--- convert numeric columns to numeric types
        symbols_data['Contract Size'] = symbols_data['Contract Size'].astype(float)
        symbols_data['Decimal Value'] = symbols_data['Decimal Value'].astype(float)
        symbols_data['Spread'] = symbols_data['Spread'].astype(int) 
        
        ##--- recategorize symbols
        symbols_data.loc[symbols_data['Category'] == 'Currency', 'Category'] = 'Forex'                                                                           # change symbol category "currency" to "forex"
        symbols_data.loc[symbols_data['Category'] == 'Indexes', 'Category'] = 'Indices'                                                                          # change the symbol category name "Indexes" to "Indices"
        symbols_data.loc[symbols_data['Description'].str.contains('Index', case = False), 'Category'] = 'Indices'                                                # symbols containing the word "Index" are "Indices"
        symbols_data.loc[symbols_data['Description'].str.contains('Future', case = False), 'Category'] = 'Futures'                                               # symbols containing the word "Future" as "Futures"
        symbols_data.loc[symbols_data['Description'].str.contains('vs', case = False) & (symbols_data['Contract Size'] == 100000), 'Category'] = 'Forex'         # make sure all forex pairs have a contract size of 100000 units
        symbols_data.loc[(symbols_data['Symbol'].str.len() < 6) & symbols_data['Symbol'].str.contains(r'\d'), 'Category'] = 'Indices'                            # symbols less than 6 characters and contain specifc character matches are "Indices"
        symbols_data.loc[symbols_data['Symbol'].str.startswith('X') & (symbols_data['Category'] == 'Commodities'), 'Category'] = 'Metals'                        # symbols that start with the letter "X" and listed as commodities are "Metals"
        symbols_data.loc[symbols_data['Description'].str.contains('commodity', case = False), 'Category'] = 'Commodities'                                        # classify symbols that contain the word "commodity" and "Commodities"
        symbols_data.loc[symbols_data['Symbol'].str.contains(r'\.'), 'Category'] = 'Stocks'                                                                      # classify symbols with periods as "Stocks"
        symbols_data.loc[symbols_data['Description'].str.contains('oil', case = False), 'Category'] = 'Oils'                                                     # define the "Oils" asset class
        symbols_data.loc[symbols_data['Category'] == 'Crypto Currency', 'Category'] = 'Cryptocurrency'                                                           # combine category 2 words "Crypto Curreny" into one word
        symbols_data.loc[symbols_data['Description'].str.contains('Coin', case = False), 'Category'] = 'Cryptocurrency'                                          # classify all symbols containing the word "coin" as category "Cryptocurrency"
        symbols_data.loc[symbols_data['Description'].str.contains('vs', case = False) & (symbols_data['Contract Size'] < 100000), 'Category'] = 'Cryptocurrency' # symbols that contain "vs" in the description but less than 100000 in contract size are "Cryptocurrency"
        symbols_data.loc[symbols_data['Symbol'].str.startswith('X'), 'Category'] = 'Metals'                                                                      # symbols that start with the letter "X" are "Metals"
        symbols_data.loc[symbols_data['Symbol'] == 'LNKUSD', 'Category'] = 'Cryptocurrency'                                                                      # LNKUSD symbol is always "Cryptocurrency"
        symbols_data.loc[symbols_data['Symbol'] == 'NATGAS', 'Category'] = 'Commodities'                                                                         # NATGAS symbol is always "Commodities"
        symbols_data.loc[symbols_data['Symbol'] == 'HKG.33', 'Category'] = 'Indices'                                                                             # HKG.33 symbol is always "Indices"
        symbols_data.loc[symbols_data['Symbol'] == 'UK.100', 'Category'] = 'Indices'                                                                             # UK.100 symbol is always "Indices"
        
        ##--- remove unnecessary features
        symbols_data['Description'] = symbols_data['Description'].str.replace(r' \(1CFD = 1.*?\)', '', regex = True) # remove characters from this symbol descriptions |
        symbols_data['Description'] = symbols_data['Description'].str.replace(r' - Spot', '', regex = True)          # <-----------------------------------------------|
        
        ##--- return the cleaned symbols data
        symbols_data = symbols_data.sort_values(by = 'Symbol') # sort the symbols data by "Symbol"
        symbols_data.reset_index(drop = True, inplace = True)  # reset the symbols data index
        return symbols_data                                    # return the symbols data
    
    #+----------------------------------------------------------------------------+
    #| @func: Clean Account Record                                                |
    #| @desc: cleans account records for better statistical handling              |
    #| @params: raw account records --> data containing all the account records   |
    #|          symbols data --> data containing the information on each symbol   |
    #| @return: account records: the cleaned dataframe containing all records     |
    #+----------------------------------------------------------------------------+
    def clean_account_records(self, account_records, symbols_data):
        ##--- initialize the dataframes
        position_records = pd.DataFrame(account_records)
        
        ##--- change column names where necessary
        position_records['Position'] = position_records['Ticket'] # change ticket column to position
        
        ##--- handle None values
        position_records['S/L'] = position_records['S/L'].where(pd.notnull(position_records['S/L']), None) # replace None stop values with close prices |
        position_records['T/P'] = position_records['T/P'].where(pd.notnull(position_records['T/P']), None) # <------------------------------------------|
        
        ##--- convert date columns to epoch time
        position_records['Date Open']  = pd.to_datetime(position_records['Date Open'], format = 'ISO8601')    # convert open and close dates to iso timestamps |
        position_records['Date Open']  = mdates.date2num(position_records['Date Open'])                       # convert further to epoch timestamps |          |
        position_records['Date Close'] = pd.to_datetime(position_records['Date Close'], format = 'ISO8601')   # <-----------------------------------|----------|
        position_records['Date Close'] = mdates.date2num(position_records['Date Close'])                      # <-----------------------------------|
        
        ##--- correct improper type formatting and manage decimal places in numeric based columns
        position_records['Contract Size'] = position_records['Contract Size'].astype(float).round(2) # set gross contract size column to 2 decimal float
        position_records['Decimals']      = position_records['Decimals'].astype(int)                 # set decimals column to integer
        position_records['Volume']        = position_records['Volume'].astype(float).round(2)        # set volume column to 2 decimal float
        
        ##--- loop through the data and ensure proper float and decimal formatting for numeric columns
        for index, row in position_records.iterrows():
            decimals = row['Decimals']
            position_records.at[index, 'S/L'] = round(float(row['S/L']), decimals)                   # set the stops columbs to floats with the correct decimals |
            position_records.at[index, 'T/P'] = round(float(row['T/P']), decimals)                   # <---------------------------------------------------------|
            position_records.at[index, 'Price Open'] = round(float(row['Price Open']), decimals)     # set the price columns to floats with the correct decimals |
            position_records.at[index, 'Price Close'] = round(float(row['Price Close']), decimals)   # <---------------------------------------------------------|
        
        ##--- convert all necessary columns to 2 decimal floats
        position_records['Gross Profit'] = position_records['Gross Profit'].astype(float).round(2) # set gross profit column to 2 decimal float
        position_records['Commission']   = position_records['Commission'].astype(float).round(2)   # set commission column to 2 decimal float
        position_records['Swap']         = position_records['Swap'].astype(float).round(2)         # set swap column to 2 decimal float
        position_records['Net Profit']   = position_records['Net Profit'].astype(float).round(2)   # set net profit column to 2 decimal float
        
        ##--- correct possible null values in the price columns
        position_records['Price Open']  = position_records.apply(                                                                                                     # correct null values from "Price Open" |
                                        lambda row: row['Price Open'] if pd.notnull(row['Price Open']) else account_records.loc[row.name, 'Price Open'], axis = 1)    # <-------------------------------------|
        position_records['Price Close'] = position_records.apply(                                                                                                     # correct null values from "Price Close" |
                                        lambda row: row['Price Close'] if pd.notnull(row['Price Close']) else account_records.loc[row.name, 'Price Close'], axis = 1) # <--------------------------------------|
        
        ##--- map symbol info to the records data
        symbol_categories                 = dict(zip(symbols_data['Symbol'], symbols_data['Category']))      # convert symbol type data to dictionary
        symbol_contract_sizes             = dict(zip(symbols_data['Symbol'], symbols_data['Contract Size'])) # convert symbol contract size data to dictionary
        position_records                  = position_records.replace('HKG.33', 'HK50')                       # convert all odd symbols matching names |
        position_records                  = position_records.replace('UK.100', 'UK100')                      # <--------------------------------------|
        position_records['Category']      = position_records['Symbol'].map(symbol_categories)                # map symbol type and contract sizes to account records |
        position_records['Contract Size'] = position_records['Symbol'].map(symbol_contract_sizes)            # <-----------------------------------------------------|
        position_records                  = position_records.sort_values(by = ['Date Close', 'Position'])    # order the dataframe by close date and position number
        position_records['Date Open']     = mdates.num2date(position_records['Date Open'])                   # change date open back to iso timestamp
        position_records['Date Close']    = mdates.num2date(position_records['Date Close'])                  # change date close back to iso timestamp
        
        ##--- return the cleaned records
        position_records = position_records.reindex(columns = ['Account', 'Position', 'Date Open', 'Symbol', 'Category', 'Type', 'Volume', 'Contract Size', 'Price Open', \
                                                               'S/L', 'T/P', 'Date Close', 'Price Close', 'Gross Profit', 'Commission', 'Swap', 'Net Profit']) # choose column order
        position_records = position_records.reset_index(drop = True)                                                                                           # reset the index
        position_records = position_records.sort_values(by = 'Date Close')                                                                                     # sort values
        return position_records                                                                                                                                # return cleaned records
    
    #+----------------------------------------------------------------------------+
    #| @func: Add Qualitative Data                                                |
    #| @desc: reads qualitative trade journaling data from a folder and joins it  |
    #| @params: position_records --> DataFrame of cleaned account records         |
    #|   qualitative_data_folder --> directory containing the journal data        |
    #| @return: complete_position_records --> position records with all the data  |
    #+----------------------------------------------------------------------------+
    def add_qualitative_data(self, position_records, qualitative_data_folder):
        ##--- make sure we have a list
        if isinstance(qualitative_data_folder, str):
            qualitative_data_folder = [qualitative_data_folder]

        ##--- list to store all qualitative dataframes
        qualitative_df_list = []

        ##--- loop through each folder
        for folder in qualitative_data_folder:
            ##--- check if files exist in a folder
            if not os.path.exists(folder):
                continue # skip invalid folders
            #--- loop through CSV files in folder
            for filename in os.listdir(folder):
                ##--- identified a CSV file in directory
                if filename.endswith('.csv'):
                    file_path = os.path.join(folder, filename)      # read file path
                    df = pd.read_csv(file_path, encoding = 'utf-8') # read CSV
                    qualitative_df_list.append(df)                  # add CSV to list

        ##--- concatenate all CSVs
        if qualitative_df_list:
            qualitative_data = pd.concat(qualitative_df_list, ignore_index = True)
        else:
            qualitative_data = pd.DataFrame(columns = ['Account', 'Position']) # ensure columns exist

        ##--- rename 'Ticket' column to 'Position'
        if 'Ticket' in qualitative_data.columns:
            qualitative_data.rename(columns = {'Ticket': 'Position'}, inplace = True)
            
        #--- drop the Timestamp column if it exists
        if 'Timestamp' in qualitative_data.columns:
            qualitative_data.drop(columns=['Timestamp'], inplace=True)

        #--- convert / clean each column
        qualitative_data['Outcome']             = qualitative_data['Outcome'].astype(str)              # "Outcome" as string
        qualitative_data['Macro Strategy']      = qualitative_data['Macro Strategy'].astype(str)       # "Macro Strategy" as string
        qualitative_data['Strategy']            = qualitative_data['Strategy'].astype(str)             # "Strategy" as string
        qualitative_data['Setup']               = qualitative_data['Setup'].astype(str)                # "Setup" as string
        qualitative_data['MACD Proximity to 0'] = qualitative_data['MACD Proximity to 0'].astype(bool) # "MACD Proximity to 0" as boolean
        qualitative_data['Reason for Entry']    = qualitative_data['Reason for Entry'].astype(str)     # "Reason for Entry" as string
        qualitative_data['Reason for Outcome']  = qualitative_data['Reason for Outcome'].astype(str)   # "Reason for Outcome" as string

        #--- clean "Star Rating": remove 'Star' and convert to int
        qualitative_data['Star Rating'] = qualitative_data['Star Rating'].str.replace('Star', '', regex = False).astype(int)
        
        ##--- remove intervals inside parentheses () and brackets [] in "Outcome"
        qualitative_data['Outcome'] = qualitative_data['Outcome'].str.replace(r'\s*[\(\[].*?[\)\]]', '', regex = True)

        #--- convert the boolean flags
        boolean_cols = ['Left Money on the Table?', 'Followed Trading Plan?', 'Trading Plan Helped?']
        for col in boolean_cols:
            qualitative_data[col] = qualitative_data[col].astype(bool)

        ##--- ensure column types match position_records
        if not qualitative_data.empty:
            qualitative_data['Account']  = qualitative_data['Account'].astype(position_records['Account'].dtype)   # make sure consistency across all "Account" columns
            qualitative_data['Position'] = qualitative_data['Position'].astype(position_records['Position'].dtype) # make sure consistency across all "Position" columns

        ##--- merge with position records
        complete_position_records = position_records.merge(
            qualitative_data, on = ['Account', 'Position'], how = 'left'
        )

        ##--- return the joined complete positon records
        return complete_position_records


##--- execute the main method
if __name__ == '__main__':
    ##--- change directory to parent directory
    os.chdir(get_parent_directory())
    
    ##--- identify the location of the CFD trade history records
    trade_history_folder = [
                            r'data/position_records'
    ]
    
    ##--- identify the location of the broker specified symbol data
    symbols_data_folder = [
                           r'data/symbols_data'
    ]
    
    ##--- identify the location of the broker specified qualitative trade data
    qualitative_data_folder = [
                           r'data/qualitative_data'
    ]
    
    ##--- create a list for holding each individual file of CFD position records
    complete_position_records_list = []
    
    ##--- create an instance of the position cleaner class
    cleaner = Clean_CFD_Position_Records()
    
    ##--- clean every account record and symbol data file in the lists
    for trade_history_path, symbols_data_path in zip(trade_history_folder, symbols_data_folder):
        account_records           = cleaner.read_trade_history_dataframes(trade_history_folder = trade_history_path)                                     # compile all trade history
        symbols_data              = cleaner.clean_symbols_data(symbols_data_folder = symbols_data_path)                                                  # clean all the broker symbols data
        position_records          = cleaner.clean_account_records(account_records = account_records, symbols_data = symbols_data)                        # clean all CFD position records
        complete_position_records = cleaner.add_qualitative_data(position_records = position_records, qualitative_data_folder = qualitative_data_folder) # join with qualitative jounral data
        complete_position_records_list.append(complete_position_records)                                                                                 # add position records to the position records list
    
    ##--- combine all position records into one dataframe
    if complete_position_records_list:
        cfd_position_records = pd.concat(complete_position_records_list, ignore_index = True)
    else:
        cfd_position_records = pd.DataFrame()

    ##--- export completed CFD Position Records dataframe as CSV
    cfd_position_records = cfd_position_records.sort_values(by = 'Date Close')
    cfd_position_records.to_csv(r'data/complete_cfd_position_records.csv', index = False)
    print(cfd_position_records)


    