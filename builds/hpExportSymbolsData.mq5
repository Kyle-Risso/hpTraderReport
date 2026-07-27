//+-------------------------------------------------------------------------+
//|                                                    hpSymbolExporter.mq5 |
//|        Copyright 2023-2024, HP Investment Trading & Gambling Strategies |
//|                                                     https://hp-fx-g.com |
//+-------------------------------------------------------------------------+
#property copyright   "Copyright 2023-2024, HP Investment Trading and Gambling Strategies"
#property link        "https://hp-fx-g.com"
#property description "Exports Account Symbol Data for All Provided Symbols on an Account"
#property version     "1.0"

//--- establish global variables
long account_number = AccountInfoInteger(ACCOUNT_LOGIN);  // get account number
input string FileName = "symbols_data.csv";               // default CSV filename

//+-------------------------------------------------------------------------+
//| @func: On Start (Script Execution)                                      |
//| @desc: retrieves the symbol data for the account and exports as CSV     |
//| @params: N/A                                                            |
//| @return: symbols data --> the CSV file full of written symbol data      |
//+-------------------------------------------------------------------------+
void OnStart() {
   //--- establish local variables
   int total_symbols = SymbolsTotal(false);                                 // get total number of symbols
   string full_path = StringFormat("%d_symbols_data.csv", account_number);  // dynamically create the filename
   //--- open file for writing
   int handle = FileOpen(full_path, FILE_WRITE|FILE_CSV);
   //--- write the CSV file with the symbol data
   if (handle != INVALID_HANDLE) {
      FileWrite(handle, "Symbol", "Description", "Category", "Contract Size", "Decimal Value", "Spread"); // set the header column names
      //--- loop through every symbol
      for (int i = 0; i < total_symbols; i++) {
         string symbol = SymbolName(i, false); // identify the extracted symbol
         //--- fill remaining header information for the identified symbol
         if (symbol != "") {
            string desc = SymbolInfoString(symbol, SYMBOL_DESCRIPTION);                  // get asset description
            long decimal = SymbolInfoInteger(symbol, SYMBOL_DIGITS);                     // get asset decimal place value
            long spread = SymbolInfoInteger(symbol, SYMBOL_SPREAD);                      // get default spread
            double contract_size = SymbolInfoDouble(symbol, SYMBOL_TRADE_CONTRACT_SIZE); // get unit value of 1 lot
            string category = SymbolInfoString(symbol, SYMBOL_SECTOR_NAME);              // get asset category
            FileWrite(handle, symbol, desc, category, contract_size, decimal, spread);   // write completed symbol info column headers
         }
      }
      //--- close the CSV file
      FileClose(handle); 
      Print("Symbol Information Extracted Successfully; Exported To: ", full_path); // print confirmation of successful extraction
   //--- print error to console if/when extraction fails
   } else {
      Print("Error in Opening File: ", full_path);
   }
}
//+-------------------------------------------------------------------------+
