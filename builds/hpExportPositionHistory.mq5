//+-------------------------------------------------------------------------+
//|                                           hpPositionHistoryExporter.mq5 |
//|        Copyright 2023-2024, HP Investment Trading & Gambling Strategies |
//|                                                     https://hp-fx-g.com |
//+-------------------------------------------------------------------------+
#property copyright   "Copyright 2023-2024, HP Investment Trading and Gambling Strategies"
#property link        "https://hp-fx-g.com"
#property description "Exports Account Trade History for the Account"
#property version     "1.1"

//--- establish global variables
long account_number = AccountInfoInteger(ACCOUNT_LOGIN);
datetime start_date = D'2023.01.01 00:00.000';
datetime end_date = TimeCurrent();  // Use TimeCurrent() directly

//+-------------------------------------------------------------------------+
//| @func: On Start (Script Execution)                                      |
//| @desc: retrieves the trade history for the account and exports as CSV   |
//| @params: N/A                                                            |
//| @return: trade history --> the CSV file full of written trade data      |
//+-------------------------------------------------------------------------+
void OnStart() {
    //--- dynamically set the file path
    string file_path = StringFormat("%d_trade_history.csv", account_number);

    //--- create the file handle for writing to CSV
    int file_handle = FileOpen(file_path, FILE_WRITE | FILE_CSV);
    string data = NULL;
    //--- throw exception when file cannot be opened/read
    if (file_handle == INVALID_HANDLE) {
        Alert("File Open Failed: ", _LastError);
        return;
    }
    //--- set the trade history data headers
    if (HistorySelect(start_date, end_date)) {
        StringAdd(data, "Account\t");  // Add Account column header
        StringAdd(data, "Ticket\t");
        StringAdd(data, "Date Open\t");
        StringAdd(data, "Symbol\t");
        StringAdd(data, "Category\t");
        StringAdd(data, "Contract Size\t");
        StringAdd(data, "Decimals\t");
        StringAdd(data, "Type\t");
        StringAdd(data, "Volume\t");
        StringAdd(data, "Strategy\t");
        StringAdd(data, "Price Open\t");
        StringAdd(data, "S/L\t");
        StringAdd(data, "T/P\t");
        StringAdd(data, "Date Close\t");
        StringAdd(data, "Price Close\t");
        StringAdd(data, "Gross Profit\t");
        StringAdd(data, "Commission\t");
        StringAdd(data, "Swap\t");
        StringAdd(data, "Net Profit");

        //--- write headers to CSV
        FileWrite(file_handle, data);
        ulong deal_in_ticket = -1;
        bool has_duplicate;
        //--- select all deals and store in array
        int end_date_total = HistoryDealsTotal();
        ulong array_positions[];
        ArrayResize(array_positions, end_date_total, true);
        //--- loop through all deals and add to position identifier
        for (int i = 0; i < end_date_total; i++) {
            //--- iterate through the position opens
            if ((deal_in_ticket = HistoryDealGetTicket(i)) > 0 &&
                HistoryDealGetInteger(deal_in_ticket, DEAL_ENTRY) == DEAL_ENTRY_IN) {
                //--- grab the position identifier
                ulong position_id = HistoryDealGetInteger(deal_in_ticket, DEAL_POSITION_ID);
                has_duplicate = false;
                //--- loop through data again for possible duplicates
                for (int j = i; j >= 0; j--) {
                    if (array_positions[j] == position_id) {
                        has_duplicate = true; // stop if/when a duplicate is found
                        break;
                    }
                }
                if (has_duplicate) continue; // iterate again if duplicate was identified
                array_positions[i] = position_id; // store ticket in array if duplicate is not found
            }
        }
        int history_deals_by_position = -1;
        int history_orders_by_position = -1;
        int size = ArraySize(array_positions);
        int cnt = 0;
        //--- process the tickets
        for (int i = 0; i < size; i++) {
            //--- declare all used variables
            long direction = -1, magic_number = -1;
            ulong position_id = 0, deal_ticket, order_ticket;
            double price_open = -1, price_close = -1, deal_volume = 0, take_profit = -1, stop_loss = -1, profit = 0, swap = 0, commission = 0;
            string symbol = NULL, date_close = NULL, date_open = NULL;
            double contract_size = 0;
            int decimals = 0;
            string category = NULL;
            //--- combine matching deals
            if (HistorySelectByPosition(array_positions[i])) {
                if (array_positions[i] == 0) continue; // skip if no ticket is found
                cnt++; // update the counter
                //--- count cases with same ticket
                history_deals_by_position = HistoryDealsTotal();
                history_orders_by_position = HistoryOrdersTotal();
                //--- looping through every deal
                for (int j = 0; j <= history_deals_by_position; j++) {
                    deal_ticket = HistoryDealGetTicket(j); // get the deal ticket
                    //--- if there is no ticket then reiterate
                    if (deal_ticket == 0) continue;
                    //--- collect deal information for position closes
                    if (HistoryDealGetInteger(deal_ticket, DEAL_ENTRY) != DEAL_ENTRY_IN) {
                        date_close = TimeToString(HistoryDealGetInteger(deal_ticket, DEAL_TIME), TIME_DATE) + " " +
                                     TimeToString(HistoryDealGetInteger(deal_ticket, DEAL_TIME), TIME_SECONDS);
                        price_close = HistoryDealGetDouble(deal_ticket, DEAL_PRICE);
                        deal_volume += HistoryDealGetDouble(deal_ticket, DEAL_VOLUME);
                    }
                    //--- collect deal information for position opens
                    if (HistoryDealGetInteger(deal_ticket, DEAL_ENTRY) == DEAL_ENTRY_IN) {
                        direction = HistoryDealGetInteger(deal_ticket, DEAL_TYPE);
                        date_open = TimeToString(HistoryDealGetInteger(deal_ticket, DEAL_TIME), TIME_DATE) + " " +
                                    TimeToString(HistoryDealGetInteger(deal_ticket, DEAL_TIME), TIME_SECONDS);
                        price_open = HistoryDealGetDouble(deal_ticket, DEAL_PRICE);
                    }
                    position_id = HistoryDealGetInteger(deal_ticket, DEAL_POSITION_ID);
                    magic_number = HistoryDealGetInteger(deal_ticket, DEAL_MAGIC);
                    symbol = HistoryDealGetString(deal_ticket, DEAL_SYMBOL);
                    commission += HistoryDealGetDouble(deal_ticket, DEAL_COMMISSION);
                    swap += HistoryDealGetDouble(deal_ticket, DEAL_SWAP);
                    profit += HistoryDealGetDouble(deal_ticket, DEAL_PROFIT);
                }
                //--- get contract size and decimals for the symbol
                contract_size = SymbolInfoDouble(symbol, SYMBOL_TRADE_CONTRACT_SIZE);
                decimals = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
                //--- get symbol category
                category = SymbolInfoString(symbol, SYMBOL_SECTOR_NAME);
                //--- looping through every order
                for (int j = history_orders_by_position - 1; j >= 0; j--) {
                    order_ticket = HistoryOrderGetTicket(j);
                    if (order_ticket == 0) continue; // iterate if there is no order ticket
                    //--- check for orders with same position
                    if (HistoryOrderGetInteger(order_ticket, ORDER_POSITION_ID) == position_id) {
                        take_profit = HistoryOrderGetDouble(order_ticket, ORDER_TP);
                        stop_loss = HistoryOrderGetDouble(order_ticket, ORDER_SL);
                    }
                }
                //--- assign proper data based on trade direction
                string position_type = NULL;
                double position_pnl_points = 0, position_swap_points = 0, position_commission_points = 0, total_pnl = 0, total_pnl_points = 0;
                if (direction == DEAL_TYPE_BUY) {
                    position_type = "buy";
                    position_pnl_points = NormalizeDouble(price_close - price_open, _Digits);
                    position_swap_points = NormalizeDouble(swap / SymbolInfoDouble(symbol, SYMBOL_POINT), _Digits);
                    position_commission_points = NormalizeDouble(commission / SymbolInfoDouble(symbol, SYMBOL_POINT), _Digits);
                    total_pnl_points = NormalizeDouble(position_pnl_points + position_swap_points - position_commission_points, _Digits);
                } else if (direction == DEAL_TYPE_SELL) {
                    position_type = "sell";
                    position_pnl_points = NormalizeDouble(price_open - price_close, _Digits);
                    position_swap_points = NormalizeDouble(swap / SymbolInfoDouble(symbol, SYMBOL_POINT), _Digits);
                    position_commission_points = NormalizeDouble(commission / SymbolInfoDouble(symbol, SYMBOL_POINT), _Digits);
                    total_pnl_points = NormalizeDouble(position_pnl_points + position_swap_points - position_commission_points, _Digits);
                }
                total_pnl = NormalizeDouble((profit + swap + commission), 2);
                //--- add trade data to variable 'data'
                data = StringFormat("%d\t%d\t%s\t%s\t%s\t%.2f\t%d\t%s\t%.2f\t%d\t%.5f\t%.5f\t%.5f\t%s\t%.5f\t%.2f\t%.2f\t%.2f\t%.2f",
                                    account_number, position_id, date_open, symbol, category, contract_size, decimals, position_type, deal_volume, magic_number, price_open,
                                    stop_loss, take_profit, date_close, price_close, profit, commission, swap, total_pnl);
                //--- write the trade data to file
                FileWrite(file_handle, data);
            }
        }
        Alert("Trade History Exported for Account ", account_number, " to file: ", file_path);
        //--- close file handle
        FileClose(file_handle);
    } else {
        Alert("History Select Error: ", _LastError);
    }
}

