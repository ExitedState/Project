import json
import pandas as pd

class StockData:
    
    def __init__(self, stock_info, years) -> None:
        self.stock_info = stock_info
        self.years = years
        self.stocks = list(stock_info.keys())
    
    def calculate_dividend_yields(self):
        num_years = len(self.years)
        dividend_yield_data = {year: {} for year in self.years}
        for ticker, data in self.stock_info.items():
            for index, year in enumerate(self.years): 
                price = data["prices"][num_years - 1 - index]  # Access prices in reverse order
                dividend = data["dividends"][num_years - 1 - index]  # Access dividends in reverse order
                dividend_yield = (dividend / price) * 100 if price != 0 else 0
                dividend_yield_data[year][ticker] = dividend_yield
        dividend_yield_df = pd.DataFrame(dividend_yield_data).T
        return dividend_yield_df.to_numpy(), dividend_yield_df
    
    # !!you can add other methods here such as capital gains, etc.
    # !!def calculate_capital_gains(self):...
    
    def print_dividend_yields(self):
        _,dividend_yield_df = self.calculate_dividend_yields()
        print(dividend_yield_df)
        
    # def to_json(self, filename):
    #     stock_data = {"stocks": []}
    #     for ticker, data in self.stock_info.items():
    #         stock_entry = {
    #             "ticker": ticker,
    #             "historical_data": {
    #                 "prices": [
    #                     {"year": str(year), "close": data["prices"][index]} 
    #                     for index, year in enumerate(self.years)
    #                 ],
    #                 "dividends": [
    #                     {"year": str(year), "dividend_amount": data["dividends"][index]} 
    #                     for index, year in enumerate(self.years)
    #                 ],
    #             }
    #         }
    #         stock_data["stocks"].append(stock_entry)
    #     with open(filename, "w") as json_file:
    #         json.dump(stock_data, json_file, indent=4)