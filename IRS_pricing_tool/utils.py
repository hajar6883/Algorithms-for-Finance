import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime
import argparse
import plotly
import plotly.graph_objects as go





# Market Data Handling
# 


class YieldCurve:
    def __init__(self):
    
        self.maturities = ['DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS3', 'DGS5', 'DGS7', 'DGS10', 'DGS20', 'DGS30']
        self.source = 'fred'
        self.data = self.fetch_yield_curve()

    def fetch_yield_curve(self, start_date = "2018-01-01", end_date= datetime.datetime.today().strftime("%Y-%m-%d") ):
        """get treasury rate for fixed maturity bonds (rates are market-implied )"""

        data = web.DataReader(self.maturities, self.source, start_date, end_date)
        data.dropna(inplace=True)
        return data

    def plot_yield_curve(self, date=None):
        """Plots the yield curve for a specific date (default is the latest available)."""

        if date is None:
            date = self.data.index[-1] 
        elif date not in self.data.index:
            print(f"No data available for {date}. Using the latest available data.")
            date = self.data.index[-1]

        yields = self.data.loc[date, self.maturities].values
        maturities_labels = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=maturities_labels, y=yields, mode='lines+markers', name=f'Yield Curve on {date.date()}'))

        fig.update_layout(title="US Treasury Yield Curve",
                          xaxis_title="Maturity",
                          yaxis_title="Yield (%)",
                          template="plotly_dark")

        fig.show()

    def plot_time_series(self, maturities=None):
        """Plots the time-series of selected maturities over time."""
        if maturities is None:
            maturities = ['DGS2', 'DGS5', 'DGS10', 'DGS30'] 

        fig = go.Figure()
        for mat in maturities:
            if mat in self.data.columns:
                fig.add_trace(go.Scatter(x=self.data.index, y=self.data[mat], mode='lines', name=mat))

        fig.update_layout(title="US Treasury Yields Over Time",
                          xaxis_title="Date",
                          yaxis_title="Yield (%)",
                          template="plotly_dark")

        fig.show()





    

def bootstrap_zero_curve(obs_prices, T, face_value=100): 

    # price, face_value= market_data
    # spot_rates = [(face_value / price) ** (1 / maturity) - 1 for price, maturity in zip(prices, maturities)]
    # return spot_rates
    pass

def get_discout_factor(spot_rates, maturities):
    return [1 / ((1 + rate) ** t) for rate, t in zip(spot_rates, maturities)]
    

def get_forward_rates(spot_curve):
    pass





    """
    fixed_leg_cashflows(notional, fixed_rate, payment_dates, day_count_convention)
floating_leg_cashflows(notional, floating_index, reset_dates, forward_curve, day_count_convention)	
discount_cashflows(cashflows, discount_factors)

compute_swap_NPV(fixed_leg, floating_leg, discount_factors).
calculate_par_swap_rate(discount_factors, floating_leg_dates, fixed_leg_dates)
calculate_DV01(fixed_leg, discount_factors)
#sensitivities:

calculate_PV01(fixed_leg, discount_factors)	Computes PV01 (the price value of a 1 basis point change in rates).
compute_IR_Delta(swap_price, shocked_rates)	Computes interest rate delta (how swap value changes with rates).
calculate_convexity_adjustment()

vis & reporting :
plot_yield_curve(yield_curve_data)	Plots the yield curve (zero rates, forward rates, etc.).
plot_cashflows(fixed_leg, floating_leg)	Visualizes swap cash flows over time.
generate_swap_report(swap_details, npv, risk_metrics)
    """




def main():
    # parser = argparse.ArgumentParser(description="Run different functions for the swap pricing tool.")
    
    # parser.add_argument("function", choices=FUNCTION_MAP.keys(), help="Function to execute")
    
    # parser.add_argument("--source", type=str, default="fred", help="Data source for yield curve")
    # parser.add_argument("--timeframe", type=str, default="DGS1", help="constant maturity Treasury yield")


    # args = parser.parse_args()

    # result = FUNCTION_MAP[args.function](**vars(args))
    


    yc = YieldCurve()
    yc.data=yc.fetch_yield_curve(start_date="2020-01-01")
    yc.plot_yield_curve()  
    yc.plot_time_series()  

if __name__ == "__main__":
    main()