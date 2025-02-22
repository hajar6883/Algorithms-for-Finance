import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime
import argparse
import plotly
import plotly.graph_objects as go





class YieldCurve:
    def __init__(self, start_date="2020-01-01", end_date=None):
    
        self.maturities = ['DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS3', 'DGS5', 'DGS7', 'DGS10', 'DGS20', 'DGS30']
        self.source = 'fred'
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.datetime.today().strftime("%Y-%m-%d")

        self.data = self.fetch_yield_curve()
    

    def fetch_yield_curve(self):
        """get treasury rate for fixed maturity bonds (rates are market-implied )"""

        data = web.DataReader(self.maturities, self.source, self.start_date, self.end_date)
        data.dropna(inplace=True)
    
   
        latest_yields = data.iloc[-1].to_dict() 

        maturity_mapping = {
            'DGS1MO': 1/12, 'DGS3MO': 3/12, 'DGS6MO': 6/12, 'DGS1': 1,
            'DGS2': 2, 'DGS3': 3, 'DGS5': 5, 'DGS7': 7, 'DGS10': 10, 'DGS20': 20, 'DGS30': 30
        }

        market_yields = {maturity_mapping[k]: v/100 for k, v in latest_yields.items() if k in maturity_mapping}
        print("\nMarket Yields (Par Rates from FRED):")
        for m, y in market_yields.items():
            print(f"{m} years: {y:.4%}")
        # maturities = sorted(market_yields.keys()) 
        return market_yields

    def plot_yield_curve(self, date=None):
        """Plots the yield curve for a specific date (default is the latest available)."""

        if date is None:
            date = self.data.index[-1]  # Use the latest available date
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

    
  
def Zero_Curve( market_yields, compounding= 'semi-annual'): # for than need bootstrapping..

        """construct a zero-coupon curve from PAR-like rates  (prices of coupon-bearing products)"""

        compounding_map = {
        'annual': 1,
        'semi-annual': 2,
        'quarterly': 4,
        'monthly': 12,
        'continuous': None  
        }
        
        freq = compounding_map.get(compounding, 2)  

        maturities = sorted(market_yields.keys())
        zero_rates = {}  

        first_maturity = maturities[0] # shortest maturity is assumed to be a zero-coupon bond (Tbill)
        zero_rates[first_maturity] = market_yields[first_maturity]
            
        # Bootstrapping logic

        for i in range(1, len(maturities)):
            T = maturities[i]
            C = market_yields[T]  # Par yield (coupon rate)
            P = 100  # Par bond price

            discounted_coupons = sum((C / freq) / (1 + zero_rates[m] / freq) ** (m * freq) for m in maturities[:i])
            zero_rates[T] = ((P + (C / freq)) / (P - discounted_coupons)) ** (1 / (T * freq)) - 1

            

        return zero_rates


def Instantaneous_Forward_Rate(StDate, Tenor, curr,data,TrDate= datetime.date.today()):
    """interpolation to estimate discount factors at specific times"""

    if isinstance(StDate, pd.Timestamp):
        StDate = StDate.date()
    if isinstance(TrDate, pd.Timestamp):
        TrDate = TrDate.date()
  

    T1 = ((StDate - TrDate).days)/365
    T2 = T1 + Tenor
    dfTable = data[data.Currency == curr]
    return (np.interp(T1,dfTable['T'], dfTable['Df']) - np.interp(T2,dfTable['T'], dfTable['Df']))/((T2-T1)*np.interp(T2,dfTable['T'], dfTable['Df']))



def Forward_Curve(maturities, zero_rates, delta_T, compounding='discrete'):
    """ derive forward rates from spot rates"""
    n = len(maturities)
    forward_rates = np.zeros(n - delta_T)  

    if compounding == 'continuous':
        for t in range(n - delta_T):
            T1 = maturities[t]
            T2 = maturities[t + delta_T]
            forward_rates[t] = (zero_rates[t + delta_T] * T2 - zero_rates[t] * T1) / (T2 - T1)
    
    else : 
        for t in range(n - delta_T):
            T1 = maturities[t]
            T2 = maturities[t + delta_T]
            forward_rates[t] = ((1 + zero_rates[t + delta_T]) ** T2 / (1 + zero_rates[t]) ** T1) ** (1 / (T2 - T1)) - 1

    return dict(zip(maturities[:-delta_T], forward_rates))




# def get_discout_factor(spot_rates, maturities):
#     return [1 / ((1 + rate) ** t) for rate, t in zip(spot_rates, maturities)]




    





"""
fixed_leg_cashflows(notional, fixed_rate, payment_dates, day_count_convention)
floating_leg_cashflows(notional, floating_index, reset_dates, forward_curve, day_count_convention)	
discount_cashflows(cashflows, discount_factors)

compute_swap_NPV(fixed_leg, floating_leg, discount_factors).
calculate_par_swap_rate(discount_factors, floating_leg_dates, fixed_leg_dates)
calculate_DV01(fixed_leg, discount_factors)
#sensitivities:

calculate_PV01(fi
xed_leg, discount_factors)	Computes PV01 (the price value of a 1 basis point change in rates).
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
    market_yields= yc.data

    zero_curve =Zero_Curve(market_yields)

    print("\nBootstrapped Zero Curve (From Par Rates):")
    for m, z in zero_curve.items():
        print(f"{m} years: {z:.4%}")

    

    

if __name__ == "__main__":
    main()