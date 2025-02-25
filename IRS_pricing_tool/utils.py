import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime as dt
import argparse
import plotly
import plotly.graph_objects as go
import ace as tools
from scipy.interpolate import interp1d



COMPOUNDING_MAP = {
    'annual': 1,
    'semi-annual': 2,
    'quarterly': 4,
    'monthly': 12,
    'continuous': None}

# class YieldCurve:
#     def __init__(self, start_date="2020-01-01", end_date=None):
    
#         self.maturities = ['DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS3', 'DGS5', 'DGS7', 'DGS10', 'DGS20', 'DGS30']
#         self.source = 'fred'
#         self.start_date = start_date
#         self.end_date = end_date if end_date else dt.datetime.today().strftime("%Y-%m-%d")

#         self.data = self.fetch_yield_curve()
    

#     def fetch_yield_curve(self):
#         """get treasury rate for fixed maturity bonds (rates are market-implied )"""

#         data = web.DataReader(self.maturities, self.source, self.start_date, self.end_date)
#         data.dropna(inplace=True)
    
   
#         latest_yields = data.iloc[-1].to_dict() 

#         maturity_mapping = {
#             'DGS1MO': 1/12, 'DGS3MO': 3/12, 'DGS6MO': 6/12, 'DGS1': 1,
#             'DGS2': 2, 'DGS3': 3, 'DGS5': 5, 'DGS7': 7, 'DGS10': 10, 'DGS20': 20, 'DGS30': 30
#         }

#         market_yields = {maturity_mapping[k]: v/100 for k, v in latest_yields.items() if k in maturity_mapping}
#         print("\nMarket Yields (Par Rates from FRED):")
#         for m, y in market_yields.items():
#             print(f"{m} years: {y:.4%}")
#         # maturities = sorted(market_yields.keys()) 
#         return market_yields

#     def plot_yield_curve(self, date=None):
#         """Plots the yield curve for a specific date (default is the latest available)."""

#         if date is None:
#             date = self.data.index[-1]  # Use the latest available date
#         elif date not in self.data.index:
#             print(f"No data available for {date}. Using the latest available data.")
#             date = self.data.index[-1]

#         yields = self.data.loc[date, self.maturities].values
#         maturities_labels = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]

#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=maturities_labels, y=yields, mode='lines+markers', name=f'Yield Curve on {date.date()}'))

#         fig.update_layout(title="US Treasury Yield Curve",
#                           xaxis_title="Maturity",
#                           yaxis_title="Yield (%)",
#                           template="plotly_dark")

#         fig.show()

#     def plot_time_series(self, maturities=None):
#         """Plots the time-series of selected maturities over time."""
#         if maturities is None:
#             maturities = ['DGS2', 'DGS5', 'DGS10', 'DGS30'] 

#         fig = go.Figure()
#         for mat in maturities:
#             if mat in self.data.columns:
#                 fig.add_trace(go.Scatter(x=self.data.index, y=self.data[mat], mode='lines', name=mat))

#         fig.update_layout(title="US Treasury Yields Over Time",
#                           xaxis_title="Date",
#                           yaxis_title="Yield (%)",
#                           template="plotly_dark")
        


#         fig.show()

    
  
def Zero_Curve( market_yields, compounding= 'semi-annual'): # for than need bootstrapping..

        """construct a zero-coupon curve from PAR-like rates  (prices of coupon-bearing products)"""

       
        
        freq = COMPOUNDING_MAP.get(compounding, 2)  

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


def Instantaneous_Forward_Rate(StDate, Tenor, Discounted_Yields, Time_to_maturity,TrDate= dt.date.today()):
    """interpolation to estimate forward rates at specific times"""

    if isinstance(StDate, pd.Timestamp):
        StDate = StDate.date()
    if isinstance(TrDate, pd.Timestamp):
        TrDate = TrDate.date()
  

    T1 = ((StDate - TrDate).days)/365
    T2 = T1 + Tenor
    # Ensure T1 and T2 are within the valid range of Time_to_maturity
    T_min, T_max = min(Time_to_maturity), max(Time_to_maturity)
    T1 = max(T_min, min(T1, T_max))
    T2 = max(T_min, min(T2, T_max))

    # Ensure T1 is not negative (floating rate resets cannot be in the past)
    T1 = max(0, T1)

    interp_df = interp1d(Time_to_maturity, Discounted_Yields, kind='linear', fill_value="extrapolate")

    # Compute interpolated discount factors
    Df_T1 = interp_df(T1)
    Df_T2 = interp_df(T2)

    # Compute forward rate
    fwd_rate = (Df_T1 - Df_T2) / ((T2 - T1) * Df_T2)
    if fwd_rate > 1: 
        fwd_rate /= 100
    
    # return (np.interp(T1, Time_to_maturity, Discounted_Yields) - np.interp(T2, Time_to_maturity, Discounted_Yields)) / ((T2 - T1) * np.interp(T2, Time_to_maturity, Discounted_Yields))
    return fwd_rate



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


def compute_day_count_fraction(start_date, end_date, day_count_convention="30/360"):

    if day_count_convention == "30/360":
        return ((end_date.year - start_date.year) * 360 + 
                (end_date.month - start_date.month) * 30 + 
                (end_date.day - start_date.day)) / 360
    elif day_count_convention == "Actual/360":
        return (end_date - start_date).days / 360
    else:
        raise ValueError("Unsupported day count convention")


def fixedLegCashflows(notional, fixed_rate, payment_dates, day_count_convention= "30/360"):
    cashflows = []
    for i in range(1, len(payment_dates)):  
        start_date = payment_dates[i-1]
        end_date = payment_dates[i]

        day_count_fraction = compute_day_count_fraction(start_date, end_date, day_count_convention)

    
        cashflow = notional * fixed_rate * day_count_fraction
        cashflows.append({"Payment Date": end_date, "Fixed Cashflow": cashflow}) #same cashflow until notional or rate is changed!

    return pd.DataFrame(cashflows)



def floatingLegCashflows(notional, zero_curve, reset_dates,day_count_convention="30/360"):

    cashflows = []

    time_to_maturity = zero_curve["T"]
    forward_rates = zero_curve["Forward Rate"]
    
    interp_fwd = interp1d(time_to_maturity, forward_rates, kind="linear", fill_value="extrapolate")

    for i in range(1, len(reset_dates)):
        start_date = reset_dates[i-1]
        end_date = reset_dates[i]

        day_count_fraction = compute_day_count_fraction(start_date, end_date, day_count_convention)

        T1 = (start_date - dt.date.today()).days / 365
        print("T1:", T1)

        # Ensure T1 is within valid interpolation range
        T_min, T_max = min(time_to_maturity), max(time_to_maturity)
        T1 = max(T_min, min(T1, T_max))

        forward_rate = interp_fwd(T1) / 100  
        print("Interpolated Forward Rate:", forward_rate)
        floating_rate = forward_rate  
        cashflow = notional * floating_rate * day_count_fraction
        print("Computed Cashflow:", cashflow)

      
        cashflows.append({"Payment Date": end_date, "Floating Cashflow": cashflow, "Forward Rate": forward_rate})

    return pd.DataFrame(cashflows)
    

def compute_discount_factors(zero_rates, maturities, compounding='semi-annual'):
    """Computes discount factors from zero rates."""
    
    freq = COMPOUNDING_MAP.get(compounding, 2)
    discount_factors = {}

    for T in maturities:
        if compounding == 'continuous':
            discount_factors[T] = np.exp(-zero_rates[T] * T)
        else:
            discount_factors[T] = (1 + zero_rates[T] / freq) ** (-T * freq)
    
    return discount_factors

def price_swap(fixed_leg, floating_leg, discount_factors):
    """
    Computes NPV of the swap.
    fixed_leg: Df with fixed leg cashflows
    floating_leg: Df with floating leg cashflows
    discount_factors:  for PV computation
    """

    fixed_pv = sum(row["Fixed Cashflow"] * discount_factors[row["Payment Date"].year] for _, row in fixed_leg.iterrows())


    floating_pv = sum(row["Floating Cashflow"] * discount_factors[row["Payment Date"].year] for _, row in floating_leg.iterrows())

    npv = fixed_pv - floating_pv

    return npv

    
def compute_DV01(fixed_leg, discount_factors, bump=0.0001):
    """
    Computes the DV01 (Dollar Value of 1 Basis Point). Bumps the discount curve by 1bp and computes change in PV.
    """
    bumped_discount_factors = {T: df * np.exp(-bump * T) for T, df in discount_factors.items()}

    fixed_pv_original = sum(row["Fixed Cashflow"] * discount_factors[row["Payment Date"].year] for _, row in fixed_leg.iterrows())
    fixed_pv_bumped = sum(row["Fixed Cashflow"] * bumped_discount_factors[row["Payment Date"].year] for _, row in fixed_leg.iterrows())

    return fixed_pv_original - fixed_pv_bumped


