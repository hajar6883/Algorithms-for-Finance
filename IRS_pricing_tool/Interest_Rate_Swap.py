from utils import * 

class IRS_Pricing:
    def __init__(self, notional, maturity, fixed_rate=None, floating_spread=0.0, 
                 payment_frequency='semi-annual', discount_factors=None, zero_curve=None):
        """
        Initialize the Interest Rate Swap pricing model.
        
        :param notional: Principal amount of the swap
        :param maturity: Swap maturity in years
        :param fixed_rate: Fixed rate (to be determined if None)
        :param floating_spread: Spread over floating benchmark (e.g., LIBOR + spread)
        :param payment_frequency: Frequency of payments (annual, semi-annual, quarterly, monthly)
        :param discount_factors: Discount factors for PV computation
        :param zero_curve: Zero-coupon yield curve (for bootstrapping forward rates)
        """
        self.notional = notional
        self.maturity = maturity
        self.fixed_rate = fixed_rate  # If None, will be computed
        self.floating_spread = floating_spread
        self.payment_frequency = payment_frequency
        self.discount_factors = discount_factors  # Dictionary {T: DF}
        self.zero_curve = zero_curve  # Dictionary {T: Zero Rate}
        
        self.freq_map = {'annual': 1, 'semi-annual': 2, 'quarterly': 4, 'monthly': 12}
        self.periods_per_year = self.freq_map.get(payment_frequency, 2)
        self.payment_dates = np.linspace(1/self.periods_per_year, maturity, int(maturity * self.periods_per_year))
    
    def compute_discount_factors(self):
        """Compute discount factors from the zero curve."""
        discount_factors = {}
        freq = self.periods_per_year #compounding frequency ..
        
        for T, rate in self.zero_curve.items():
            discount_factors[T] = (1 + rate / freq) ** (-T * freq)
        
        return discount_factors
    
    def compute_floating_cashflows(self):
        """Compute floating leg cashflows using forward rates."""
        cashflows = []
        
        # Extract zero curve time and rates for interpolation
        time_to_maturity = np.array(list(self.zero_curve.keys()))
        forward_rates = np.array(list(self.zero_curve.values()))
        
        # Interpolate forward rates
        interp_fwd = interp1d(time_to_maturity, forward_rates, kind="linear", fill_value="extrapolate")
        
        for T in self.payment_dates:
            forward_rate = interp_fwd(T) + self.floating_spread  
            df = self.discount_factors.get(T, (1 + forward_rate / self.periods_per_year) ** (-T * self.periods_per_year))
            cashflow = self.notional * forward_rate * (1 / self.periods_per_year)
            present_value = cashflow * df
            
            cashflows.append({'Time': T, 'Cashflow': cashflow, 'PV': present_value, 'Forward Rate': forward_rate})
        
        return pd.DataFrame(cashflows)
    
    def compute_fixed_cashflows(self, fixed_rate=None):
        """Compute fixed leg cashflows."""
        if fixed_rate is None:
            fixed_rate = self.fixed_rate
        
        cashflows = []
        
        for T in self.payment_dates:
            df = self.discount_factors.get(T, (1 + fixed_rate / self.periods_per_year) ** (-T * self.periods_per_year))
            cashflow = self.notional * fixed_rate * (1 / self.periods_per_year)
            present_value = cashflow * df
            
            cashflows.append({'Time': T, 'Cashflow': cashflow, 'PV': present_value})
        
        return pd.DataFrame(cashflows)
    
    def compute_fair_fixed_rate(self):
        floating_leg_pv = self.compute_floating_cashflows()["PV"].sum()
        # discount_sum = sum(self.discount_factors[T] for T in self.payment_dates)
        df_times = np.array(list(self.discount_factors.keys()))
        df_values = np.array(list(self.discount_factors.values()))
        
        interp_df = interp1d(df_times, df_values, kind="linear", fill_value="extrapolate")
        
        discount_sum = sum(interp_df(T) for T in self.payment_dates)
    
        fair_rate = floating_leg_pv / (self.notional * (1 / self.periods_per_year) * discount_sum)
        return fair_rate
    
    def compute_swap_npv(self):

        fixed_leg_pv = sum(row["PV"] for _, row in self.compute_fixed_cashflows().iterrows())
        floating_leg_pv = sum(row["PV"] for _, row in self.compute_floating_cashflows().iterrows())

        return fixed_leg_pv - floating_leg_pv
    
    def compute_dv01(self, bump=0.0001):
    
        df_times = np.array(list(self.discount_factors.keys()))
        df_values = np.array(list(self.discount_factors.values()))
        interp_df = interp1d(df_times, df_values, kind="linear", fill_value="extrapolate")

        bumped_discount_factors = {T: interp_df(T) * np.exp(-bump * T) for T in self.payment_dates} # shifting using an exponential approx. cause the bump is small ..
        
        fixed_pv_original = self.compute_fixed_cashflows()["PV"].sum()
        fixed_pv_bumped = sum(row["Cashflow"] * bumped_discount_factors[row["Time"]] 
                            for _, row in self.compute_fixed_cashflows().iterrows())
        
        return fixed_pv_original - fixed_pv_bumped

def main():
    file_path= 'data/zero_rates.csv'
    yield_data = pd.read_csv(file_path)
    usd_yield_data = yield_data[yield_data["Ticker"].str.startswith("US")].copy()

    usd_yield_data["Maturity"] = pd.to_datetime(usd_yield_data["Maturity"])
    usd_yield_data["T"] = (usd_yield_data["Maturity"] - pd.to_datetime("today")).dt.days / 365

    usd_yield_data["Yield"] /= 100

    usd_yield_data.dropna(inplace=True)

    usd_yield_data.sort_values("T", inplace=True)

    print(usd_yield_data)
    zero_curve = dict(zip(usd_yield_data["T"], usd_yield_data["Yield"]))

    notional = 1_000_000  
    maturity = 5  
    floating_spread = 0.002  # 20 basis points above LIBOR
    payment_frequency = "semi-annual"

  
    swap = IRS_Pricing(
            notional=notional,
            maturity=maturity,
            fixed_rate=None,  # To be computed
            floating_spread=floating_spread,
            payment_frequency=payment_frequency,
            discount_factors=None,  # Will be computed inside the class 
            zero_curve=zero_curve ) # Using the preprocessed zero curve

    swap.discount_factors = swap.compute_discount_factors()

    fair_fixed_rate = swap.compute_fair_fixed_rate()
    print(f"Fair Fixed Rate: {fair_fixed_rate:.6f}")

    swap.fixed_rate = fair_fixed_rate  

    swap_npv = swap.compute_swap_npv()
    
    print(f"Swap NPV: {swap_npv:.2f}")

    dv01 = swap.compute_dv01()
    print(f"DV01: {dv01:.2f}") 




    return

    

if __name__ == "__main__":
    main()


