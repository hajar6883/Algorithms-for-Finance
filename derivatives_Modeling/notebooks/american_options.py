import numpy as np   
# import numpy.random as npr 
from scipy.stats import norm
from numpy.polynomial.laguerre import lagval
from sklearn.linear_model import LinearRegression

def LSM_american_option(S0, K, T, r, sigma, option_type="call", num_simulations=100000, num_steps=50):


    dt = T / num_steps
    discount = np.exp(-r * dt)

    Z = np.random.randn(num_simulations, num_steps)
    S = np.zeros((num_simulations, num_steps + 1))
    S[:, 0] = S0

    for t in range(1, num_steps + 1):
        S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1])

    if option_type == "call":
        payoffs = np.maximum(S[:, 1:] - K, 0)
    else:
        payoffs = np.maximum(K - S[:, 1:], 0)

    # Backward induction
    option_values = np.copy(payoffs[:, -1])
    
    for t in range(num_steps - 1, 0, -1):
        ITM = payoffs[:, t] > 0 #in_the_money
        X = S[ITM, t]  # Stock prices where exercise is possible
        Y = option_values[ITM] * discount  # Discounted future values

        if len(X) > 0:
            # Fit regression to approx the Continuation Value(using Laguerre polynomials)
            X_poly = np.column_stack([X**i for i in range(3)])
            model = LinearRegression().fit(X_poly, Y)

           
            continuation_values = model.predict(X_poly)
            exercise_values = payoffs[ITM, t]

            # Exercise if immediate exercise is better
            exercise_now = exercise_values > continuation_values
            option_values[ITM] = np.where(exercise_now, exercise_values, option_values[ITM] * discount)

    # Discount final option value
    return np.mean(option_values) * discount

def Stochastic_Mesh_american_option(): #alternative to regression

    pass

def Binomial_Tree_american_option(S0, K, T, r, sigma, N, option_type='call'): 
    """

    
    u : up factor (%)
    d : down factor 
    """
    dt = T / N  
    u = np.exp(sigma * np.sqrt(dt)) 
    d = 1 / u  
    p = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability

    # stock price tree
    stock_tree = np.zeros((N+1, N+1))
    for i in range(N+1):
        for j in range(i+1):
            stock_tree[j, i] = S0 * (u ** (i-j)) * (d ** j)

    # values at maturity
    option_tree = np.zeros((N+1, N+1))
    if option_type == "call":
        option_tree[:, -1] = np.maximum(stock_tree[:, -1] - K, 0)
    elif option_type == "put":
        option_tree[:, -1] = np.maximum(K - stock_tree[:, -1], 0)

    # go backward to get option price at t=0
    for i in range(N-1, -1, -1):
        for j in range(i+1):
            option_tree[j, i] = np.exp(-r * dt) * (p * option_tree[j, i+1] + (1 - p) * option_tree[j+1, i+1])

    return option_tree[0, 0]


    
def Finite_Difference(): #solve PDE numerically then use a finite difference grid to approximate

    pass



def Policy_Iteration_american_option(): #make it an MDP problem 

    pass