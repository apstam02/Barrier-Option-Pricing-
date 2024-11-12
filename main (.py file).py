import numpy as np
import matplotlib.pyplot as plt

#Computational Finance 2, 5th Project, Stamenas Apostolos

def barrier_option_pricing(So,K,r,d,sigma,T,barrier,option_type,barrier_type,steps=365,simulations=200):
    """
    Prices a Barrier Option via a Monte Carlo Simulation (standard settings: 365 steps, 200 simulations)
    Parameters:
        So: Initial stock price.
        K: Strike Price
        r: Risk-free rate.
        d: Dividend yield.
        sigma: Volatility of the asset (in std terms).
        barrier = the level of barrier
        T: Total time for the simulation (in years).
        option_type: Call or Put
        barrier type: In our Out (that is, Option is to be activated if barrier hit, or deactivated if barrier hit)
    Methodology:
        This function is comprised of nested functions which:
        1. Create spot price pathes, by simulating the price for one year, using the GBM in a Monte Carlo Simulation, for a 1 year horizon.
        2. Find the payoff for each path.
        3. Iterate the above process, by creating 200 paths, finding the mean of the payoffs, and then discounting into the present to find the Option's Price.
        """
    def MonteCarloSim(So, r, d, sigma, T, steps):
        z = np.random.standard_normal(steps)
        path = list()
        path.append(So)
        S = So
        dt = T / steps
        for i in range(steps):
            S = S * np.exp((r - d - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z[i])
            path.append(S)
        return path

    def path_payoff_calc(So, r, d, sigma, T, steps, K, barrier, option_type, barrier_type):
        path = MonteCarloSim(S, r, d, sigma, T, steps)
        option_diactivation = False
        option_activation = False
        if S < barrier:
            if barrier_type == 'in':
                for i in range(len(path)):
                    if path[i] >= barrier:
                        option_activation = True
                        break
                if option_activation == True:
                    if option_type == 'call':
                        payoff = np.maximum(path[-1] - K, 0)
                    else:
                        payoff = np.maximum(K - path[-1], 0)
                else:
                    payoff = 0
            elif barrier_type == 'out':
                for i in range(len(path)):
                    if path[i] >= barrier:
                        option_diactivation = True
                        break
                if option_diactivation == True:
                    payoff = 0
                else:
                    if option_type == 'call':
                        payoff = np.maximum(path[-1] - K, 0)
                    else:
                        payoff = np.maximum(K - path[-1], 0)
        if S > barrier:
            if barrier_type == 'in':
                for i in range(len(path)):
                    if path[i] <= barrier:
                        option_activation = True
                        break
                if option_activation == True:
                    if option_type == 'call':
                        payoff = np.maximum(path[-1] - K, 0)
                    else:
                        payoff = np.maximum(K - path[-1],0)
                else:
                        payoff = 0
            elif barrier_type == 'out':
                for i in range(len(path)):
                    if path[i] <= barrier:
                        option_diactivation = True
                        break
                if option_diactivation == True:
                    payoff = 0
                else:
                    if option_type == 'call':
                        payoff = np.maximum(path[-1] - K, 0)
                    else:
                        payoff = np.maximum(K - path[-1],0)
        return payoff
    payoffs = list()
    for i in range(simulations):
        payoffs.append(path_payoff_calc(So, r, d, sigma, T, steps, K, barrier, option_type, barrier_type))
    expected_value = np.mean(payoffs)
    pv = np.exp(-r*T)*expected_value
    return pv

S = 100
r = 0.05
d = 0.01
T = 1
sigma = 0.3

barrier_put = 90
put_in = list()
put_out = list()
for i in np.arange(75,130,5):
    put_in.append(barrier_option_pricing(S,i,r,d,sigma,T,barrier_put,'put','in'))
    put_out.append(barrier_option_pricing(S,i,r,d,sigma,T,barrier_put,'put','out'))

barrier_call = 110
call_in = list()
call_out = list()
for i in np.arange(75,130,5):
    call_in.append(barrier_option_pricing(S,i,r,d,sigma,T,barrier_call,'call','in'))
    call_out.append(barrier_option_pricing(S,i,r,d,sigma,T,barrier_call,'call','out'))

plt.subplot(2, 2, 1)
plt.grid(True)
plt.ylabel('Put Price')
plt.xlabel('K')
plt.plot(np.arange(75,130,5),put_in)
plt.title('Put Option - down and in barrier (barrier = 90)')

plt.subplot(2, 2, 2)
plt.grid(True)
plt.xlabel('K')
plt.ylabel('Put Price')
plt.plot(np.arange(75,130,5),put_out)
plt.title('Put Option - down and out barrier (barrier = 90)')

plt.subplot(2, 2, 3)
plt.xlabel('K')
plt.ylabel('Call Price')
plt.grid(True)
plt.plot(np.arange(75,130,5),call_in)
plt.title('Call Option - up and in barrier (barrier = 110)')

plt.subplot(2, 2, 4)
plt.grid(True)
plt.ylabel('Call Price')
plt.xlabel('K')
plt.plot(np.arange(75,130,5),call_out)
plt.title('Call Option - up and out barrier (barrier = 110) ')

plt.suptitle('Barrier Option Prices on different Strike levels')
plt.tight_layout()
plt.show()















