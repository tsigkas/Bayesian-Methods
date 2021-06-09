""" Some useful packages"""
import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt
import pystan as stan
import seaborn as sea

np.random.seed(1111) # Set seed for reproducibility

""" Functions that draws samples from an AR(1)-process """
def AR1(mu, phi, sigma, T, x0):
    x = np.zeros(T+1)
    x[0] = x0
    for t in range(1, T+1):
        x[t] = st.norm.rvs(loc= mu+phi*(x[t-1]-mu), scale=sigma)
    return x

""" Explore the impact of phi on x """
x = AR1(20, 0.3, 2, 200, 20)
y = AR1(20, 0.9, 2, 200, 20)

plt.plot(x)
plt.plot(y)
plt.legend([r"$X$: $\phi=0.3$","$Y$: $\phi=0.9$"])

""" Define the AR-process in PyStan """
AR_sm = stan.StanModel(file='AR1.stan')

""" And treat the previous simulations as the datasets """
AR_x_data = {"T" : 201,
             "x" : x}
AR_y_data = {"T" : 201,
             "x" : y}

x_fit = AR_sm.sampling(data=AR_x_data, iter=5000, warmup=1000)
# Increase warmup to get a better n_eff
y_fit = AR_sm.sampling(data=AR_y_data, iter=5000, warmup=2500)

x_fit_df = x_fit.to_dataframe()[['phi', 'mu', 'sigma']]
y_fit_df = y_fit.to_dataframe()[['phi', 'mu', 'sigma']]

sea.pairplot(x_fit_df)
sea.pairplot(y_fit_df)
