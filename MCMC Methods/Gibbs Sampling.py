""" Some useful packages"""
import pandas as pd
import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt

np.random.seed(1111) # set seed for reproducability

""" Import and transform the data """
data = pd.read_csv('rainfall.dat', header = None)
log_y = np.log(data.values)

""" Helper functions"""
# Draws 1 sample from the Scale-Inverse-chi2 distribution
def draw_chi2inv(df, scale):
    x = st.chi2.rvs(df-1, scale, size=1)
    sigma_sq = (df-1)*scale/x
    return sigma_sq

# Returns the parameters for the posterior of mu
def mu_post_params(pri_params, sigma, data):
    [mu_0, tau_0, nu_0, sigma_0] = pri_params
    n = len(data)
    w = (n/sigma**2)/(n/sigma**2 + tau_0**(-2))
    mu_n = w*np.mean(data)+(1-w)*mu_0
    tau_n = (1/(n/sigma**2 + 1/tau_0**2))**0.5
    return [mu_n, tau_n] # Mean and sd for the normal distribution

# Returns the parameters for the posterior of sigma^2
def sigma_post_params(pri_params, mu, data):
    [mu_0, tau_0, nu_0, sigma_0] = pri_params
    n = len(data)
    nu_n = nu_0 + n
    scale = (nu_0*sigma_0**2 + np.sum((data-mu)**2) )/(nu_n)
    return [nu_n, scale] #DoF and scale parameter for the Scale-Inv-chi2

# Draws nDraws Gibbs-samples from the posterior
def Gibbs(pri_params, post_params1, post_params2, dist1, dist2, iVals, nDraws, data):
    var1 = np.zeros(nDraws+1)
    var2 = np.zeros_like(var1)
    # Initialize the Gibbs algorithm with the initial values
    [var1[0], var2[0]] = iVals
    # Sample iteratively from the conditional marginal distributions
    for i in range(1,nDraws+1):
        [mu_n, tau_n] = post_params1(pri_params, var2[i-1], data)
        var1[i] = dist1(mu_n, tau_n)
        [nu_n, scale] = post_params2(pri_params, var1[i],   data)
        var2[i] = dist2(nu_n, scale)
    return [var1[1:], var2[1:]]

# Sample autocorrelation
def acf(x):
    n = len(x)
    autocorr = []
    for i in range(1, len(x)-1):
        autocorr.append(np.corrcoef(x[:n-i],x[i:])[0,1])
    return np.array(autocorr)
    
""" Prior parameters """
# pri_params = mu_0, tau_0, nu_0, sigma_0
pri_params = np.array([3.7, 100.0, 1000.0, 1.0])
iVals = np.ones(2)

[sample_mu, sample_sigma2] = Gibbs(pri_params, 
                            mu_post_params, sigma_post_params, 
                            st.norm.rvs, draw_chi2inv,
                            iVals,
                            5000,
                            log_y
                            )

# Calculate the inefficiency factor
autocorr = acf(sample_sigma2)
IF = 1 + 2*np.sum(autocorr)

# Scatterplot of the simulated draws
plt.scatter(sample_mu, sample_sigma2)
plt.xlabel(r"$\mu$")
plt.ylabel(r"$\sigma^2$")
plt.show

# Trajectories of the two Markov chains
fig, axs = plt.subplots(2,1)
axs[0].plot(sample_mu)
axs[1].plot(sample_sigma2, color = "r")

axs[0].set_title(r"$\mu$ (blue), $\sigma^2$ (red)")
plt.savefig("Gibbs_Convergence.png", dpi = 800)
plt.show

# Compare the empirical data to the posterior predictive distribution 
# from the MCMC simulation
y_pred = np.zeros((5000,1))

for i in range(len(y_pred)):
    y_pred[i] = sample_mu[i] + np.sqrt(sample_sigma2[i])*st.norm.rvs(1)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.hist(log_y, density = True, color = "b")
ax2.hist(y_pred, color="r", histtype = "step", density = True)

ax1.set_xlabel('log of precipitation')
ax1.set_ylabel('Real data', color='b')
ax2.set_ylabel('Simulated posterior predictive density', color='r')
plt.show()