""" Useful packages """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st

np.random.seed(1234) # set seed for reproducability

""" Helper functions for drawing samples and calculating posterior parameters """
# Generates num random draws from a scale-inverse-chi2 distribution
def draw_chi2inv(df, scale, num):
    x = st.chi2.rvs(df-1, scale, size = num)
    sigma_sq = (df-1)*scale/x
    return sigma_sq

# Generates multivariate normal random draws for different sigmas
def draw_mvnorm(mu, Omega_inv, sigma_sq):
    beta = np.zeros((len(sigma_sq),3))
    for i, sig2 in enumerate(sigma_sq):
        beta[i] = st.multivariate_normal.rvs(mean = mu, cov=sig2*Omega_inv, size = 1)
    return np.transpose(beta)

# Function that generates posterior parameters for the posterior
def post_params(X, y, mu_0, Omega_0, sigma_0):
    n = len(y)
    
    X_dagger =np.linalg.pinv(X)
    XTX = X.T@X
    beta_hat = X_dagger @ y
    
    mu_n = np.linalg.inv(Omega_0 + XTX) @ XTX@beta_hat + Omega_0@mu_0
    Omega_n = XTX + Omega_0
    nu_n = nu_0 + n
    
    yTy = y.T@y
    mu_Omega_mu_0 = mu_0.T @ Omega_0 @ mu_0
    mu_Omega_mu_n = mu_n.T @ Omega_0 @ mu_n
    
    sigma_n2 = (nu_0*sigma_0**2 + yTy + mu_Omega_mu_0 + mu_Omega_mu_n )*1/nu_n
    
    return [mu_n, Omega_n, nu_n, sigma_n2]

# Returns num random draws from the posterior of beta and sigma^2, 
# given the posterior parameters as input
def post_draws(mu_n, Omega_n, nu_n, sigma_n2, num):
    sigma_2     = draw_chi2inv(nu_n, sigma_n2, num)
    beta_post   = draw_mvnorm(mu_n, np.linalg.inv(Omega_n),sigma_2)
    return [beta_post, sigma_2]

""" Import data"""
data = pd.read_csv("TempLinkoping.txt", sep = "\t+") 

Y = data.temp.values
# Data matrix consisting of 1, temp, temp^2
X = np.column_stack(((np.ones_like(data.time.values), data.time.values, data.time.values**2)))

# Plot the data
plt.scatter(data.time.values,Y)
plt.xlabel("Time (year)")
plt.ylabel("Temperature (deg C)")
plt.show

# Prior parameters
mu_0 = np.array([-15,120,-110])
Omega_0 = 0.05*np.eye(3) 
Omega_0_inv  = np.linalg.inv(Omega_0)
nu_0 = 8
sigma_0 = 1

# Draw 50 draws from the prior of beta
sigma_sq = draw_chi2inv(nu_0-1, sigma_0**2, 30)
beta_prior = draw_mvnorm(mu_0, Omega_0_inv, sigma_sq)

# Check visually if the priors generate a reasonable regression curve
zero_to_one = np.linspace(0,1,100)
X_test = np.column_stack((np.ones(100), zero_to_one, zero_to_one**2))

# By plotting the data 
# together with the simulated regression curves from the prior
Y_prior = X_test@beta_prior
plt.plot(zero_to_one, Y_prior)
plt.scatter(data.time.values,Y)
plt.show
# Change the prior parameters until the results look reasonable

[mu_n, Omega_n, nu_n, sigma_n2] = post_params(X, Y, mu_0, Omega_0, sigma_0)
[beta_post, sigma2_post] = post_draws(mu_n, Omega_n, nu_n, sigma_n2, 10000)

# Histograms of the beta coefficients
fig, axs = plt.subplots(1,3, sharey = True)

for i in range(3):
    axs[i].hist(beta_post[0,:])

axs[0].set_title(r"$\beta_0$")
axs[1].set_title(r"$\beta_1$")
axs[2].set_title(r"$\beta_2$")

post_temp = np.zeros((365,10000))
for i in range(10000):
    post_temp[:,i] = X@beta_post[:,i] + np.sqrt(sigma2_post[i])*st.norm.rvs(1)

# Calculate the 2.5, 50 and 97.5 percentiles of the temperatures of each day
post_0025 = np.percentile(post_temp, 2.5,  axis=1)
post_0500 = np.percentile(post_temp, 50,   axis=1)
post_0975 = np.percentile(post_temp, 97.5, axis=1)

# Plot implied regression curves
plt.plot(X[:,1], post_0025)
plt.plot(X[:,1], post_0500)
plt.plot(X[:,1], post_0975)

plt.legend(["2.5% posterior percentile","Median", "97.5% posterior percentile"])
plt.xlabel("Time (year)")
plt.ylabel("Temperature (deg C)")

plt.scatter(data.time.values,Y)
plt.show

# Find the simulated distribution of the time with maximal temperature
# from the posterior distribution of beta
time_max = -0.5*beta_post[1]/beta_post[2]

# Plot the distribution of time_max
plt.hist(time_max)
plt.xlabel("Time (year) with the highest temperature")
plt.ylabel("Frequency in the simulated data")
