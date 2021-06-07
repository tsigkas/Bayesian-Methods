""" Useful packages """
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st
from scipy.optimize import minimize as fmin

np.random.seed(1234) # set seed for reproducability

""" Helper functions """
# Returns the log posterior of the logistic function
def log_posterior(beta, y, X, mu, Cov):
    log_prior = np.log( st.multivariate_normal.pdf(beta, mean=mu, cov=Cov) )
    regr = X@beta
    logL = np.sum( regr * y - np.log(1+np.exp(regr) ) )
    return (logL + log_prior)

# Returns the posterior distribution of the probability that a particular sample works
def posterior_prob(x, mu, Cov, num):
    beta = st.multivariate_normal.rvs(mean = mu, cov = Cov, size = num)
    return st.logistic.cdf( np.matmul( x , np.transpose( beta ) ) )

# Returns the posterior distribution of the number of samples that work
def posterior_num(x, mu, Cov, num):
    n = len(x)
    prob = posterior_prob(x, mu, Cov, num)
    mult_pred  = np.zeros(num)
    for i, p in enumerate(prob):
        mult_pred[i] = st.binom.rvs(n, p)
    return mult_pred

""" Import the data """
data = pd.read_csv("WomenWork.dat", sep = "   ")

# Binary response variable
y = data.Work.values

# Covariates
X = data.iloc[:, 1:].values
    
# Prior parameters
mu_0 = np.zeros(8)
Cov_0 = 100*np.eye(8)

# Define the log posterior as a function of beta only, for the optimization
# Also, multiply with -1, as we seek the maximum 
# and the algorithm performs minimization
def obj_fun(beta):
    return -log_posterior(beta, y, X, mu_0, Cov_0)

# Find the max of the log posterior using a quasi-Newton method
# Which also returns a numerical approximation of H^-1
x0 = 1*np.ones(8) # Initial guess
opt_res = fmin(obj_fun, x0, method = "BFGS", options={'disp' : True})

# The inverse of J, no minus sign 
# as the objective function already has "switched signs"
J_inv = opt_res["hess_inv"]
# The mode, i.e. the variables at optimum
post_mode = opt_res["x"]

# Extract parameters for the coefficient of "NSmallChild", 
# i.e. the second to last covariate
mu_SC  = post_mode[-2]
sd_SC = J_inv[-2,-2]**0.5

eq_tail_CI_SC =  [mu_SC-st.norm.ppf(0.925)*sd_SC, mu_SC+st.norm.ppf(0.975)*sd_SC]

""" Data for a woman with:
- a husband with income 13
- 8 years of education
- 11 years of work experience
- 37 years of age
- 2 small children (under the age of 7)
"""
x_1w = np.array([1, 13, 8, 11, (11.0/10.0)**2, 37, 0, 0])

# The posterior probability that this particular woman works
post_prob = posterior_prob(x_1w, post_mode, J_inv, 20000)

plt.hist(post_prob)
plt.xlabel("Probability that the woman works")
plt.ylabel("Frequency")
plt.savefig("work_prob_hist.png", dpi = 1500)
plt.show

# The posterior distribution of the nunmber of women working
post_num = posterior_num(x_1w, post_mode, J_inv, 1000)

plt.hist(post_num, range=(0,9))
plt.xlabel("Number of women that work")
plt.ylabel("Frequency")
plt.savefig("work_num_hist.png", dpi = 1500)
plt.show

