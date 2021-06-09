""" Some useful packages"""
import pandas as pd
import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import minimize as fmin

np.random.seed(1111) # set seed for reproducability

""" Helper functions """
# An expression proportional to the log of the posterior, minus the terms
# that are not dependent on beta,
# to be used for the numerical optimization and the MRW
def log_posterior(beta, X, y):
    regr = X@beta
    logL = np.sum(-np.exp(regr) + y*regr)
    prior = np.log(st.multivariate_normal.pdf(beta, mean=np.zeros_like(beta), cov = 100*np.linalg.inv(X.T@X)))
    return (logL + prior)

class RWM_res:
    def __init__(self, init, samples):
        self.initial = init
        self.iterations = samples
        self.theta = np.zeros((len(init), samples+1))
        self.theta[:,0] = self.initial # Initialize
        self.acc_p = 0.0 # Keep track of the acceptance probability
    def accept(self, theta_p, it):
        self.acc_p += 1.0/(self.theta.shape[1])
        self.theta[:,it] = theta_p
    def reject(self, it):
        self.theta[:,it] = self.theta[:,it-1]
    def draw(self, it, Cov):
        return st.multivariate_normal.rvs(mean=self.theta[:,it-1], cov=Cov)
    def hat(self, burn_in):
        return np.mean(self.theta[:,burn_in:], axis=1)
    def plot(self, x, y):
        order = np.reshape(np.arange(0,x*y),(x,y))
        fig, axs = plt.subplots(x,y, figsize = (8,8))
        for i in range(x):
            for j in range(y):
                axs[i,j].plot(self.theta[order[i,j],:], linewidth=0.5)
                axs[i,j].set_title(r"$\theta_%i$" %(order[i,j]+1))
            fig.tight_layout()
            plt.show
    def iterate(self, log_post_fun, X, y, Cov, c):
        for i in range(1,self.iterations+1):    
            theta_p = self.draw(i, c*Cov)
            # Calculate the probability of acceptance
            post_ratio = np.exp(log_post_fun(theta_p, X, y) - log_post_fun(self.theta[:,i-1], X, y))
            # Draw a random sample from a Bern(alfa) distribution
            # To determine if the iteration should be accepted or not
            alfa = np.nanmin([1, post_ratio])
            if st.bernoulli.rvs(alfa):
                self.accept(theta_p, i)
            else:
                self.reject(i)

""" Import and transform the data """
data = pd.read_csv('eBayNumberOfBidderData.csv', sep = ";")

y = data.nBids.values # Response variable
X = data.iloc[:, 1:].values # Covariates

"""Fit a generalized linear model using the statsmodel package"""
exog, endog = X, y
model = sm.GLM(endog, exog, 
               family = sm.families.Poisson(link=sm.families.links.log))

poisson_fit = model.fit()
print(poisson_fit.summary())

"""Find the normal approximation of the posterior, to get the inverse hessian"""
# Define the log posterior as a function of beta only, for the optimization
# Also, multiply with -1, as we seek the maximum
# and the algorithm performs minimization
def obj_fun(beta):
    return -1*log_posterior(beta, X, y)

# Find the max of the log posterior using a quasi-Newton method
# Which also returns a numerical approximation of H^-1
x0 = np.zeros(9) # Initial guess
opt_res = fmin(obj_fun, x0, method="BFGS", options={"disp" : True})

# Numerical approximation of -H^-1
H_inv = opt_res["hess_inv"] 
# The mode, i.e. the beta vector at the optimum
post_mode = opt_res['x']

"""Draw from the actual posterior, 
using the Random Walk Metropolis algorithm"""
RWM_results = RWM_res(np.zeros(9),5000) # Initialize
RWM_results.iterate(log_posterior, X, y, H_inv, 0.6)

RWM_results.acc_p # Check the average acceptance probability

# Plot the marginal trajectories to check convergence visually
RWM_results.plot(3,3)
plt.savefig("MCMC_convergence.png", dpi=800)

# Compute the sample mean, eliminating the first 500 iterations
beta_hat = RWM_results.hat(500)

# Using the iterations to sample from the posterior predictive density
beta_draws = RWM_results.theta[:,501:]

# For the following auction:
"""
The x-vector for an auction with:
    A power seller,
    with verified ID.
    A sealed product with
    a major defect
    No major negative feedback
    A log book-value of 1
    A minimum bid of 0.7
"""    
x_sample = np.array([1,1,1,1,0,1,0,1,0.7])

y_pred = np.zeros(4500)
for i in range(len(y_pred)):
    lamda = np.exp(x_sample@beta_draws[:,i])
    y_pred[i] = st.poisson.rvs(1, lamda) 

# The posterior distribution of this sample
lamda_MLE = np.exp(x_sample@poisson_fit.params)
y_MLE = st.poisson.pmf(np.arange(0,10), lamda_MLE)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(np.arange(0,10), y_MLE, color = "b", align = "edge")
ax2.hist(y_pred, color="r", histtype = "step", density = True, align = "mid")

ax1.set_xlabel('Number of bidders')
ax1.set_ylabel('ML Estimate', color='b')
ax2.set_ylabel('Simulated posterior predictive density', color='r')
plt.show()

# The probability that the sample gets no bids
p0 = np.sum(y_pred == 0)/len(y_pred)