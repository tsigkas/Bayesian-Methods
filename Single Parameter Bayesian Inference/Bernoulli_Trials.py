from scipy import stats as st
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123) # set seed for reproducability

# Parameters for beta prior distribution
alpha0 = 3
beta0  = 3

n = 24 # number of samples from Bernoulli trial data
s = 8 # number of sucesses in sample 
f = n - s # number of failiures

# Analytical mean and standard deviation for posterior
mean, var = st.beta.stats(alpha0+s, beta0+f, moments="mv")
std = np.sqrt(var)

# Draw 10 000 random samples from posterior
post_sim = st.beta.rvs(alpha0+s, beta0+f, size=10000)

# Check convergence of mean, standard deviation
mean_sim = np.zeros(1000)
std_sim  = np.zeros(1000)

for i in range(1000):
    mean_sim[i] = np.mean(post_sim[0:i])
    std_sim[i]  = np.std(post_sim[0:i])
    
plt.plot(mean_sim)
plt.plot(std_sim)
plt.legend(["sample mean","sample standard deviation"])
plt.xlabel("number of samples")
plt.savefig("mean_sd_convergence.png", dpi=1500)
plt.show

#Compare simulated probability to exact posterior
geq04 = post_sim >= 0.4
# ratio of number of samples >= 0.4 and total samples
p_sim = float(sum(geq04))/float(len(post_sim))
p_true = 1 - st.beta.cdf(0.4, alpha0 + s, beta0 + f)

# Explore the distribution of the log-odds 
logodds_sim = np.log(post_sim/(1-post_sim))
plt.hist(logodds_sim)
plt.title("Distribution of the log-odds")
plt.xlabel("log-odds")
plt.ylabel("frequency in simulated data")
plt.savefig("logodds_hist", dpi = 1500)
plt.show
