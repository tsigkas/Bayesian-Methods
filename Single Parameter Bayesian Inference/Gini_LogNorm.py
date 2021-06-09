from scipy import stats as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

np.random.seed(123) # set seed for reproducability

# Dataset of monthly income in KSEK/mo.
y = np.array([38, 20, 49, 58, 31, 70, 18, 56, 25, 78])
n = len(y)

# Known parameter of log-normal distribution
mean = 3.8

# Draw 10 000 random samples from the scaled inverse chi2
# by first drawing from a chi2 distribution with n-1 df's
tau2 = np.sum((np.log(y)-mean)**2)/n
X = st.chi2.rvs(n-1, size=10000)
var_sim = (n-1)*tau2/X

# Explore the distribution of the Gini-coefficient
# By fitting a gaussian density kernel to the data
G = 2*st.norm.cdf(np.sqrt(var_sim/2))-1
G_pdf = st.gaussian_kde(G) 

def G_cdf(x):
    return quad(G_pdf, 0, x)[0]

# Plot histogram from the simulated variances
plt.hist(G)
plt.title("Distribution of the Gini coefficient")
plt.xlabel("G")
plt.ylabel("frequency in simulated data")
plt.show

# Evaluate the pdf and cdf for a dense grid of values.
x = np.linspace(0,1,1000)
G_pdf_vals = G_pdf(x)
G_cdf_vals = np.zeros_like(x)

for i, vals in enumerate(x):
    G_cdf_vals[i] = G_cdf(vals)

# Generate 90% highest posterior density interval
sorted_index = np.argsort(G_pdf_vals)[::-1]
HPDI = [x[sorted_index[1]]]

# By adding points until they integrate to 0.9
for i in range(1,len(sorted_index)):
    if quad(G_pdf,min(HPDI), max(HPDI))[0] < 0.9:
        HPDI.append(x[sorted_index[i]])

HPDI = [min(HPDI), max(HPDI)]

# Generate 90% equal tail credible interval from the cdf values
# By having 5% of the mass to the left and to the right
lb_index = np.argmin(np.abs(G_cdf_vals-0.05))
ub_index = np.argmin(np.abs(G_cdf_vals-0.95))

eq_tail_CI = [x[lb_index], x[ub_index]]

# Plot density
# With the equal tail CI and HPDI
plt.plot(x,G_pdf_vals)
plt.title("Distribution of the Gini coefficient")
plt.xlabel("G")
plt.vlines(eq_tail_CI[0], 0, G_pdf(eq_tail_CI[0]))
plt.vlines(eq_tail_CI[1], 0, G_pdf(eq_tail_CI[1]))
plt.vlines(HPDI[0], 0, G_pdf(HPDI[0]), color = "red")
plt.vlines(HPDI[1], 0, G_pdf(HPDI[1]), color = "red")
plt.show
