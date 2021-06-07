import numpy as np
import matplotlib.pyplot as plt
from scipy.special import iv as bessel
from scipy.integrate import quad

np.random.seed(123) # set seed for reproducability

# Dataset of wind directions in radians
y = np.array([-2.44, 2.14, 2.54, 1.83, 2.02, 2.33, -2.79, 2.23, 2.07, 2.02])

mu    = 2.39
lamda = 1

# Function proportional to the posterior distribution
# Given von-Mises(kappa,mu) likelihood and Exp(lamda) prior
def posterior(kappa, y, lamda, mu):
    n = len(y)
    return (lamda/bessel(0,kappa))**n*np.exp(kappa*(np.sum(np.cos(y-mu))-lamda))

# Find posterior for different values of kappa
kappa = np.linspace(0.01,5,10000)
posterior_kappa = np.zeros_like(kappa)

posterior_kappa = posterior(kappa, y, lamda, mu)

# Normalize the posterior so that it integrates to 1
posterior_kappa = posterior_kappa/(quad(posterior, 0.01, 5, args = (y, lamda, mu))[0])

# Find the posterior mode from the density
mode = kappa[np.argmax(posterior_kappa)]

# Plot posterior
plt.plot(kappa, posterior_kappa)
plt.xlabel(r"$\kappa$")
plt.ylabel("posterior density")
plt.vlines(mode,0,np.max(posterior_kappa))
plt.savefig("kappa_posterior_pdf.png", dpi=1500)
plt.show