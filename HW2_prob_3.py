import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
print "\n"

def gaussian(mu, sigma, k):
    fn = (1.0/np.sqrt(2.0*np.pi*(sigma**2.0))) * np.exp(-0.5*((k-mu)/sigma)**2.0)
    return fn 

def fit_gaussian(distrib, k):

    # Initial guess:
    guess_sigma=0.25*np.max(k)
    guess_mu=0.5*np.max(k)

    # Optimize:
    optimize_func = lambda x: gaussian(x[0], x[1], k) - distrib
    
    est_mu, est_sigma = leastsq(optimize_func, [guess_mu, guess_sigma])[0]

    sub_k = np.linspace(np.min(k),np.max(k),num=(10*len(k)))
    data_fit = gaussian(est_mu, abs(est_sigma), sub_k)

    return (sub_k, data_fit, est_mu, abs(est_sigma))

from scipy.stats import poisson
from scipy.stats import binom

p = 0.35
N = 40.0

k=np.arange(30) #1 #successes
mu = N*p # mean

# Part A:
plt.plot(k, poisson.pmf(k, 0.35*4),marker="o",ls="--",label="Poisson  N=4")
plt.plot(k, binom.pmf(k, 4, p), marker="d",ms=7,ls="--",label="Binomial N=4")
plt.plot(k, poisson.pmf(k, mu),marker="o",ls="--",label="Poisson  N=40")
plt.plot(k, binom.pmf(k, 40, p), marker="d",ms=7,ls="--",label="Binomial N=40")

plt.xlabel("k successes",fontsize=18)
plt.ylabel("P( k successes | N trials )",fontsize=18)
plt.legend(frameon=False,fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# Part B:
plt.subplot(2,2,1)
plt.plot(k, poisson.pmf(k, 0.35*4),marker="o",ls="--",label="Poisson  N=4\n")
sub_k, best_fit, g_mu, g_sigma = fit_gaussian(poisson.pmf(k, 0.35*4), k)
plt.plot(sub_k, best_fit, ls="-",label="Gaussian fit:\nmu="+str(np.round(g_mu,1))+"\nsigma="+str(np.round(g_sigma,1)))

plt.xlabel("k successes",fontsize=14)
plt.ylabel("P( k successes | N trials )",fontsize=14)
plt.legend(frameon=False,fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.subplot(2,2,2)
plt.plot(k, poisson.pmf(k, 0.35*40),marker="o",ls="--",label="Poisson  N=40\n")
sub_k, best_fit, g_mu, g_sigma = fit_gaussian(poisson.pmf(k, 0.35*40), k)
plt.plot(sub_k, best_fit, ls="-",label="Gaussian fit:\nmu="+str(np.round(g_mu,1))+"\nsigma="+str(np.round(g_sigma,1)))

plt.xlabel("k successes",fontsize=14)
plt.ylabel("P( k successes | N trials )",fontsize=14)
plt.legend(frameon=False,fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.subplot(2,2,3)
plt.plot(k, binom.pmf(k,4,0.35),marker="o",ls="--",label="Binomial N=4\n")
sub_k, best_fit, g_mu, g_sigma = fit_gaussian(binom.pmf(k, 4, 0.35), k)
plt.plot(sub_k, best_fit, ls="-",label="Gaussian fit:\nmu="+str(np.round(g_mu,1))+"\nsigma="+str(np.round(g_sigma,1)))

plt.xlabel("k successes",fontsize=14)
plt.ylabel("P( k successes | N trials )",fontsize=14)
plt.legend(frameon=False,fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.subplot(2,2,4)
plt.plot(k, binom.pmf(k, 40, 0.35),marker="o",ls="--",label="Binomial N=40\n")
sub_k, best_fit, g_mu, g_sigma = fit_gaussian(binom.pmf(k, 40, 0.35), k)
plt.plot(sub_k, best_fit, ls="-",label="Gaussian fit:\nmu="+str(np.round(g_mu,1))+"\nsigma="+str(np.round(g_sigma,1)))

plt.xlabel("k successes",fontsize=14)
plt.ylabel("P( k successes | N trials )",fontsize=14)
plt.legend(frameon=False,fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

#plt.tight_layout()
plt.show()

