import torch as nn
import numpy as np
import matplotlib as plt
from scipy.stats import norm

# Constantes du problème
d = 10
S0 = np.ones(d) * 100.
r = 0.1
vol = 0.2
T = 1
rho = 0.5
gamma = rho*np.ones((d,d)) + (1-rho)* np.identity(d)

#Fonctions

def blackscholes_mc(t, T, n_paths, S0, vol, r, gammma):
    dt = T-t
    G = np.random.randn(n_paths)
    paths = S0 * np.exp((r-1/2*vol**2)*dt + vol*np.dot( np.linalg.cholesky(gamma),G))
    return paths

N = norm.cdf

def Call_BS(S,K,T,t,r,sigma):
  d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)
  return S * N(d1) - K * np.exp(-r*(T-t)) * N(d2)

if __name__ == '__main__':
    blackscholes_mc(0,1,100, S0, vol, r, gamma)
