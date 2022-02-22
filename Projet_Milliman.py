#import torch as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Constantes du problème
d = 3
S0 = np.ones(d) * 100.
r = 0.1
vol = 0.2
T = 1
rho = 0.5
gamma = rho*np.ones((d,d)) + (1-rho)* np.identity(d)

#Fonctions

def blackscholes_mc(t, T, n_paths, S0, vol, r, gammma, d):
    dt = T-t
    dW = np.sqrt(dt)*np.random.multivariate_normal(np.zeros(d), np.identity(d), n_paths).T
    paths = np.multiply(S0,(np.exp((r-1/2*vol**2)*dt + vol*np.dot(np.linalg.cholesky(gamma),dW))).T)
    return paths

def sigma_barre(sigma,T,t):
    return np.sqrt((T-t)*np.matmul(np.transpose(sigma), np.matmul(gamma,sigma))/(d**2))

def F(t,S,sigma,T,r):
    return np.exp(np.mean(np.log(S) + (r - pow(sigma,2)/2)*(T-t))) * np.exp(-np.matmul(np.transpose(sigma), sigma)/2*(T-t))

N = norm.cdf

def Call_BS(S,K,T,t,r,sigma):
  d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)
  return S * N(d1) - K * np.exp(-r*(T-t)) * N(d2)

def polynomial_reg(t, T , S0, r, gamma, d, K, n_paths, vol):
    S_t = blackscholes_mc(0, t, n_paths, S0, vol, r, gamma, d)
    S_T = np.zeros_like(S_t)
    for i in range (len(S_t)):
        S_T[i] = blackscholes_mc(t, T, 1, S_t[i], vol, r, gamma, d)[0]
    V_t = np.exp(-r*(T-t)) * np.maximum(np.prod(S_T, axis = 1)**(1/d)-K,0)

    p = np.polyfit(S_t, V_t, deg =2)
    return p
    
    
if __name__ == '__main__':
    print(polynomial_reg(0.5, T , S0, r, gamma, d, 100, 10, vol))
    
