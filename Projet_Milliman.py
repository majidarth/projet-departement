#import torch as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import tensorflow as tf
from tensorflow import keras
import statsmodels.api as sm
import scipy

# Constantes du problème
d = 2
S0 = np.ones(d) * 100.
K = 100
r = 0.1
vol = 0.2
T = 5.
t = 0.1
rho = 0.5
gamma = rho*np.ones((d,d)) + (1-rho)* np.identity(d)
print(gamma)
deg = 5 #degree of polynomial regression

#Fonctions

def blackscholes_mc(t, T, n_paths, S0, vol, r, gamma, d):
    dt = T-t
    dW = np.sqrt(dt)*np.random.multivariate_normal(np.zeros(d), np.identity(d), n_paths).T
    paths = S0[:, np.newaxis]* np.exp((r-1/2*vol**2)*dt + vol*np.dot(np.linalg.cholesky(gamma),dW))
    return paths.T

def sigma_barre(sigma,T,t):
    return np.sqrt(np.matmul(np.transpose(sigma*np.ones(d)), np.matmul(gamma,sigma*np.ones(d)))/(d**2))

def F(t,S,sigma,T,r):
    if len(S.shape) == 1:
        return np.exp(np.mean(np.log(S) + (r - pow(sigma,2)/2)*(T-t)))*np.exp(sigma**2/2*(T-t))
    else:
        return np.exp(np.mean(np.log(S) + (r - pow(sigma,2)/2)*(T-t),axis =1))*np.exp(sigma_barre(sigma,T,t)**2/2*(T-t))
N = norm.cdf

def Call_BS(S,K,T,t,r,sigma):
  d1 = (np.log(S/K) + (r + sigma**2/2)*(T-t)) / (sigma*np.sqrt(T-t))
  d2 = d1 - sigma * np.sqrt(T-t)
  return S * N(d1) - K * np.exp(-r*(T-t)) * N(d2)


def nested_mc_expect(t, T, vol, r, gamma, d, K, nnested, S_t):
    dt = T-t
    dW = np.random.randn(len(S_t), d, nnested)*np.sqrt(dt)
    root_gamma = np.linalg.cholesky(gamma)
    S_t = S_t.T
    S_T = S_t[:, :, np.newaxis]* np.exp((r-1/2*vol**2)*dt + vol*np.dot(root_gamma,dW))
    payoff = np.exp(-r*(T-t)) * np.maximum(np.prod(S_T, axis = 0 )**(1/d) - K, 0)
    res = 1/nnested * payoff.sum(axis = 1)
    return res

def polynomial_reg(t, T , S0, r, gamma, vol,  d, K, n_paths, nnested, deg):
    S_t = blackscholes_mc(0, t, n_paths, S0, vol, r, gamma, d)
    V_t = nested_mc_expect(t, T, vol, r, gamma, d, K, nnested, S_t)
    
    poly = PolynomialFeatures(degree=deg)
    poly_variables = poly.fit_transform(S_t)

    regression = linear_model.LinearRegression()

    model = regression.fit(poly_variables, V_t)
    return model

def deePL_reg(t, T , S0, r, gamma, vol,  d, K, n_paths):
    S_t = blackscholes_mc(0, t, n_paths, S0, vol, r, gamma, d)
    V_t = nested_mc_expect(t, T, vol, r, gamma, d, K, 1, S_t)
    
    X_train_full = S_t
    
    #normalize input
    mX = np.mean(X_train_full)
    sX = np.std(X_train_full)
    X_train_full = ((X_train_full - mX) / sX)
    Y_train_full = (V_t)

    # split the dataset to a training and a validation set
    train_size = int(len(X_train_full)*0.75)
    X_train = X_train_full[:train_size]
    X_valid = X_train_full[train_size:]
    Y_train = Y_train_full[:train_size]
    Y_valid = Y_train_full[train_size:]

    model = keras.models.Sequential([
      keras.layers.Dense(20, activation='relu', input_shape=[d]),
      keras.layers.Dense(20, activation='relu'),
      keras.layers.Dense(20, activation='relu'),
      keras.layers.Dense(1)])
    model.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(learning_rate=0.001))
    
    #training
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=4, min_delta=1e-5, restore_best_weights=True)
    model.fit(X_train, Y_train, epochs=50, batch_size=128, validation_data=(X_valid, Y_valid), verbose=True, callbacks=[early_stopping_cb])
    return model, mX, sX
    

def sanity_check():
    S_t = blackscholes_mc( 0, t, 10000, S0, vol, r, gamma, d)
    true_value = Call_BS( F(t,S_t,vol,T,r), K, T, 0, 0, sigma_barre(vol,T,t)) * np.exp(-r*(T-t))
    expected_value = nested_mc_expect(t, T, vol, r, gamma, d, K, 1000, S_t)
    plt.hist([true_value,expected_value] , range = (np.min(true_value), np.max(true_value)), bins = 50, color = ['yellow', 'blue'],edgecolor = 'red')
    plt.title("Histogramme valeur. True value (en jaune) et Monte Carlo (en bleu)")
    plt.show()
    
    H,X1 = np.histogram( true_value, bins = 100, normed = True )
    dx = X1[1] - X1[0]
    F1 = np.cumsum(H)*dx
    
    H,X1 = np.histogram( expected_value, bins = 100, normed = True )
    dx = X1[1] - X1[0]
    F2 = np.cumsum(H)*dx
    
    plt.scatter(F1, F2)
    plt.plot( np.linspace(0,1,100), np.linspace(0,1,100), color = 'c')
    plt.title('Q-Q plot')
    plt.show()

    
if __name__ == '__main__':
    
    sanity_check()
    
    S_t = blackscholes_mc( 0, t, 1000000, S0, vol, r, gamma, d)
    true_value = Call_BS( F(t,S_t,vol,T,r), K, T, 0, 0, sigma_barre(vol,T,t)) * np.exp(-r*(T-t))
    
    #polynomial regression
    npaths = 1000
    nnested = 1000
    model_poly = polynomial_reg(0.5, 1 , S0, r, gamma, vol, d, K, npaths, nnested,deg)
    poly = PolynomialFeatures(degree=deg)
    S_t_ = poly.fit_transform(S_t)
    poly_value = model_poly.predict(S_t_)
    
    plt.scatter(poly_value, true_value)
    x = np.linspace(0,100,10000)
    plt.plot(x,x, 'r')
    plt.xlabel("Polynomial etimation of V_t")
    plt.ylabel("True of V_t according to BS model")
    plt.show()