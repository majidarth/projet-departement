#import torch as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import tensorflow as tf
from tensorflow import keras

# Constantes du problème
d = 3
S0 = np.ones(d) * 100.
r = 0.1
vol = 0.2
T = 1.
t = 0.5
rho = 0.5
gamma = rho*np.ones((d,d)) + (1-rho)* np.identity(d)
deg = 4 #degree of polynomial regression

#Fonctions

def blackscholes_mc(t, T, n_paths, S0, vol, r, gammma, d):
    dt = T-t
    dW = np.sqrt(dt)*np.random.multivariate_normal(np.zeros(d), np.identity(d), n_paths).T
    paths = S0[:, np.newaxis]* np.exp((r-1/2*vol**2)*dt + vol*np.dot(np.linalg.cholesky(gamma),dW))
    return paths.T

def sigma_barre(sigma,T,t):
    return np.sqrt((T-t)*np.matmul(np.transpose(sigma*np.ones(d)), np.matmul(gamma,sigma*np.ones(d)))/(d**2))

def F(t,S,sigma,T,r):
    if len(S.shape) == 1:
        return np.exp(np.mean(np.log(S) + (r - pow(sigma,2)/2)*(T-t))) * np.exp(-sigma/2*(T-t))
    else:
        return np.exp(np.mean(np.log(S) + (r - pow(sigma,2)/2)*(T-t),axis =1)) * np.exp(-sigma/2*(T-t))

N = norm.cdf

def Call_BS(S,K,T,t,r,sigma):
  d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)
  return S * N(d1) - K * np.exp(-r*(T-t)) * N(d2)


def nested_mc_expect(t, T, vol, r, gamma, d, K, nnested, S_t):
    dt = T-t
    dW = np.random.randn(len(S_t), d, nnested)*np.sqrt(dt) * np.sqrt(dt)
    root_gamma = np.linalg.cholesky(gamma)
    coeff = np.exp((r-1/2*vol**2)*dt + vol*np.dot(root_gamma,dW))
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
    
    
if __name__ == '__main__':
    S_t = blackscholes_mc( 0, 0.5, 100000, S0, vol, r, gamma, d)
    true_value = Call_BS( F(t,S_t,vol,T,r), 100, T, t, r, sigma_barre(vol,T,t))
    
    #polynomial regression
    print("hello")
    model_poly = polynomial_reg(0.5, 1 , S0, r, gamma, vol, d, 100, 100000, 100,deg)
    poly = PolynomialFeatures(degree=deg)
    S_t_ = poly.fit_transform(S_t)
    poly_value = model_poly.predict(S_t_)
    
    plt.scatter(poly_value, true_value)
    x = np.linspace(0,100,10000)
    plt.plot(x,x, 'r')
    plt.xlabel("Polynomial etimation of V_t")
    plt.ylabel("True of V_t according to BS model")
    plt.show()
    
    #deepl
    model_DL, mean, std = deePL_reg(0.5, 1 , S0, r, gamma, vol, d, 100, 100000)
    deepl_value = model_DL.predict((S_t - mean) / std)
    plt.scatter(deepl_value, true_value)
    x = np.linspace(0,100,10000)
    plt.plot(x,x, 'r')
    plt.xlabel("Deepl etimation of V_t")
    plt.ylabel("True of V_t according to BS model")
        

