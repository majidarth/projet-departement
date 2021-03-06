#import torch as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFE
from sklearn import linear_model
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
import statsmodels.api as sm
import scipy
import seaborn as sns

# Constantes du problème
d = 3
S0 = np.ones(d) * 100.
K = 100
r = 0.1
vol = 0.2
T = 1.
t = .5
rho = 0.5
gamma = rho*np.ones((d,d)) + (1-rho)* np.identity(d)
deg = 10 #degree of polynomial regression


class MyHyperModel(kt.HyperModel):
  def build(self, hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_of_layers',2,15)):         
        #providing range for number of neurons in hidden layers
        model.add(keras.layers.Dense(units=hp.Int('num_of_neurons'+ str(i),min_value=8,max_value=256,step=32),
                                    activation='relu'))
    model.add(keras.layers.Dense(1, activation='relu'))
    model.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(learning_rate=0.001))
    return model


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
        return np.exp(np.mean(np.log(S) + (r - pow(sigma,2)/2)*(T-t)))*np.exp(sigma_barre(sigma,T,t)**2/2*(T-t))
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

def polynomial_reg_sfs(t, T , S0, r, gamma, vol,  d, K, n_paths, nnested, deg):
    S_t = blackscholes_mc(0, t, n_paths, S0, vol, r, gamma, d)
    V_t = nested_mc_expect(t, T, vol, r, gamma, d, K, nnested, S_t)
    
    poly = PolynomialFeatures(degree=deg)
    poly_variables = poly.fit_transform(S_t)

    regression = linear_model.LinearRegression()
    
    
    sfs = RFE(regression)
    sfs = sfs.fit(poly_variables, V_t)
    
    
    poly_variables_sfs = sfs.transform(poly_variables)
    print(poly_variables.shape, "kglugvjhvhjv")
    model = regression.fit(poly_variables_sfs, V_t)
    return model, sfs


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
    
def deePL_reg_tune(t, T , S0, r, gamma, vol,  d, K, n_paths):
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

    early_stopping_cb = keras.callbacks.EarlyStopping(patience=4, min_delta=1e-5, restore_best_weights=True)

    tuner = kt.RandomSearch(MyHyperModel(),objective='val_loss',max_trials=5, overwrite=True)
    tuner.search(X_train, Y_train, epochs=50, batch_size=512, validation_data=(X_valid, Y_valid), callbacks=[early_stopping_cb], verbose=True)
    return tuner.get_best_models()[0], mX, sX


def sanity_check():
    S_t = blackscholes_mc( 0, t, 10000, S0, vol, r, gamma, d)
    true_value = Call_BS( F(t,S_t,vol,T,r), K, T, t, 0, sigma_barre(vol,T,t)) * np.exp(-r*(T-t))
    expected_value = nested_mc_expect(t, T, vol, r, gamma, d, K, 100, S_t)
    plt.hist([true_value,expected_value] , range = (min(np.min(true_value),np.min(expected_value)) , max(np.max(true_value),np.max(expected_value))), bins = 50, color = ['yellow', 'blue'],edgecolor = 'red')
    plt.title("Histogramme valeur. True value (en jaune) et Monte Carlo (en bleu)")
    plt.show()
    
    H,X1 = np.histogram( true_value, bins = 100, normed = True )
    dx = X1[1] - X1[0]
    F1 = np.cumsum(H)*dx
    
    H,X2 = np.histogram( expected_value, bins = 100, normed = True )
    dx = X2[1] - X2[0]
    F2 = np.cumsum(H)*dx
    
    plt.scatter(F1, F2)
    plt.plot( np.linspace(0,1,100), np.linspace(0,1,100), color = 'c')
    plt.title('Q-Q plot')
    plt.show()
    
    sns.kdeplot(expected_value)
    sns.kdeplot(true_value)
    plt.show()
    
if __name__ == '__main__':
    
    #sanity_check()
    
    S_t = blackscholes_mc( 0, t, 100000, S0, vol, r, gamma, d)
    true_value = Call_BS( F(t,S_t,vol,T,r), K, T, t, 0, sigma_barre(vol,T,t)) * np.exp(-r*(T-t))
    
    #polynomial regression
    # npaths = 10000
    # nnested = 100
    # model_poly = polynomial_reg(0.5, 1 , S0, r, gamma, vol, d, K, npaths, nnested,deg)
    # poly = PolynomialFeatures(degree=deg)
    # S_t_ = poly.fit_transform(S_t)
    # poly_value = model_poly.predict(S_t_)
    
    # plt.scatter(poly_value, true_value)
    # x = np.linspace(0,100,10000)
    # plt.plot(x,x, 'r')
    # plt.xlabel("Polynomial etimation of V_t")
    # plt.ylabel("True of V_t according to BS model")
    #plt.show()
    
    #polynomial regression sfs
    # npaths = 1000
    # nnested = 10
    # model_poly, sfs = polynomial_reg_sfs(0.5, 1 , S0, r, gamma, vol, d, K, npaths, nnested,deg)
    # poly = PolynomialFeatures(degree=deg)
    # S_t_ = poly.fit_transform(S_t)
    # S_t_ = S_t_[:,sfs.get_support(indices=True)]
    # poly_value = model_poly.predict(S_t_)
    
    # plt.scatter(poly_value, true_value)
    # x = np.linspace(0,100,10000)
    # plt.plot(x,x, 'r')
    # plt.xlabel("Polynomial etimation of V_t")
    # plt.ylabel("True of V_t according to BS model")
    # plt.show()
    
    #deepl
    #npaths = 10000
    #model_DL, mean, std = deePL_reg(0.5, 1 , S0, r, gamma, vol, d, K, npaths)
    #deepl_value = model_DL((S_t - mean) / std).numpy()
    #plt.scatter(deepl_value, true_value)
    #x = np.linspace(0,100,10000)
    #plt.plot(x,x, 'r')
    #plt.xlabel("Deepl etimation of V_t")
    #plt.ylabel("True value of V_t according to BS model")
    #plt.show()
    
    #sns.kdeplot(true_value)
    #sns.kdeplot(deepl_value.T[0])
    #plt.show()

    #deepl tune
    npaths = 500000
    model_DL, mean, std = deePL_reg_tune(0.5, 1 , S0, r, gamma, vol, d, K, npaths)
    deepl_value = model_DL((S_t - mean) / std).numpy()
    plt.scatter(deepl_value, true_value)
    x = np.linspace(0,100,10000)
    plt.plot(x,x, 'r')
    plt.xlabel("Deepl etimation of V_t")
    plt.ylabel("True of V_t according to BS model")
    plt.show()
    
    print(model_DL.summary())
    
    sns.kdeplot(true_value)
    sns.kdeplot(deepl_value.T[0])
    plt.show()
    
    #print(f"Ecart relatif entre deepl et BS: {np.linalg.norm((deepl_value.T) - true_value)/ np.linalg.norm(true_value):.5f}")
    #print(f"Ecart relatif entre poly et BS: {np.linalg.norm(poly_value - true_value)/ np.linalg.norm(true_value):.5f}")
    
    #Calcul de la valeur initiale du portefeuille :
    V0 = Call_BS(F(0,S0,vol,T,r), K, T, 0, 0, sigma_barre(vol,T,0))*np.exp(-r*T)
    print(V0)
    
    #Calcul de la VaR:
    print("Value at risk de 5% mc:",np.quantile(V0-true_value, 0.95))
    
    varg = np.quantile(np.random.normal(size=1000000),0.95)

    h = F(0,S0,vol,t,r)*np.exp(sigma_barre(vol,t,0)**2/2*(T-2*t)+sigma_barre(vol,t,0)*np.sqrt(t)*(-varg) + (r-vol**2/2)*(T-t))
    Vt = np.exp(-r*(T-t))*Call_BS(h, K, T, t, 0, sigma_barre(vol, T, t))
    
    print("Value at risk de 5% formule fermée:",V0-Vt)
        
    x = [i/1000 for i in range(1001)]
    #plt.plot(x,np.quantile(V0-true_value,x))
    g = np.quantile(np.random.normal(size=1000000),x)
    
    
    plt.plot(x,[V0 - np.exp(-r*(T-t))*Call_BS(F(0,S0,vol,t,r)*np.exp(sigma_barre(vol,t,0)**2/2*(T-2*t)+sigma_barre(vol,t,0)*np.sqrt(t)*(-varg) + (r-vol**2/2)*(T-t)), K, T, t, 0, sigma_barre(vol, T, t)) for varg in g])
    plt.title(r'Value at Risk de niveau $1-\alpha$')
    plt.xlabel(r'$1-\alpha$')
    plt.ylabel(r'VaR($1-\alpha$)')
    plt.show()