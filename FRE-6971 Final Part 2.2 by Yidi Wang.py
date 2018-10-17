
# coding: utf-8

# FRE-6971 Final Part 2 5/14/2018 by Yidi Wang Hope you enjoy my final project. :)

# Problem 2

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import scipy.optimize as opt
import math as math
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# 2.1 Estimation & residuals.
# I choose the 1-Factor Vasicek Model.
# dr(t) = (mu-k*r(t))dt + vol*dW(t)
def mean_r(r0, p, t, dist='Q'):
    mu,k,vol = p
    m = r0 * np.exp(-k*t) + mu / k * (1 - np.exp(-k * t))
    if (dist == 'F'):
        m -= (vol ** 2) * pow(t,2)
    return m
def A(p, tau):
    mu,k,vol = p
    return (mu/k - vol**2/(2*pow(k,2))) * (B(tau,k)-tau) - vol**2/(4*k)*B(tau,k)
def B(tau, k):
    return (1-np.exp(-k*tau))/k
def zero(t0, T, r_t0):
    tau = T - t0
    return np.exp(A(tau) - B(tau)*r_t0)


# In[3]:


def fwd_rate(r0, p, t_fix, T0, T1, dist = 'Q', acrual = 0.25):
    mu, k, vol = p
    A_diff = A(p, T0-t_fix) - A(p, T1-t_fix)
    B_diff = B(T0-t_fix, k) - B(T1-t_fix, k)
    m = A_diff - B_diff * mean_r(r0,p,t_fix,dist)
    v = vol ** 2 * t_fix * pow(B_diff, 2)
    e = np.exp(m + v/2)
    return 1/acrual * (e-1)


# In[4]:


# Use the EuroDollar Futures Rate to fit the model.
df = pd.read_csv('D:/FIproject/Constant_Maturity_ED.csv')
df = df.set_index(['Date'])
df.index = pd.to_datetime(df.index)
sample_A = df[(df.index >= '2011-1-1') & (df.index <= '2015-1-1')]
df_rates = df[df.index == '2011-1-3']


# In[5]:


df_rates


# In[6]:


x = range(1,21)
term = [xx/4. for xx in x]
y = df_rates.ix[0,:]
def error(p):
    yy = [fwd_rate(r0,p,t,t,t+0.25,'Q',acrual=0.25)*100 for t in term]
    return yy - y


# In[7]:


r0 = 0.01
p0 = (0.01, 0.01, 0.01)
para = opt.leastsq(error, p0)[0]
print('The parameters we get from the calibration: \n')
print('mu:      %.6f'%para[0])
print('k:       %.6f'%para[1])
print('sigma:  %.6f'%para[2])
print('1-Factor Vasicek Model: dr(t)=(%.6f-(%.6f*r(t))dt + %.6f*dW(t)'%(para[0],para[1],para[2]))


# In[8]:


plt.figure(figsize = (8,6))
plt.scatter(term, y, color = 'red', label = 'sample', linewidth = 3)
xx = np.linspace(0, 21/4., 1000)
yyy = [fwd_rate(r0, para, t, t, t+0.25, 'Q', acrual = 0.25)*100 for t in xx]
plt.plot(xx, yyy, color = 'blue', label = 'fit', linewidth = 3)
plt.show()


# In[9]:


# 2. On each iteration you should inspect if the problem becomes collinear.
# Inspect the Jacobian matrix.
para_j = opt.least_squares(error, p0).jac
print('Jacobian Matrixi is: \n',para_j)


# In[10]:


print('Correlation matrix of the Jacobian is :\n', pd.DataFrame(para_j).corr())


# In[11]:


# According to the above result that it really shows collinearity.
# 3. Use the PCA rank reduction to work with this problem.
coef = pd.DataFrame(para_j)
C = coef.cov()
evals, evecs = np.linalg.eig(C)
indices = np.argsort(evals)
indices = indices[::-1]
evals = evals[indices]
evecs = evecs[:,indices]
print('eigenvalue: \n')
print(evals)


# In[12]:


print('PCA parameters: \n')
print(evecs)


# In[13]:


# Use mu, k and vol to substitute the PCA_1.
print('PCA_1: %.6f*mu + %.6f*k + %.6f*vol'%(evecs[0,0],evecs[1,0],evecs[2,0]))


# In[14]:


# 4. Generate time series of the residuals (input futures date - model futures rate) 
#    for all 20 interploated futrues and study in-sample time series properties of the residuals.
#    Stationary, mean-reversion(half-life), volatitlity and shape of the distribution.


# In[15]:


# Calculate the overall residuals.
residual_list = []
for i in range(len(sample_A.index)):
    y = sample_A.iloc[i,:]
    yy = [fwd_rate(r0,para,t,t,t+0.25,'Q',acrual=0.25)*100 for t in term]
    residual = y - yy 
    residual_list.append(residual)


# In[16]:


# Define a function to generate the time-series residual from the data.
def generate_ts(residual,n):
    list_ED = []
    for i in range(992):
        ED = residual[i].iloc[n]
        list_ED.append(ED)
    return pd.DataFrame(list_ED,index=sample_A.index)


# In[17]:


ED1_residual = generate_ts(residual_list,0)[0]


# In[18]:


results = ts.adfuller(ED1_residual)
# Define the function to calculate the properties of the residuals:
# including stationarity, half-life and volatility.
def AR_1(data):
    data_lag = data.shift(1).dropna()
    df = data.drop(data.index[0]).dropna()
    X = data_lag.as_matrix()
    X_I = sm.add_constant(X)
    Y = df.as_matrix()
    l1 = sm.OLS(Y, X_I).fit()
    return l1
def HF(b):
    hf = math.log(0.5)/math.log(b)
    return hf


# In[19]:


# Define the function to show the properties.
def output(ED_residual):
    adf = ts.adfuller(ED_residual)
    adf_p = adf[1]
    hf = HF(AR_1(ED_residual).params[1])
    ED_std = ED_residual.std()
    return [adf_p,hf,ED_std]


# In[20]:


ED1_residual_ = output(ED1_residual)


# In[21]:


# Hihglight the results for 2y, 3y, 4y, 5y futures.
ED8_residual = generate_ts(residual_list,7)[0]
ED12_residual = generate_ts(residual_list,11)[0]
ED16_residual = generate_ts(residual_list,15)[0]
ED20_residual = generate_ts(residual_list,19)[0]


# In[22]:


def print_p(ED_residual):
    list_ = output(ED_residual)
    print('The ADF test p-value is: %.6f'%list_[0])
    print('The half-life is: %.6f'%list_[1])
    print('The standard error is: %.6f'%list_[2])


# In[23]:


# For 2 year.
print('The outcome of 2y Reiduals: \n')
print_p(ED8_residual)


# In[24]:


# For 3 year.
print('The outcome of 3y Reiduals: \n')
print_p(ED12_residual)


# In[25]:


# For 4 year.
print('The outcome of 4y Reiduals: \n')
print_p(ED16_residual)


# In[26]:


# For 5 year.
print('The outcome of 5y Reiduals: \n')
print_p(ED20_residual)


# In[27]:


# 2.2 Cointegrated pairs of residuals & signal analysis.
# 1. Construct cointegrated pairs of the residuals, using the combinations and weights determined in 1.1.
C = sample_A.cov()
evals, evecs = np.linalg.eig(C)
indices = np.argsort(evals)
indices = indices[::-1]
evecs = evecs[:,indices]
evals = evals[indices]
w = [evecs[7,0]/evecs[11,0], evecs[11,0]/evecs[15,0], evecs[15,0]/evecs[19,0],
     evecs[7,0]/evecs[15,0], evecs[11,0]/evecs[19,0]]
pair_1 = ED8_residual - w[0] * ED12_residual
pair_2 = ED12_residual - w[1] * ED16_residual
pair_3 = ED16_residual - w[2] * ED20_residual
pair_4 = ED8_residual - w[3] * ED16_residual
pair_5 = ED12_residual - w[4] * ED20_residual


# In[28]:


fit_1 = AR_1(pair_1).params
fit_2 = AR_1(pair_2).params
fit_3 = AR_1(pair_3).params
fit_4 = AR_1(pair_4).params
fit_5 = AR_1(pair_5).params
print('AR(1) model is: \n')
print('AR(1) model of pair 1 is: Yt = %f + %f * Yt-1'%(fit_1[0], fit_1[1]))
print('AR(1) model of pair 2 is: Yt = %f + %f * Yt-1'%(fit_2[0], fit_2[1]))
print('AR(1) model of pair 3 is: Yt = %f + %f * Yt-1'%(fit_3[0], fit_3[1]))
print('AR(1) model of pair 4 is: Yt = %f + %f * Yt-1'%(fit_4[0], fit_4[1]))
print('AR(1) model of pair 5 is: Yt = %f + %f * Yt-1'%(fit_5[0], fit_5[1]))


# In[29]:


def EMA_AR_1(lambda_, data):
    M = data.dropna()
    for i in range(1,len(M)):
        M[i] = (1 - lambda_)*M[i-1] + lambda_ * data[i]
    data_EMA = data - M
    # Use this model to fit the AR(1) model.
    return AR_1(data_EMA), data_EMA


# In[30]:


def error(lambda_, data):
    EMA_fit = EMA_AR_1(lambda_, data)[0].params
    hf = HF(EMA_fit[1])
    error = abs(hf - 5)
    return error


# In[31]:


initial_lambda = 0.1
lambda_1 = opt.leastsq(error, initial_lambda, args = pair_1)[0]
lambda_2 = opt.leastsq(error, initial_lambda, args = pair_2)[0]
lambda_3 = opt.leastsq(error, initial_lambda, args = pair_3)[0]
lambda_4 = opt.leastsq(error, initial_lambda, args = pair_4)[0]
lambda_5 = opt.leastsq(error, initial_lambda, args = pair_5)[0]


# In[32]:


print('Opitimized lambda for each EMA paris: \n')
print('lambda_1 for Half-Life ~5days of EMA_1 = %f'%(lambda_1))
print('lambda_2 for Half-Life ~5days of EMA_2 = %f'%(lambda_2))
print('lambda_3 for Half-Life ~5days of EMA_3 = %f'%(lambda_3))
print('lambda_4 for Half-Life ~5days of EMA_4 = %f'%(lambda_4))
print('lambda_5 for Half-Life ~5days of EMA_5 = %f'%(lambda_5))


# In[33]:


fit_EMA1 = EMA_AR_1(lambda_1,pair_1)[0].params
fit_EMA2 = EMA_AR_1(lambda_2,pair_2)[0].params
fit_EMA3 = EMA_AR_1(lambda_3,pair_3)[0].params
fit_EMA4 = EMA_AR_1(lambda_4,pair_4)[0].params
fit_EMA5 = EMA_AR_1(lambda_5,pair_5)[0].params

