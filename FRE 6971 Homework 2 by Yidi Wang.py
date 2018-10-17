
# coding: utf-8

# FRE 6971 Homework 2 by Yidi Wang
# 4/11/2018
# 1. Definitions: (1) Sample1 : 1/2/2012 to 12/31/2015
#                 (2) WFLY: 5Y*w1 - 7Y + 10Y*w2, weights = (w1,-1,w2)
# 2. Build a Jupyter Notebook to do the following:
#     a. Download a panel of CMT rates into pandas dataframe & remove '1M' column from the dataset.
#     b. Perform PCA on the dataset using Sample1.
#     c. Use this PCA model to analyze the CMT curve move on the Election Day: 11/8/2016 to 11/9/2016.
#            i. Plot CMT curve move vs the move explained by the first PCA factor, first 2 PCA factors,
#               first 3 factors.
#     d. Compute weights of the WFLY to make sure that WFLY does not have PCA1, 2 risk exposure in Sample1.
#        Let's call this combination WFLY1.
#     e. Choose weights of the WFLY from cointegration analysis (weights correspond to the best
#        conintegrated vector). Let's call this combination WFLY2.
#            i. Use Chou-Ng estimation procedure or Box-Tiao.
# 3. Compute Half-Life & ADF statistic for WFLY1, WFLY2 using Sample1, compare results
#     a. Note that you are using time series of levels, not daily differences.
# 4. Repeat Step3 out-of-sample: using 3m, 6m, 12m out of sample periods.
#     a. How do out-of-sample results compare across periods and combinations

# In[1]:


# 1 Definitions.
# Try to construct the standard datasets.
# I prepare two datasets, sample1 from 1/2/2012 to 12/31/2015, and the original data. 
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
data = pd.read_csv("D:\\data.csv", index_col=0)
sample = data.iloc[5506:6506]
sample1 = sample.set_index('Date')
data_original = data.set_index('Date')


# In[2]:


print("The sample1 data:")
print(sample1.head(1),sample1.tail(1))
print("\nThe original data:")
print(data_original.head(1),data_original.tail(1))


# In[3]:


# 2. Build a Jupyter Notebook to do the following.
# a. Download a panel of CMT rates into pandas dataframe and remove '1M' column from the dataset.
sample1_ret = sample1.diff(periods = 1).dropna()
print(sample1_ret.head(1), sample1_ret.tail(1))


# In[4]:


# b. Perform PCA on the dataset using Sample1.
# Perform PCA for both the original data and difference data.
from sklearn.decomposition import PCA
pca_ret = PCA(n_components = 10)
pca_ret.fit(sample1_ret)
f_ret = pca_ret.components_
var_ratio_ret = np.cumsum(np.round(pca_ret.explained_variance_ratio_, decimals = 4))
print('% variance explained for the dif sample1 data: ')
print(var_ratio_ret)

pca = PCA(n_components = 10)
pca.fit(sample1)
f = pca.components_
var_ratio = np.cumsum(np.round(pca.explained_variance_ratio_, decimals = 4))
print('% variance explained for the sample1 data')
print(var_ratio)


# In[5]:


# c. Use this PCA model to analyze the CMT curve move on the Election Day: 11/8/2016 to 11/9/2016.
# i. Plot CMT curve move.
plt.figure(figsize = (20, 10))
plt.plot(data_original[['3-Month','6-Month','1-Year','2-Year','3-Year',
                        '5-Year','7-Year','10-Year','20-Year','30-Year']],linewidth = 3)
plt.title('The CMT curve.')
plt.xlabel('Time')
plt.ylabel('The yield rate.')
plt.show()


# In[6]:


# c. Use the PCA model for the data_ret.
data_original_ret = data_original.diff(periods = 1).dropna()
c_ret = data_original_ret.cov()
val_ret, vec_ret = np.linalg.eig(c_ret)
tr_ret = sum(val_ret)
var_ratio_custom_ret = np.cumsum(np.round(val_ret / tr_ret, decimals = 4))
print(var_ratio_custom_ret)

# c. Use the PCA model for the data.
c = data_original.cov()
val, vec = np.linalg.eig(c)
tr = sum(val)
var_ratio_custom = np.cumsum(np.round(val / tr, decimals = 4))
print(var_ratio_custom)


# In[7]:


# Plot the pca1, pca2, and pca3 loading for the data_ret.
term = [3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
plt.figure(figsize = (20, 10))
plt.plot(term,vec_ret[:,0], label = 'pca1 loading for the data_ret', linewidth = 3)
plt.plot(term,vec_ret[:,1], label = 'pca2 loading for the data_ret', linewidth = 3)
plt.plot(term,vec_ret[:,2], label = 'pca3 loading for the data_ret', linewidth = 3)
plt.legend(loc = 'best', fontsize = 'xx-large')
plt.title('pca1,pca2 and pca3 of data_ret.')


# In[8]:


# Plot the pca1, pca2, and pca3 loading for the data.
term = [3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
plt.figure(figsize = (20, 10))
plt.plot(term,vec[:,0], label = 'pca1 loading for the data', linewidth = 3)
plt.plot(term,vec[:,1], label = 'pca2 loading for the data', linewidth = 3)
plt.plot(term,vec[:,2], label = 'pca3 loading for the data', linewidth = 3)
plt.legend(loc = 'best', fontsize = 'xx-large')
plt.title('pca1,pca2, and pca3 of data.')


# In[9]:


def plotme(dts, frame):
    f = plt.figure(figsize = (20, 10))
    ax = f.add_subplot(111)
    ax.yaxis.tick_right()
    x = [3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
    crvs = [frame.loc[dt] for dt in dts]
    plots = [plt.plot(x, crv.values, label = dt, linewidth = 3) for crv, dt in zip(crvs, dts)]
    plt.legend(loc = 'lower rigth', fontsize = 'xx-large')
    plt.show()
    
dts = ['2016-11-08T00:00:00', '2016-11-09T00:00:00']
plotme(dts, data_original)    


# In[10]:


# ii. Explain your calculations and results.
print(data_original_ret.columns.tolist(), vec_ret[:,0])
print(data_original_ret.columns.tolist(), vec_ret[:,1])
print(data_original_ret.columns.tolist(), vec_ret[:,2])

print(data_original.columns.tolist(), vec[:,0])
print(data_original.columns.tolist(), vec[:,1])
print(data_original.columns.tolist(), vec[:,2])


# In[11]:


# d. Compute weights of the WFLY to make sure that WFLY does not have PCA1,2 risk exposure in Sample1.
c_sample1 = sample1.cov()
val_sample1, vec_sample1 = np.linalg.eig(c_sample1)
tr_sample1 = sum(val_sample1)
var_ratio_sample1 = np.cumsum(np.round(val_sample1 / tr_sample1, decimals = 4))
print(var_ratio_sample1)

c_sample1_ret = sample1_ret.cov()
val_sample1_ret, vec_sample1_ret = np.linalg.eig(c_sample1_ret)
tr_sample1_ret = sum(val_sample1_ret)
var_ratio_sample1_ret = np.cumsum(np.round(val_sample1_ret / tr_sample1_ret, decimals = 4))
print(var_ratio_sample1_ret)


# In[12]:


print(sample1.columns.tolist(), vec_sample1[:,0])
print(sample1.columns.tolist(), vec_sample1[:,1])


# In[13]:


print(sample1_ret.columns.tolist(), vec_sample1_ret[:,0])
print(sample1_ret.columns.tolist(), vec_sample1_ret[:,1])


# In[14]:


pca_w = np.array([[0.428087, 0.46030264], [-0.39877806, 0.11506476]])
pca_7y = np.array([0.49814324, -0.21408868])
print(np.linalg.solve(pca_w, pca_7y))


# In[15]:


pca_w_ret = np.array([[0.38584075, 0.43905623], [-0.37708859, 0.10439566]])
pca_7y_ret = np.array([0.44672618, -0.14604436])
print(np.linalg.solve(pca_w_ret, pca_7y_ret))


# In[16]:


# e. Choose weights of the WFLY from cointegration analysis.
# i. Use Chou-Ng estimation procedure.
import statsmodels.api as sm

def CCA_Chou_Ng(data_set):
    df_lag = data_set.shift(1).dropna()
    df = data_set.drop(data_set.index[0]).dropna()
    n = len(data_set.columns)
    
    X = df_lag.as_matrix()
    X_I = sm.add_constant(X)
    Y = df.as_matrix()
    l1 = sm.OLS(Y, X_I).fit()
    B = l1.params[1:(n+1)]
    
    Y_I = sm.add_constant(X)
    l2 = sm.OLS(X, Y_I).fit()
    A = l2.params[1:(n+1)]
    C = np.dot(A,B)
    eig_val, eig_vec = np.linalg.eig(C)
    return eig_val, eig_vec, C

legs = ['2-Year', '10-Year']
belly = '5-Year'
flylist = [legs[0], belly, legs[1]]
fly = sample1.loc['2012-01-03T00:00:00':'2015-12-30T00:00:00',flylist]
df1 = fly
n = len(df1.columns)
val, vec, C = CCA_Chou_Ng(df1)
vec = pd.DataFrame(vec)
vec = vec.rename(columns = lambda x: n-x-1, inplace = False)
print(vec)
w = vec[2].values
w_cca_cn = [-w[0]/w[1], -w[2]/w[1]]
print(w_cca_cn)


# In[17]:


legs = ['2-Year', '10-Year']
belly = '5-Year'
flylist = [legs[0], belly, legs[1]]
fly = sample1_ret.loc['2012-01-03T00:00:00':'2015-12-30T00:00:00',flylist]
df1 = fly
n = len(df1.columns)
val, vec, C = CCA_Chou_Ng(df1)
vec = pd.DataFrame(vec)
vec = vec.rename(columns = lambda x: n-x-1, inplace = False)
print(vec)
w = vec[2].values
w_cca_cn = [-w[0]/w[1], -w[2]/w[1]]
print(w_cca_cn)


# In[18]:


# 3. Compute Half-Life & ADF statistic for WFLY1, WFLY2 using Sample1.
# PCA
sample1['Portfolio'] = 0.66947332 * sample1['5-Year'] - sample1['7-Year'] + 0.45958983 * sample1['10-Year']
sample1['Portfolio'].describe()


# In[19]:


sample1_ret['Portfolio'] = 0.53806958 * sample1_ret['5-Year'] - sample1_ret['7-Year'] + 0.54461592 * sample1_ret['10-Year']
sample1_ret['Portfolio'].describe()


# In[20]:


# So difference method is much better than the original data.
sample1_ret['Portfolio_CCA'] = 1.6721588439085835 * sample1_ret['5-Year'] - sample1_ret['7-Year'] + 0.39639517135404684 * sample1_ret['10-Year']
sample1_ret['Portfolio_CCA'].describe()


# In[ ]:


# PCA is much better than CCA method.

