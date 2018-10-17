
# coding: utf-8

# In[1]:


import quandl
import pandas as pd
import numpy as np


# In[2]:


token = "LRiiiFFLhBXpxpGz2THc"
n = range(1,9)
nms = ["CHRIS/CME_ED"+str(i) for i in n]
dfs = [quandl.get(nm, authtoken=token,start_date='2004-01-01', end_date='2007-03-31') for nm in nms]


# In[3]:


edf_8 = dfs[0:8]


# In[4]:


edf1 = edf_8[0][["Settle"]]
edf2 = edf_8[1][["Settle"]]
edf3 = edf_8[2][["Settle"]]
edf4 = edf_8[3][["Settle"]]
edf5 = edf_8[4][["Settle"]]
edf6 = edf_8[5][["Settle"]]
edf7 = edf_8[6][["Settle"]]
edf8 = edf_8[7][["Settle"]]


# In[5]:


edf1_in = edf1["2004-01-02":"2006-12-31"]
edf1_out = edf1["2006-01-01":"2007-03-30"]

edf2_in = edf2["2004-01-02":"2006-12-31"]
edf2_out = edf2["2006-01-01":"2007-03-30"]

edf3_in = edf3["2004-01-02":"2006-12-31"]
edf3_out = edf3["2006-01-01":"2007-03-30"]

edf4_in = edf4["2004-01-02":"2006-12-31"]
edf4_out = edf4["2006-01-01":"2007-03-30"]

edf5_in = edf5["2004-01-02":"2006-12-31"]
edf5_out = edf5["2006-01-01":"2007-03-30"]

edf6_in = edf6["2004-01-02":"2006-12-31"]
edf6_out = edf6["2006-01-01":"2007-03-30"]

edf7_in = edf7["2004-01-02":"2006-12-31"]
edf7_out = edf7["2006-01-01":"2007-03-30"]

edf8_in = edf8["2004-01-02":"2006-12-31"]
edf8_out = edf8["2006-01-01":"2007-03-30"]


# In[6]:


df12 = pd.concat([edf1_in, edf2_in], axis=1)
df13 = pd.concat([edf1_in, edf3_in], axis=1)
df68 = pd.concat([edf6_in, edf8_in], axis=1)
df78 = pd.concat([edf7_in, edf8_in], axis=1)


# In[7]:


# CCA function
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


# In[8]:


list_in = [df12, df13, df78, df68]
list_w = [ ]


# In[9]:


for df in list_in:
    val, vec, C = CCA_Chou_Ng(df)
    w = val[1] / val[0]
    list_w.append(w)


# In[10]:


list_w


# In[11]:


CCA12_in = edf1_in - list_w[0] * edf2_in
CCA13_in = edf1_in - list_w[1] * edf3_in
CCA78_in = edf7_in - list_w[2] * edf8_in
CCA68_in = edf6_in - list_w[3] * edf8_in


# In[12]:


import statsmodels.tsa.stattools as ts
list_in = [CCA12_in, CCA13_in, CCA78_in, CCA68_in]
for CCA_in in list_in:
    print(CCA_in.describe())
    adf = ts.adfuller(CCA_in['Settle'].dropna())
    print(adf[1])


# In[13]:


CCA12_out = edf1_out - list_w[0] * edf2_out
CCA13_out = edf1_out - list_w[1] * edf3_out
CCA68_out = edf6_out - list_w[2] * edf8_out
CCA78_out = edf7_out - list_w[3] * edf8_out


# In[14]:


list_out = [CCA12_out, CCA13_out, CCA78_out, CCA68_out]
for CCA_out in list_out:
    print(CCA_out.describe())
    adf = ts.adfuller(CCA_out['Settle'].dropna())
    print(adf[1])


# In[25]:


df_imm = pd.read_csv("D:/immDate.csv",index_col='Date')
df_imm = df_imm['1/2/2004':'3/30/2007']
ED1_imm = df_imm['ED1']

