
# coding: utf-8

# In[1]:


import quandl
import pandas as pd
import numpy as np


# In[2]:


token = "LRiiiFFLhBXpxpGz2THc"
n = range(1,21)
nms = ["CHRIS/CME_ED"+str(i) for i in n]
dfs = [quandl.get(nm, authtoken=token,start_date='2004-01-01', end_date='2007-03-31') for nm in nms]


# In[3]:


dfs[0].describe()


# In[4]:


ls = []
for i in range(0,20):
    l = dfs[i].describe().iloc[1,5]
    ls.append(l)
sort(ls)


# In[5]:


# So the frist 8 rolling Eurodollar futures are:
# CHRIS/CME_ED4, CHRIS/CME_ED5, CHRIS/CME_ED3, CHRIS/CME_ED6,
# CHRIS/CME_ED2, CHRIS/CME_ED7, CHRIS/CME_ED1, CHRIS/CME_ED8.
edf_8 = dfs[0:8]


# In[6]:


edf1 = edf_8[0][["Settle"]]
edf2 = edf_8[1][["Settle"]]
edf3 = edf_8[2][["Settle"]]
edf4 = edf_8[3][["Settle"]]
edf5 = edf_8[4][["Settle"]]
edf6 = edf_8[5][["Settle"]]
edf7 = edf_8[6][["Settle"]]
edf8 = edf_8[7][["Settle"]]


# In[7]:


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


# In[8]:


df12 = pd.concat([edf1_in, edf2_in], axis=1)
df13 = pd.concat([edf1_in, edf3_in], axis=1)
df14 = pd.concat([edf1_in, edf4_in], axis=1)
df15 = pd.concat([edf1_in, edf5_in], axis=1)
df16 = pd.concat([edf1_in, edf6_in], axis=1)
df17 = pd.concat([edf1_in, edf7_in], axis=1)
df18 = pd.concat([edf1_in, edf8_in], axis=1)
df23 = pd.concat([edf2_in, edf3_in], axis=1)
df24 = pd.concat([edf2_in, edf4_in], axis=1)
df25 = pd.concat([edf2_in, edf5_in], axis=1)
df26 = pd.concat([edf2_in, edf6_in], axis=1)
df27 = pd.concat([edf2_in, edf7_in], axis=1)
df28 = pd.concat([edf2_in, edf8_in], axis=1)
df34 = pd.concat([edf3_in, edf4_in], axis=1)
df35 = pd.concat([edf3_in, edf5_in], axis=1)
df36 = pd.concat([edf3_in, edf6_in], axis=1)
df37 = pd.concat([edf3_in, edf7_in], axis=1)
df38 = pd.concat([edf3_in, edf8_in], axis=1)
df45 = pd.concat([edf4_in, edf5_in], axis=1)
df46 = pd.concat([edf4_in, edf6_in], axis=1)
df47 = pd.concat([edf4_in, edf7_in], axis=1)
df48 = pd.concat([edf4_in, edf8_in], axis=1)
df56 = pd.concat([edf5_in, edf6_in], axis=1)
df57 = pd.concat([edf5_in, edf7_in], axis=1)
df58 = pd.concat([edf5_in, edf8_in], axis=1)
df67 = pd.concat([edf6_in, edf7_in], axis=1)
df68 = pd.concat([edf6_in, edf8_in], axis=1)
df78 = pd.concat([edf7_in, edf8_in], axis=1)


# In[9]:


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


# In[10]:


# 3.1 Using Sample1 construct all possible conintegrated WSPR using CCA.
# Obviously, there are C(8,2)=28 combinations of WSPRS.
list_in = [df12, df13, df14, df15, df16, df17, df18,
                 df23, df24, df25, df26, df27, df28,
                       df34, df35, df36, df37, df38,
                             df45, df46, df47, df48,
                                   df56, df57, df58,
                                         df67, df68,
                                               df78]
list_w = []
for df in list_in:
    val, vec, C = CCA_Chou_Ng(df)
    w = val[1] / val[0]
    list_w.append(w)


# In[11]:


list_w


# In[12]:


CCA12_in = edf1_in - list_w[0] * edf2_in
CCA13_in = edf1_in - list_w[1] * edf3_in
CCA14_in = edf1_in - list_w[2] * edf4_in
CCA15_in = edf1_in - list_w[3] * edf5_in
CCA16_in = edf1_in - list_w[4] * edf6_in
CCA17_in = edf1_in - list_w[5] * edf7_in
CCA18_in = edf1_in - list_w[6] * edf8_in
CCA23_in = edf2_in - list_w[7] * edf3_in
CCA24_in = edf2_in - list_w[8] * edf4_in
CCA25_in = edf2_in - list_w[9] * edf5_in
CCA26_in = edf2_in - list_w[10] * edf6_in
CCA27_in = edf2_in - list_w[11] * edf7_in
CCA28_in = edf2_in - list_w[12] * edf8_in
CCA34_in = edf3_in - list_w[13] * edf4_in
CCA35_in = edf3_in - list_w[14] * edf5_in
CCA36_in = edf3_in - list_w[15] * edf6_in
CCA37_in = edf3_in - list_w[16] * edf7_in
CCA38_in = edf3_in - list_w[17] * edf8_in
CCA45_in = edf4_in - list_w[18] * edf5_in
CCA46_in = edf4_in - list_w[19] * edf6_in
CCA47_in = edf4_in - list_w[20] * edf7_in
CCA48_in = edf4_in - list_w[21] * edf8_in
CCA56_in = edf5_in - list_w[22] * edf6_in
CCA57_in = edf5_in - list_w[23] * edf7_in
CCA58_in = edf5_in - list_w[24] * edf8_in
CCA67_in = edf6_in - list_w[25] * edf7_in
CCA68_in = edf6_in - list_w[26] * edf8_in
CCA78_in = edf7_in - list_w[27] * edf8_in


# In[13]:


import numpy as np 
import statsmodels.tsa.stattools as ts
list_CCA_in = [CCA12_in, CCA13_in, CCA14_in, CCA15_in, CCA16_in, CCA17_in, CCA18_in,
                         CCA23_in, CCA24_in, CCA25_in, CCA26_in, CCA27_in, CCA28_in,
                                   CCA34_in, CCA35_in, CCA36_in, CCA37_in, CCA38_in,
                                             CCA45_in, CCA46_in, CCA47_in, CCA48_in,
                                                       CCA56_in, CCA57_in, CCA58_in,
                                                                 CCA67_in, CCA68_in,
                                                                           CCA78_in]
for CCA_in in list_CCA_in:
    adf = ts.adfuller(CCA_in['Settle'].dropna())
    print(adf[1])


# In[14]:


CCA12_out = edf1_out - list_w[0] * edf2_out
CCA13_out = edf1_out - list_w[1] * edf3_out
CCA14_out = edf1_out - list_w[2] * edf4_out
CCA15_out = edf1_out - list_w[3] * edf5_out
CCA16_out = edf1_out - list_w[4] * edf6_out
CCA17_out = edf1_out - list_w[5] * edf7_out
CCA18_out = edf1_out - list_w[6] * edf8_out
CCA23_out = edf2_out - list_w[7] * edf3_out
CCA24_out = edf2_out - list_w[8] * edf4_out
CCA25_out = edf2_out - list_w[9] * edf5_out
CCA26_out = edf2_out - list_w[10] * edf6_out
CCA27_out = edf2_out - list_w[11] * edf7_out
CCA28_out = edf2_out - list_w[12] * edf8_out
CCA34_out = edf3_out - list_w[13] * edf4_out
CCA35_out = edf3_out - list_w[14] * edf5_out
CCA36_out = edf3_out - list_w[15] * edf6_out
CCA37_out = edf3_out - list_w[16] * edf7_out
CCA38_out = edf3_out - list_w[17] * edf8_out
CCA45_out = edf4_out - list_w[18] * edf5_out
CCA46_out = edf4_out - list_w[19] * edf6_out
CCA47_out = edf4_out - list_w[20] * edf7_out
CCA48_out = edf4_out - list_w[21] * edf8_out
CCA56_out = edf5_out - list_w[22] * edf6_out
CCA57_out = edf5_out - list_w[23] * edf7_out
CCA58_out = edf5_out - list_w[24] * edf8_out
CCA67_out = edf6_out - list_w[25] * edf7_out
CCA68_out = edf6_out - list_w[26] * edf8_out
CCA78_out = edf7_out - list_w[27] * edf8_out

list_CCA_out = [CCA12_out, CCA13_out, CCA14_out, CCA15_out, CCA16_out, CCA17_out, CCA18_out,
                           CCA23_out, CCA24_out, CCA25_out, CCA26_out, CCA27_out, CCA28_out,
                                      CCA34_out, CCA35_out, CCA36_out, CCA37_out, CCA38_out,
                                                 CCA45_out, CCA46_out, CCA47_out, CCA48_out,
                                                            CCA56_out, CCA57_out, CCA58_out,
                                                                       CCA67_out, CCA68_out,
                                                                                  CCA78_out]


# In[15]:


for CCA_out in list_CCA_out:
    adf = ts.adfuller(CCA_out['Settle'].dropna())
    print(adf[1])

