
# coding: utf-8

# Assignment 1.2 by Yidi Wang
# 3/31/2018
# Build a Jupyter Notebook to do the following:
# a.	Download the dataset into pandas dataframe
# b.	Remove ‘1M’ column and use the date from 1998-2016 (we will leave 2017-2018 data out for now)
# c.	Construct series of daily differences
# d.	Compute correlations and volatilities among the series (using level data)
# e.	Compute correlations and volatilities among the series (using daily differences)
# f.	Plot the volatility curves computed in 2d & 2e
# 

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# a. Download the dataset into pandas dataframe. I convert the xml original data to csv data with the help of online converter.
data = pd.read_csv("D:/Tandon 2nd Semester/Fixed Income Quantitative Trading/Homework 1/data.csv")


# In[3]:


# b. Remove '1M' column.
# Drop all the useless columns and just keep columns with real number.
data1 = data.dropna(axis=1,how='all') 


# In[4]:


# Select column by its name in the csv file.
data2 = data1.loc[:, ['content/properties/NEW_DATE/__text', 
                      'content/properties/BC_3MONTH/__text',
                      'content/properties/BC_6MONTH/__text',
                      'content/properties/BC_1YEAR/__text',
                      'content/properties/BC_2YEAR/__text',
                      'content/properties/BC_3YEAR/__text',
                      'content/properties/BC_5YEAR/__text',
                      'content/properties/BC_7YEAR/__text',
                      'content/properties/BC_10YEAR/__text',
                      'content/properties/BC_20YEAR/__text',
                      'content/properties/BC_30YEAR/__text',]]


# In[5]:


# Rename the columns in a simple way.
data2.columns = ['Date','3-Month','6-Month','1-Year','2-Year','3-Year','5-Year','7-Year','10-Year','20-Year','30-Year']


# In[6]:


# Modify the dataframe with the index of data date.
data2.set_index('Date')


# In[7]:


# b. Select the data from 1998-2016.
data3 = data2.iloc[2002:6757]


# In[8]:


# Get a standard overview of the dataframe.
data3


# In[9]:


# Get the prepared data.
data_prepared = data3.set_index('Date')


# In[10]:


# c. Construct the daily difference. 
data_difference = data_prepared.diff()


# In[11]:


data_difference


# In[12]:


# d. Compute correlations and volatilities among the series (using level data).
data_prepared.std()


# In[13]:


data_prepared.corr()


# In[14]:


# e. Compute correlations and volatilities among the series (using daily differences)
data_difference.std()


# In[15]:


data_difference.corr()


# In[16]:


# f. Plot the volatility curves computed in 2d & 2e
data_prepared_std = data_prepared.std()
data_prepared_std.plot()
plt.title('The volatility curve computed in 2d.')
plt.xlabel('Maturity')
plt.ylabel('The value of volatility.')


# In[17]:


data_difference_std = data_difference.std()
data_difference_std.plot()
plt.title('The volatility curve computed in 2e.')
plt.xlabel('Maturity')
plt.ylabel('The value of volatility.')

