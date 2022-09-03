#!/usr/bin/env python
# coding: utf-8

# # Clustering Crypto

# In[1]:


# Initial imports
import pandas as pd
import hvplot.pandas
from path import Path
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# In[3]:


pip install pickleshare


# ### Deliverable 1: Preprocessing the Data for PCA

# In[57]:


# Load the crypto_data.csv dataset.
crypto = pd.read_csv('crypto_data.csv')
crypto


# In[58]:


# Keep all the cryptocurrencies that are being traded.
traded_crypto = crypto[crypto['IsTrading'] == True]
traded_crypto


# In[59]:


# Keep all the cryptocurrencies that have a working algorithm.
working_crypto = traded_crypto.dropna()
working_crypto


# In[60]:


# Remove the "IsTrading" column. 
working_crypto_new = working_crypto.drop(['IsTrading'], axis=1)
working_crypto_new


# In[61]:


# Remove rows that have at least 1 null value.
crypto_new = working_crypto_new.dropna()
crypto_new


# In[62]:


# Keep the rows where coins are mined.
crypto_new_mined = crypto_new[crypto_new['TotalCoinsMined'] > 0]
crypto_new_mined


# In[64]:


# Create a new DataFrame that holds only the cryptocurrencies names.
crypto_name = crypto_new_mined[['CoinName']]
crypto_name


# In[65]:


# Drop the 'CoinName' column since it's not going to be used on the clustering algorithm.
crypto_no_name = crypto_new_mined.drop(['CoinName'], axis=1)
crypto_no_name


# In[66]:


# Use get_dummies() to create variables for text features.
crypto_no_name.get_dummies()


# In[11]:


# Standardize the data with StandardScaler().
# YOUR CODE HERE


# ### Deliverable 2: Reducing Data Dimensions Using PCA

# In[12]:


# Using PCA to reduce dimension to three principal components.
# YOUR CODE HERE


# In[13]:


# Create a DataFrame with the three principal components.
# YOUR CODE HERE


# ### Deliverable 3: Clustering Crytocurrencies Using K-Means
# 
# #### Finding the Best Value for `k` Using the Elbow Curve

# In[14]:


# Create an elbow curve to find the best value for K.
# YOUR CODE HERE


# Running K-Means with `k=4`

# In[15]:


# Initialize the K-Means model.
# YOUR CODE HERE

# Fit the model
# YOUR CODE HERE

# Predict clusters
# YOUR CODE HERE


# In[16]:


# Create a new DataFrame including predicted clusters and cryptocurrencies features.
# Concatentate the crypto_df and pcs_df DataFrames on the same columns.
# YOUR CODE HERE

#  Add a new column, "CoinName" to the clustered_df DataFrame that holds the names of the cryptocurrencies. 
# YOUR CODE HERE

#  Add a new column, "Class" to the clustered_df DataFrame that holds the predictions.
# YOUR CODE HERE

# Print the shape of the clustered_df
print(clustered_df.shape)
clustered_df.head(10)


# ### Deliverable 4: Visualizing Cryptocurrencies Results
# 
# #### 3D-Scatter with Clusters

# In[17]:


# Creating a 3D-Scatter with the PCA data and the clusters
# YOUR CODE HERE


# In[18]:


# Create a table with tradable cryptocurrencies.
# YOUR CODE HERE


# In[19]:


# Print the total number of tradable cryptocurrencies.
# YOUR CODE HERE


# In[20]:


# Scaling data to create the scatter plot with tradable cryptocurrencies.
# YOUR CODE HERE


# In[21]:


# Create a new DataFrame that has the scaled data with the clustered_df DataFrame index.
# YOUR CODE HERE

# Add the "CoinName" column from the clustered_df DataFrame to the new DataFrame.
# YOUR CODE HERE

# Add the "Class" column from the clustered_df DataFrame to the new DataFrame. 
# YOUR CODE HERE

plot_df.head(10)


# In[22]:


# Create a hvplot.scatter plot using x="TotalCoinsMined" and y="TotalCoinSupply".
# YOUR CODE HERE


# In[ ]:




