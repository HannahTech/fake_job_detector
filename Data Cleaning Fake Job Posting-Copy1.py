#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# to remove the harmless warning
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('fake_job_postings.csv')
df


# In[4]:


#Split our data into train and test data
from sklearn.model_selection import train_test_split


# In[5]:


train_data,test_data=train_test_split(df,test_size=0.33,shuffle=True)


# In[6]:


train_data.shape


# In[7]:


test_data.shape


# In[8]:


train_data.fraudulent.value_counts()


# In[10]:


test_data.fraudulent.value_counts()


# In[11]:


# Train data cleaning


# In[12]:


train_data.shape


# In[13]:


train_data.isna().sum()


# In[48]:


plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# In[14]:


#Drop all the columns with missing values 


# In[18]:


train = train_data.drop(['location','department','salary_range','company_profile','requirements','benefits',
                   'employment_type','required_experience','required_education','industry','function'],1)


# In[19]:


train.isna().sum()


# In[21]:


train=train.drop('job_id',1)


# In[22]:


train.head()


# In[24]:


x=train.drop('fraudulent',1)
y=train['fraudulent']


# In[25]:


x.head()


# In[26]:


y.head()


# In[27]:


# Test data cleaning


# In[28]:


test_data.head()


# In[29]:


test_data.isna().sum()


# In[30]:


test = test_data.drop(['location','department','salary_range','company_profile','requirements','benefits',
                   'employment_type','required_experience','required_education','industry','function'],1)


# In[31]:


test.isna().sum()


# In[32]:


test.dropna(axis=0,inplace=True)


# In[33]:


test.isna().sum()


# In[34]:


#Dropping for the accuracy of the prediction
test1=test.drop('fraudulent',1)


# In[35]:


test1=test1.drop('job_id',1)


# In[36]:


test1


# In[37]:


real_target=test['fraudulent']


# In[38]:


real_target


# In[39]:


# DATA PREPROCESSING


# In[40]:


# Train data


# In[41]:


x.head()


# In[42]:


x.shape


# In[43]:


message=x.copy()


# In[44]:


message.reset_index(drop=True,inplace=True)


# In[45]:


message.head()


# In[46]:


message.shape


# In[ ]:




