#!/usr/bin/env python
# coding: utf-8

# Springboard API Mini Project

# These are your tasks for this mini project:
# Collect data from the Franfurt Stock Exchange, for the ticker AFX_X, for the whole year 2017 (keep in mind that the date format is YYYY-MM-DD).
# Convert the returned JSON object into a Python dictionary.
# Calculate what the highest and lowest opening prices were for the stock in this period.
# What was the largest change in any one day (based on High and Low price)?
# What was the largest change between any two days (based on Closing Price)?
# What was the average daily trading volume during this year?
# (Optional) What was the median trading volume during this year. (Note: you may need to implement your own function for calculating the median.)

# In[1]:


# Store the API key as a string - according to PEP8, constants are always named in all upper case
#ticker - AFX_X


# In[2]:


# First, import the relevant modules
import requests
import collections
import json


# In[3]:


# Now, call the Quandl API and pull out a small sample of the data (only one day) to get a glimpse
# into the JSON structure that will be returned
response = requests.get("https://www.quandl.com/api/v3/datasets/FSE/AFX_X.json?start_date=2017-01-02&end_date=2017-12-29&exclude_column_names=true&order=asc")


# In[4]:


response.status_code


# In[5]:


print(response.content)


# In[6]:


data = response.json()


# In[7]:


data


# In[8]:


type(data)


# In[9]:


stock_data = data['dataset']['data']


# In[10]:


type(stock_data)


# In[11]:


stock_data


# In[12]:


len(stock_data)


# In[34]:


#define highest and lowest opening stock prices
highest_opening = 0
lowest_opening = 1000000


# In[39]:


for row in stock_data:
    if (row[1] != None and row[1] < lowest_opening):
        lowest_opening = row[1]
    if (row[1] != None and row[1] > highest_opening):
        highest_opening = row[1]


# In[30]:


lowest_opening


# In[40]:


highest_opening


# In[47]:


#Largest change in one day based on high and low prices 
#create the variables 
difference = 0


# In[49]:


for row in stock_data:
    if ((row[2]-row[3]) > difference):
        difference = row[2]-row[3]


# In[50]:


difference


# 5. What was the largest change between any two days (based on Closing Price)?

# In[72]:


closing_diff = 0
for i in range(0, total_days-1):
    if (abs(stock_data[i][4] - stock_data[i+1][4]) > closing_diff):
        closing_diff = abs(stock_data[i][4] - stock_data[i+1][4])


# In[70]:


closing_diff


# In[53]:


#6 avg daily trading volume
avg_trade_vol = 0
total_trade_vol = 0
total_days = len(stock_data)


# In[54]:


for row in stock_data:
    total_trade_vol += row[6]


# In[57]:


avg_trade_vol = total_trade_vol / total_days


# In[58]:


avg_trade_vol


# 7. (Optional) What was the median trading volume during this year. (Note: you may need to implement your own function for calculating the median.)

# In[59]:


#create new list and then sort the volume
median_list = []


# In[60]:


for row in stock_data:
    median_list.append(row[6])


# In[62]:


median_list.sort()


# In[67]:


median_trading_vol = median_list[int(total_days/2)]


# In[68]:


median_trading_vol


# In[ ]:




