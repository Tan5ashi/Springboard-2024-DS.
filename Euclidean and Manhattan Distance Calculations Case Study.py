#!/usr/bin/env python
# coding: utf-8

# ## Euclidean and Manhattan Distance Calculations
# 
# In this short mini project you will see examples and comparisons of distance measures. Specifically, you'll visually compare the Euclidean distance to the Manhattan distance measures. The application of distance measures has a multitude of uses in data science and is the foundation of many algorithms you'll be using such as Prinical Components Analysis.

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


plt.style.use('ggplot')


# In[5]:


# Load Course Numerical Dataset
df = pd.read_csv('data/distance_dataset.csv',index_col=0)
df.head()


# In[6]:


df.describe()


# #### Data
# 
# - X,Y,Z centered at **5 ± 2**
# - range from **1 to 9**

# ### Euclidean Distance
# 
# Let's visualize the difference between the Euclidean and Manhattan distance.
# 
# We are using Pandas to load our dataset .CSV file and use Numpy to compute the __Euclidean distance__ to the point (Y=5, Z=5) that we choose as reference. On the left here we show the dataset projected onto the YZ plane and color coded per the Euclidean distance we just computed. As we are used to, points that lie at the same Euclidean distance define a regular 2D circle of radius that distance.
# 
# Note that the __SciPy library__ comes with optimized functions written in C to compute distances (in the scipy.spatial.distance module) that are much faster than our (naive) implementation.

# In[7]:


# In the Y-Z plane, we compute the distance to ref point (5,5)
distEuclid = np.sqrt((df.Z - 5)**2 + (df.Y - 5)**2)


# In[8]:


figEuclid = plt.figure(figsize=[10,8])

plt.scatter(df.Y - 5, df.Z-5, c=distEuclid, s=20)
plt.ylim([-5,5])
plt.xlim([-5,5])
plt.xlabel('Y - 5', size=14)
plt.ylabel('Z - 5', size=14)
plt.title('Euclidean Distance')
cb = plt.colorbar()
cb.set_label('Distance from (5,5)', size=14)

figEuclid.savefig('plots/Euclidean_055.png')


# **<font color='teal'>Create a distance to reference point (3,3) matrix similar to the above example.</font>**

# In[9]:


distEuclid2 = np.sqrt((df.Z - 3)**2 + (df.Y - 3)**2)


# **<font color='teal'>Replace the value set to 'c' in the plotting cell below with your own distance matrix and review the result to deepen your understanding of Euclidean distances. </font>**

# In[10]:


figEuclid = plt.figure(figsize=[10,8])

plt.scatter(df.Y - 3, df.Z-3, c=distEuclid2, s=20)
plt.ylim([-2,6])
plt.xlim([-2,6])
plt.xlabel('Y - 3', size=14)
plt.ylabel('Z - 3', size=14)
plt.title('Euclidean Distance')
cb = plt.colorbar()
cb.set_label('Distance from (3,3)', size=14)

# figEuclid.savefig('plots/Euclidean_033.png')


# ### Manhattan Distance
# 
# Manhattan distance is simply the sum of absolute differences between the points coordinates. This distance is also known as the taxicab or city block distance as it measure distances along the coorinate axis which creates "paths" that look like a cab's route on a grid-style city map.
# 
# We display the dataset projected on the XZ plane here color coded per the Manhattan distance to the (X=5, Z=5) reference point. We can see that points laying at the same distance define a circle that looks like a Euclidean square.

# In[11]:


# In the Y-Z plane, we compute the distance to ref point (5,5)
distManhattan = np.abs(df.X - 5) + np.abs(df.Z - 5)


# In[12]:


figM = plt.figure(figsize=[10,8])

plt.scatter(df.X - 5, df.Z-5, c=distManhattan, s=20)
plt.ylim([-5,5])
plt.xlim([-5,5])
plt.xlabel('X - 5', size=14)
plt.ylabel('Z - 5', size=14)
plt.title('Manhattan Distance')
cb = plt.colorbar()
cb.set_label('Distance from (5,5)', size=14)
figM.savefig('plots/Manhattan_505.png')


# **<font color='teal'>Create a Manhattan distance to reference point (4,4) matrix similar to the above example and replace the value for 'c' in the plotting cell to view the result.</font>**

# In[13]:


# In the Y-Z plane, we compute the distance to ref point (4,4)
distManhattan = np.abs(df.X - 4) + np.abs(df.Z - 4)

figM = plt.figure(figsize=[10,8])

plt.scatter(df.X - 4, df.Z-4, c=distManhattan, s=20)
plt.ylim([-4,6])
plt.xlim([-4,6])
plt.xlabel('X - 4', size=14)
plt.ylabel('Z - 4', size=14)
plt.title('Manhattan Distance')
cb = plt.colorbar()
cb.set_label('Distance from (4,4)', size=14)
figM.savefig('plots/Manhattan_404.png')


# Now let's create distributions of these distance metrics and compare them. We leverage the scipy dist function to create these matrices similar to how you manually created them earlier in the exercise.

# In[14]:


import scipy.spatial.distance as dist

mat = df[['X','Y','Z']].to_numpy()
DistEuclid = dist.pdist(mat,'euclidean')
DistManhattan = dist.pdist(mat, 'cityblock')
largeMat = np.random.random((10000,100))


# In[15]:


x = pd.DataFrame(pd.Series(DistEuclid))


# In[16]:


distdf = pd.DataFrame(pd.concat([pd.Series(DistEuclid), pd.Series(DistManhattan)]))


# In[17]:


distdf['type'] = 'Euclid'


# In[18]:


distdf.iloc[len(DistEuclid):,1] = 'Manhattan'


# In[19]:


distdf.value_counts('type')


# **<font color='teal'>Plot histograms of each distance matrix for comparison.</font>**

# In[20]:


import seaborn as sns


# In[21]:


sns.histplot(distdf)


# In[22]:


plt.figure(figsize=[10,8])
plt.hist(DistManhattan, color='blue', alpha=0.8,  bins=100)
plt.hist(DistEuclid, color='red', alpha=0.5, bins=100)
plt.legend(['Manhattan', 'Euclid'])
plt.title('Distance Comparison')
plt.savefig('plots/histogram.png')
plt.show()

