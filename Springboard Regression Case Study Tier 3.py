#!/usr/bin/env python
# coding: utf-8

# # Springboard Regression Case Study, Unit 8 - The Red Wine Dataset - Tier 3

# Welcome to the Unit 8 Springboard Regression case study! Please note: this is ***Tier 3*** of the case study.
# 
# This case study was designed for you to **use Python to apply the knowledge you've acquired in reading *The Art of Statistics* (hereinafter *AoS*) by Professor Spiegelhalter**. Specifically, the case study will get you doing regression analysis; a method discussed in Chapter 5 on p.121. It might be useful to have the book open at that page when doing the case study to remind you of what it is we're up to (but bear in mind that other statistical concepts, such as training and testing, will be applied, so you might have to glance at other chapters too).  
# 
# The aim is to ***use exploratory data analysis (EDA) and regression to predict alcohol levels in wine with a model that's as accurate as possible***. 
# 
# We'll try a *univariate* analysis (one involving a single explanatory variable) as well as a *multivariate* one (involving multiple explanatory variables), and we'll iterate together towards a decent model by the end of the notebook. The main thing is for you to see how regression analysis looks in Python and jupyter, and to get some practice implementing this analysis.
# 
# Throughout this case study, **questions** will be asked in the markdown cells. Try to **answer these yourself in a simple text file** when they come up. Most of the time, the answers will become clear as you progress through the notebook. Some of the answers may require a little research with Google and other basic resources available to every data scientist. 
# 
# For this notebook, we're going to use the red wine dataset, wineQualityReds.csv. Make sure it's downloaded and sitting in your working directory. This is a very common dataset for practicing regression analysis and is actually freely available on Kaggle, [here](https://www.kaggle.com/piyushgoyal443/red-wine-dataset).
# 
# You're pretty familiar with the data science pipeline at this point. This project will have the following structure: 
# **1. Sourcing and loading** 
# - Import relevant libraries
# - Load the data 
# - Exploring the data
# - Choosing a dependent variable
#  
# **2. Cleaning, transforming, and visualizing**
# - Visualizing correlations
#   
#   
# **3. Modeling** 
# - Train/Test split
# - Making a Linear regression model: your first model
# - Making a Linear regression model: your second model: Ordinary Least Squares (OLS) 
# - Making a Linear regression model: your third model: multiple linear regression
# - Making a Linear regression model: your fourth model: avoiding redundancy
# 
# **4. Evaluating and concluding** 
# - Reflection 
# - Which model was best?
# - Other regression algorithms

# ### 1. Sourcing and loading

# #### 1a. Import relevant libraries 

# In[1]:


# Import relevant libraries and packages.
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns # For all our visualization needs.
import statsmodels.api as sm # What does this do? Find out and type here.
from statsmodels.graphics.api import abline_plot # What does this do? Find out and type here.
from sklearn.metrics import mean_squared_error, r2_score # What does this do? Find out and type here.
from sklearn.model_selection import train_test_split #  What does this do? Find out and type here.
from sklearn import linear_model, preprocessing # What does this do? Find out and type here.
import warnings # For handling error messages.
# Don't worry about the following two instructions: they just suppress warnings that could occur later. 
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


# #### 1b. Load the data

# In[2]:


# Load the data. 
df = pd.read_csv('wineQualityReds.csv', index_col=0)


# #### 1c. Exploring the data

# In[3]:


# Check out its appearance. 
df.head()


# In[4]:


# Another very useful method to call on a recently imported dataset is .info(). Call it here to get a good
# overview of the data
df.info()


# What can you infer about the nature of these variables, as output by the info() method?
# 
# Which variables might be suitable for regression analysis, and why? For those variables that aren't suitable for regression analysis, is there another type of statistical modeling for which they are suitable?

# In[5]:


# We should also look more closely at the dimensions of the dataset. 
df.shape


# #### 1d. Choosing a dependent variable

# We now need to pick a dependent variable for our regression analysis: a variable whose values we will predict. 
# 
# 'Quality' seems to be as good a candidate as any. Let's check it out. One of the quickest and most informative ways to understand a variable is to make a histogram of it. This gives us an idea of both the center and spread of its values. 

# In[6]:


# Making a histogram of the quality variable.
df['quality'].hist()


# We can see so much about the quality variable just from this simple visualization. Answer yourself: what value do most wines have for quality? What is the minimum quality value below, and the maximum quality value? What is the range? Remind yourself of these summary statistical concepts by looking at p.49 of the *AoS*.
# 
# But can you think of a problem with making this variable the dependent variable of regression analysis? Remember the example in *AoS* on p.122 of predicting the heights of children from the heights of parents? Take a moment here to think about potential problems before reading on. 
# 
# The issue is this: quality is a *discrete* variable, in that its values are integers (whole numbers) rather than floating point numbers. Thus, quality is not a *continuous* variable. But this means that it's actually not the best target for regression analysis. 
# 
# Before we dismiss the quality variable, however, let's verify that it is indeed a discrete variable with some further exploration. 

# In[7]:


# Get a basic statistical summary of the variable 
df['quality'].describe()

# What do you notice from this summary? 


# In[8]:


# Get a list of the values of the quality variable, and the number of occurrences of each. 
df['quality'].value_counts() 


# The outputs of the describe() and value_counts() methods are consistent with our histogram, and since there are just as many values as there are rows in the dataset, we can infer that there are no NAs for the quality variable. 
# 
# But scroll up again to when we called info() on our wine dataset. We could have seen there, already, that the quality variable had int64 as its type. As a result, we had sufficient information, already, to know that the quality variable was not appropriate for regression analysis. Did you figure this out yourself? If so, kudos to you!
# 
# The quality variable would, however, conduce to proper classification analysis. This is because, while the values for the quality variable are numeric, those numeric discrete values represent *categories*; and the prediction of category-placement is most often best done by classification algorithms. You saw the decision tree output by running a classification algorithm on the Titanic dataset on p.168 of Chapter 6 of *AoS*. For now, we'll continue with our regression analysis, and continue our search for a suitable dependent variable. 
# 
# Now, since the rest of the variables of our wine dataset are continuous, we could — in theory — pick any of them. But that does not mean that are all equally sutiable choices. What counts as a suitable dependent variable for regression analysis is determined not just by *intrinsic* features of the dataset (such as data types, number of NAs etc) but by *extrinsic* features, such as, simply, which variables are the most interesting or useful to predict, given our aims and values in the context we're in. Almost always, we can only determine which variables are sensible choices for dependent variables with some **domain knowledge**. 
# 
# Not all of you might be wine buffs, but one very important and interesting quality in wine is [acidity](https://waterhouse.ucdavis.edu/whats-in-wine/fixed-acidity). As the Waterhouse Lab at the University of California explains, 'acids impart the sourness or tartness that is a fundamental feature in wine taste.  Wines lacking in acid are "flat." Chemically the acids influence titrable acidity which affects taste and pH which affects  color, stability to oxidation, and consequantly the overall lifespan of a wine.'
# 
# If we cannot predict quality, then it seems like **fixed acidity** might be a great option for a dependent variable. Let's go for that.

# So if we're going for fixed acidity as our dependent variable, what we now want to get is an idea of *which variables are related interestingly to that dependent variable*. 
# 
# We can call the .corr() method on our wine data to look at all the correlations between our variables. As the [documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html) shows, the default correlation coefficient is the Pearson correlation coefficient (p.58 and p.396 of the *AoS*); but other coefficients can be plugged in as parameters. Remember, the Pearson correlation coefficient shows us how close to a straight line the data-points fall, and is a number between -1 and 1. 

# In[9]:


# Call the .corr() method on the wine dataset 
df.corr()


# Ok - you might be thinking, but wouldn't it be nice if we visualized these relationships? It's hard to get a picture of the correlations between the variables without anything visual. 
# 
# Very true, and this brings us to the next section.

# ### 2. Cleaning, Transforming, and Visualizing 

# #### 2a. Visualizing correlations 
# The heading of this stage of the data science pipeline ('Cleaning, Transforming, and Visualizing') doesn't imply that we have to do all of those operations in *that order*. Sometimes (and this is a case in point) our data is already relatively clean, and the priority is to do some visualization. Normally, however, our data is less sterile, and we have to do some cleaning and transforming first prior to visualizing. 

# Now that we've chosen alcohol level as our dependent variable for regression analysis, we can begin by plotting the pairwise relationships in the dataset, to check out how our variables relate to one another.

# In[10]:


# Make a pairplot of the wine data
_ = sns.pairplot(df)


# If you've never executed your own Seaborn pairplot before, just take a moment to look at the output. They certainly output a lot of information at once. What can you infer from it? What can you *not* justifiably infer from it?
# 
# ... All done? 
# 
# Here's a couple things you might have noticed: 
# - a given cell value represents the correlation that exists between two variables 
# - on the diagonal, you can see a bunch of histograms. This is because pairplotting the variables with themselves would be pointless, so the pairplot() method instead makes histograms to show the distributions of those variables' values. This allows us to quickly see the shape of each variable's values.  
# - the plots for the quality variable form horizontal bands, due to the fact that it's a discrete variable. We were certainly right in not pursuing a regression analysis of this variable.
# - Notice that some of the nice plots invite a line of best fit, such as alcohol vs density. Others, such as citric acid vs alcohol, are more inscrutable.

# So we now have called the .corr() method, and the .pairplot() Seaborn method, on our wine data. Both have flaws. Happily, we can get the best of both worlds with a heatmap. 

# In[11]:


# Make a heatmap of the data 
correlation_matrix = df.corr()
_ = sns.heatmap(correlation_matrix)


# Take a moment to think about the following questions:
# - How does color relate to extent of correlation?
# - How might we use the plot to show us interesting relationships worth investigating? 
# - More precisely, what does the heatmap show us about the fixed acidity variable's relationship to the density variable? 
# 
# There is a relatively strong correlation between the density and fixed acidity variables respectively. In the next code block, call the scatterplot() method on our sns object. Make the x-axis parameter 'density', the y-axis parameter 'fixed.acidity', and the third parameter specify our wine dataset.  

# In[12]:


# Plot density against alcohol
_ = sns.scatterplot(x='density', y='fixed.acidity', data=df)


# We can see a positive correlation, and quite a steep one. There are some outliers, but as a whole, there is a steep looking line that looks like it ought to be drawn. 

# In[13]:


# Call the regplot method on your sns object, with parameters: x = 'density', y = 'fixed.acidity'
_ = sns.regplot(x='density', y='fixed.acidity', data=df)


# The line of best fit matches the overall shape of the data, but it's clear that there are some points that deviate from the line, rather than all clustering close. 

# Let's see if we can predict fixed acidity based on density using linear regression. 

# ### 3. Modeling 

# #### 3a. Train/Test Split
# While this dataset is super clean, and hence doesn't require much for analysis, we still need to split our dataset into a test set and a training set.
# 
# You'll recall from p.158 of *AoS* that such a split is important good practice when evaluating statistical models. On p.158, Professor Spiegelhalter was evaluating a classification tree, but the same applies when we're doing regression. Normally, we train with 75% of the data and test on the remaining 25%. 
# 
# To be sure, for our first model, we're only going to focus on two variables: fixed acidity as our dependent variable, and density as our sole independent predictor variable. 
# 
# We'll be using [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) here. Don't worry if not all of the syntax makes sense; just follow the rationale for what we're doing. 

# In[14]:


# Subsetting our data into our dependent and independent variables.
y = df['fixed.acidity']
X = df[['density']]

# Split the data. This line uses the sklearn function train_test_split().
# The test_size parameter means we can train with 75% of the data, and test on 25%. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[15]:


# We now want to check the shape of the X train, y_train, X_test and y_test to make sure the proportions are right. 
for i in [X_train, X_test, y_train, y_test]:
    print(i.shape)


# #### 3b. Making a Linear Regression model: our first model
# Sklearn has a [LinearRegression()](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) function built into the linear_model module. We'll be using that to make our regression model. 

# In[16]:


# Create the model: make a variable called rModel, and use it linear_model.LinearRegression appropriately
rModel = linear_model.LinearRegression()


# In[17]:


# We now want to train the model on our test data.
rModel.fit(X_train, y_train)


# In[18]:


# Evaluate the model  
rModel.score(X_train, y_train)


# The above score is called R-Squared coefficient, or the "coefficient of determination". It's basically a measure of how successfully our model predicts the variations in the data away from the mean: 1 would mean a perfect model that explains 100% of the variation. At the moment, our model explains only about 23% of the variation from the mean. There's more work to do!

# In[19]:


# Use the model to make predictions about our test data
y_pred = rModel.predict(X_test)


# In[20]:


# Let's plot the predictions against the actual result. Use scatter()
_ = plt.scatter(y_test, y_pred)
_ = plt.xlabel('Actual acidity')
_ = plt.ylabel('Predicted acidity')


# The above scatterplot represents how well the predictions match the actual results. 
# 
# Along the x-axis, we have the actual fixed acidity, and along the y-axis we have the predicted value for the fixed acidity.
# 
# There is a visible positive correlation, as the model has not been totally unsuccesful, but it's clear that it is not maximally accurate: wines with an actual fixed acidity of just over 10 have been predicted as having acidity levels from about 6.3 to 13.

# Let's build a similar model using a different package, to see if we get a better result that way.

# #### 3c. Making a Linear Regression model: our second model: Ordinary Least Squares (OLS)

# In[21]:


# Create the test and train sets. Here, we do things slightly differently.  
# We make the explanatory variable X as before.
X = df[['density']]

# But here, reassign X the value of adding a constant to it. This is required for Ordinary Least Squares Regression.
# Further explanation of this can be found here: 
# https://www.statsmodels.org/devel/generated/statsmodels.regression.linear_model.OLS.html
X = sm.add_constant(X)


# In[22]:


# The rest of the preparation is as before.

# Split the data using train_test_split()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[23]:


# Create the model
model2 = sm.OLS(y_train, X_train)

# Fit the model with fit() 
rModel2 = model2.fit()


# In[24]:


# Evaluate the model with .summary()
rModel2.summary()


# One of the great things about Statsmodels (sm) is that you get so much information from the summary() method. 
# 
# There are lots of values here, whose meanings you can explore at your leisure, but here's one of the most important: the R-squared score is 0.455, the same as what it was with the previous model. This makes perfect sense, right? It's the same value as the score from sklearn, because they've both used the same algorithm on the same data.
# 
# Here's a useful link you can check out if you have the time: https://www.theanalysisfactor.com/assessing-the-fit-of-regression-models/

# In[25]:


# Let's use our new model to make predictions of the dependent variable y. Use predict(), and plug in X_test as the parameter
y_pred = rModel2.predict(X_test)


# In[26]:


# Plot the predictions
# Build a scatterplot
_ = plt.plot(y_test, y_pred, marker='.', linestyle='none')

# Add a line for perfect correlation. Can you see what this line is doing? Use plot()
test_range = [min(y_test), max(y_test)]
_ = plt.plot(test_range, test_range)

# Label it nicely
_ = plt.xlabel('Actual acidity')
_ = plt.ylabel('Predicted acidity')


# The red line shows a theoretically perfect correlation between our actual and predicted values - the line that would exist if every prediction was completely correct. It's clear that while our points have a generally similar direction, they don't match the red line at all; we still have more work to do. 
# 
# To get a better predictive model, we should use more than one variable.

# #### 3d. Making a Linear Regression model: our third model: multiple linear regression
# Remember, as Professor Spiegelhalter explains on p.132 of *AoS*, including more than one explanatory variable into a linear regression analysis is known as ***multiple linear regression***. 

# In[27]:


# Create test and train datasets
# This is again very similar, but now we include more columns in the predictors
# Include all columns from data in the explanatory variables X except fixed.acidity and quality (which was an integer)
X = df.drop(['fixed.acidity','quality'], axis=1)
y = df['fixed.acidity']

# Create constants for X, so the model knows its bounds
X = sm.add_constant(X)


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[28]:


# We can use almost identical code to create the third model, because it is the same algorithm, just different inputs
# Create the model
model3 = sm.OLS(y_train, X_train)

# Fit the model
rModel3 = model3.fit()


# In[29]:


# Evaluate the model
rModel3.summary()


# The R-Squared score shows a big improvement - our first model predicted only around 45% of the variation, but now we are predicting 87%!

# In[30]:


# Use our new model to make predictions
y_pred = rModel3.predict(X_test)


# In[31]:


# Plot the predictions
# Build a scatterplot
plt.plot(y_test, y_pred, marker='.', linestyle='none')

# Add a line for perfect correlation
test_range = [min(y_test), max(y_test)]
plt.plot(test_range, test_range)

# Label it nicely
plt.xlabel('Actual acidity')
plt.ylabel('Predicted acidity')


# We've now got a much closer match between our data and our predictions, and we can see that the shape of the data points is much more similar to the red line. 

# We can check another metric as well - the RMSE (Root Mean Squared Error). The MSE is defined by Professor Spiegelhalter on p.393 of *AoS*, and the RMSE is just the square root of that value. This is a measure of the accuracy of a regression model. Very simply put, it's formed by finding the average difference between predictions and actual values. Check out p. 163 of *AoS* for a reminder of how this works. 

# In[32]:


# Define a function to check the RMSE. Remember the def keyword needed to make functions? 
def rmse(test, pred):
    return np.sqrt(np.mean((test-pred) ** 2))
rmse(y_test, y_pred)


# In[33]:


# Get predictions from rModel3
y_pred

# Put the predictions & actual values into a dataframe
model3_df = pd.DataFrame({'actuals': y_test, 'preds': y_pred})
model3_df.head()


# The RMSE tells us how far, on average, our predictions were mistaken. An RMSE of 0 would mean we were making perfect predictions. 0.6 signifies that we are, on average, about 0.6 of a unit of fixed acidity away from the correct answer. That's not bad at all.

# #### 3e. Making a Linear Regression model: our fourth model: avoiding redundancy 

# We can also see from our early heat map that volatile.acidity and citric.acid are both correlated with pH. We can make a model that ignores those two variables and just uses pH, in an attempt to remove redundancy from our model.

# In[34]:


# Create test and train datasets
# Include the remaining six columns as predictors
X = df.drop(['fixed.acidity', 'quality', 'volatile.acidity', 'citric.acid'], axis=1)

# Create constants for X, so the model knows its bounds
sm.add_constant(X)

# Split the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[35]:


# Create the fifth model
model4 = sm.OLS(y_train, X_train)
# Fit the model
rModel4 = model4.fit()
# Evaluate the model
rModel4.summary()


# The R-squared score has reduced, showing us that actually, the removed columns were important.

# ### Conclusions & next steps

# Congratulations on getting through this implementation of regression and good data science practice in Python! 
# 
# Take a moment to reflect on which model was the best, before reading on.
# 
# .
# .
# .
# 
# Here's one conclusion that seems right. While our most predictively powerful model was rModel3, this model had explanatory variables that were correlated with one another, which made some redundancy. Our most elegant and economical model was rModel4 - it used just a few predictors to get a good result. 
# 
# All of our models in this notebook have used the OLS algorithm - Ordinary Least Squares. There are many other regression algorithms, and if you have time, it would be good to investigate them. You can find some examples [here](https://www.statsmodels.org/dev/examples/index.html#regression). Be sure to make a note of what you find, and chat through it with your mentor at your next call.
# 
