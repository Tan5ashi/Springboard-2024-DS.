

---

# **Boston Housing Prices Prediction Report**

## **1. Introduction**

This report details the process of predicting Boston housing prices (MEDV) using various regression models. The dataset contains information on housing attributes in Boston, and our goal is to predict the median value of owner-occupied homes.

## **2. Data Import and Initial Analysis**

### **2.1 Importing Libraries**

The following libraries were imported for data analysis and machine learning tasks:

- `numpy` and `pandas` for data manipulation
- `matplotlib` and `seaborn` for visualization
- `statsmodels` for statistical modeling
- `scipy.stats` for statistical tests
- `sklearn` for machine learning models and data processing

### **2.2 Loading the Dataset**

The dataset was loaded from a CSV file with the following column names:

```python
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataset = pd.read_csv('/kaggle/input/boston-house-prices/housing.csv', delimiter=r'\s+', names=column_names)
```

### **2.3 Exploring the Dataset**

- **Shape of the Dataset:** (506 rows, 14 columns)
- **Top 5 Rows:** The dataset contains features such as crime rate, proportion of residential land, and more.

### **2.4 Descriptive Statistics**

The dataset's descriptive statistics show details such as mean, standard deviation, and percentiles for each feature. Key observations include:

- `ZN` and `CHAS` are categorical features with skewed distributions.
- `MEDV` has a maximum value of 50,000, indicating potential censoring.

## **3. Data Preprocessing**

### **3.1 Handling Missing Values**

The dataset was checked for missing values and none were found.

### **3.2 Outlier Detection and Treatment**

Outliers were detected using boxplots and the Interquartile Range (IQR) method. Features with high percentages of outliers include:

- `CRIM` (13.04%)
- `B` (15.22%)
- `MEDV` (7.91%)

#### **Outlier Treatment Strategy**

1. **Removing Extreme Outliers:** Extreme outliers were removed from `B` and `CRIM`.
2. **Replacing Remaining Outliers:** Remaining outliers were replaced with the mean for non-censored variables and a maximum of 50 for `MEDV`.

### **3.3 Feature Selection**

#### **3.3.1 P-Value Based Feature Selection**

Using backward elimination with a significance level of 0.05, the final features selected were:

- `NOX`
- `RM`
- `DIS`
- `RAD`
- `TAX`
- `PTRATIO`
- `LSTAT`

#### **3.3.2 Multicollinearity Check**

A correlation heatmap was used to detect multicollinearity. Highly correlated features were:

- `TAX` and `RAD` (0.86)
- `DIS` and `NOX` (0.75)

Based on this analysis:

- `TAX` was dropped in favor of `RAD`.
- `NOX` was dropped in favor of `DIS`.

#### **3.3.3 Final Feature Selection**

The remaining features after removing multicollinearity and checking correlation with `MEDV` were:

- `RM`
- `PTRATIO`
- `LSTAT`

## **4. Machine Learning Models**

### **4.1 Model Implementation**

The following regression models were implemented:

1. **Linear Regression**
2. **Polynomial Regression** (degree 3)
3. **Support Vector Regression** (SVR) with polynomial kernel
4. **Random Forest Regression** (100 estimators)
5. **K-Nearest Neighbors Regression** (13 neighbors)

### **4.2 Model Training and Prediction**

Each model was trained on the training set and evaluated on the test set. The comparison of actual vs. predicted values was performed for each model.

### **4.3 Model Performance**

#### **R-Squared Scores**

The R-squared scores for the models using K-fold cross-validation (5 folds) were:

- **Linear Regression:** 0.51
- **Polynomial Regression:** 0.64
- **Support Vector Regression:** 0.50
- **Random Forest Regression:** 0.72
- **K-Nearest Neighbors Regression:** 0.64

### **4.4 Model Comparison**

A bar chart comparing the R-squared scores of the different models indicates that:

- **Random Forest Regression** has the highest score (0.72), making it the best-performing model.
- **Polynomial Regression** and **K-Nearest Neighbors Regression** also performed well with scores of 0.64.
- **Linear Regression** and **Support Vector Regression** had lower scores (0.51 and 0.50, respectively).

## **5. Conclusion**

Based on the analysis and model performance:

- **Key Features:** `RM`, `PTRATIO`, and `LSTAT` are the most relevant features for predicting `MEDV`.
- **Best Model:** **Random Forest Regression** provides the best accuracy with an R-squared score of 0.72.
- **Alternative Models:** Polynomial and K-Nearest Neighbors Regression models also show comparable performance and can be considered for prediction.

## **6. Recommendations**

- **Model Selection:** Use the Random Forest Regression model for the most accurate predictions.
- **Feature Engineering:** Explore additional feature engineering techniques and interactions to improve model performance.
- **Further Validation:** Perform additional validation and tuning to ensure robustness and avoid overfitting.

## **7. Future Work**

- **Hyperparameter Tuning:** Optimize model parameters for better performance.
- **Data Augmentation:** Consider using more data if available to enhance model accuracy.
- **Exploratory Data Analysis (EDA):** Further investigate data distributions and relationships to refine feature selection and model choice.

---

This detailed report summarizes the data preprocessing, feature selection, model implementation, and performance evaluation, providing insights into the best practices for predicting housing prices in Boston.
