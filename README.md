<div align="center">
  <h1>Customer Credit Card Usage Segmentation Analysis</h1>
</div>


<div align="center">
This project involves analyzing customer credit card usage and transaction behavior using a dataset that includes variables such as balance, purchases, cash advances, payments, and credit limits.
</div>



<div align="center">
  <img src="https://github.com/Ras-codes/Customer-Credit-Card-Usage-Segmentation-Analysis/assets/164164852/9abea4c2-6464-42a9-aad0-e6cff7bd2d04">
</div>



## Tools

- **Programming Language**: Python 🐍
- **IDE**: Jupyter Notebook 📓
- **Data Manipulation and Analysis**:
  - NumPy 📊
  - pandas 🐼
- **Data Visualization**:
  - Matplotlib 📊
  - Seaborn 📈
- **Statistical Analysis**: 
  - scipy.stats 📈
- **Clustering**:
  - sklearn.decomposition (PCA) 🧩
  - sklearn.cluster (KMeans) 🔍
- **Data Preprocessing**:
  - sklearn.preprocessing (StandardScaler) ⚖️

## Dataset Description: 

### Table- CC GENERAL


### Variables-

| Variable Name                        | Data Type | Description                                                  |
|--------------------------------------|-----------|--------------------------------------------------------------|
| CUST_ID                              | int       | Unique identifier for each customer                          |
| BALANCE                              | float     | The balance amount on the credit card                        |
| BALANCE_FREQUENCY                    | float     | How frequently the balance is updated                        |
| PURCHASES                            | float     | Total purchase amount on the credit card                     |
| ONEOFF_PURCHASES                     | float     | Total amount of one-off purchases                            |
| INSTALLMENTS_PURCHASES               | float     | Total amount of installment purchases                        |
| CASH_ADVANCE                         | float     | Total cash advance amount                                    |
| PURCHASES_FREQUENCY                  | float     | Frequency of purchases                                       |
| ONEOFF_PURCHASES_FREQUENCY           | float     | Frequency of one-off purchases                               |
| PURCHASES_INSTALLMENTS_FREQUENCY     | float     | Frequency of installment purchases                           |
| CASH_ADVANCE_FREQUENCY               | float     | Frequency of cash advances                                   |
| CASH_ADVANCE_TRX                     | int       | Number of cash advance transactions                          |
| PURCHASES_TRX                        | int       | Number of purchase transactions                              |
| CREDIT_LIMIT                         | float     | Credit limit on the credit card                              |
| PAYMENTS                             | float     | Total payments made by the customer                          |
| MINIMUM_PAYMENTS                     | float     | Minimum payments made by the customer                        |
| PRC_FULL_PAYMENT                     | float     | Percentage of full payments                                  |
| TENURE                               | int       | Number of months the customer has been using the credit card |


# ------------------------------------------------------------------------------


# Segmentation

This repository contains code and resources for Customer Segmentation using clustering techniques on credit card data. The project demonstrates fundamental data manipulation and clustering techniques using Python programming language. It covers essential operations such as data loading, cleaning, handling missing values, data transformation, feature scaling, and clustering analysis using popular libraries like Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn. The repository includes Jupyter notebooks with step-by-step explanations, datasets for practice, and examples showcasing various data manipulation tasks, clustering methods such as K-Means, and data visualization graphs to interpret the clusters.


# ------------------------------------------------------------------------------


# Insights from the Dataset

- After importing the dataset, our first step is to check if the data is imported properly, we can use `DA.shape` to check the number of observations (rows) and features (columns) in the dataset
- Output will be : ![image](https://github.com/Ras-codes/Customer-Credit-Card-Usage-Segmentation-Analysis/assets/164164852/85052272-9dc8-4c74-8990-0bfdd6ebff3f)
- which means that the dataset contains 8950 records and 18 variables.
- We will now use `DA.head()` to display the top 5 observations of the dataset
- ![image](https://github.com/Ras-codes/Customer-Credit-Card-Usage-Segmentation-Analysis/assets/164164852/c9a09b10-707b-4a92-962b-e08a86ca511b)
- To understand more about the data, including the number of non-null records in each columns, their data types, the memory usage of the dataset, we use `DA.info()`
- ![image](https://github.com/Ras-codes/Customer-Credit-Card-Usage-Segmentation-Analysis/assets/164164852/5139379b-a895-4d50-ae78-77da4bdf8706)
- Checking descriptive statistics of data with `DA.describe()`
- ![image](https://github.com/Ras-codes/Customer-Credit-Card-Usage-Segmentation-Analysis/assets/164164852/d77a111e-a863-4496-90e9-c2fa06ace938)


# ------------------------------------------------------------------------------


# Handling Missing Values:

Next step is to check for missing values in the dataset. It is very common for a dataset to have missing values.

- `DA.isna().sum()` isna() is used for detecting missing values in the dataframe, paired with sum() will return the number of missing values in each column.
- ![image](https://github.com/Ras-codes/Customer-Credit-Card-Usage-Segmentation-Analysis/assets/164164852/75d38665-eb49-4565-9d6a-4beb56d86f09)
- Treating the missing values
````
DA['CREDIT_LIMIT'] = DA.CREDIT_LIMIT.fillna(DA['CREDIT_LIMIT'].median())
DA['MINIMUM_PAYMENTS'] = DA.MINIMUM_PAYMENTS.fillna(DA['MINIMUM_PAYMENTS'].median())
````
- Checking for duplicate data with `DA.duplicated().sum()`
- There is no duplicate data in our dataset.


# ------------------------------------------------------------------------------


# Standardization of data:


````
sc = StandardScaler()
DA_copy_scaled = sc.fit_transform(DA_copy)
DA_copy_scaled = pd.DataFrame(DA_copy)
DA_copy_scaled.head()
````
- StandardScaler is used to standardize the features of the dataset.
- fit_transform computes the mean and standard deviation, then scales the data.
- The scaled data is then converted back to a pandas DataFrame.


# ------------------------------------------------------------------------------


# PCA : Principle Component Analysis


````
pc = PCA(n_components = 17).fit(DA_copy_scaled)
pc.explained_variance_
````
- Initialization and Fit: We initialized the PCA model to reduce the dataset to 17 components and fit it to the standardized data.
````
sum(pc.explained_variance_)
pc.explained_variance_ /  sum(pc.explained_variance_)
pc.explained_variance_ratio_
````
- Explained Variance: We examined the variance explained by each principal component and the proportion of total variance they explain.
````
var = np.round(np.cumsum(pc.explained_variance_ratio_) * 100, 2)
pd.DataFrame({'Eigen_Values':pc.explained_variance_,
                   'VAR':np.round(pc.explained_variance_ratio_*100,2),
                     'CUM_VAR':var},index=range(1,18))
````
- Cumulative Variance: We calculated the cumulative variance explained by the components to understand how many components capture a significant portion of the total variance.
````
pc_final=PCA(n_components=6).fit(DA_copy_scaled)
pc_final.explained_variance_
````
- Component Selection: We selected 6 principal components based on cumulative variance and refitted the PCA model.
````
reduced_cr=pc_final.transform(DA_copy_scaled)
dimensions=pd.DataFrame(reduced_cr)
dimensions.columns=['C1','C2','C3','C4','C5','C6']
````
- Data Transformation: We transformed the data to the 6 principal components, creating a new DataFrame for these reduced dimensions.






















































































