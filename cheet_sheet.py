import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

a = np.array([[1, 2, 3, 4, 5],
              [6, 7, 8, 9, 10]])
a.dtype  # data type of the array
a.shape  # shape of the array -> (2,5)
a.ndim  # number of dimensions -> 2
a.size  # number of elements -> 10
a.reshape(10, 1)  # reshaping the array -> (10, 1)
a.flatten()  # flattening the array -> [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
np.arange(1, 10, 2)  # array of numbers from 1 to 10 with step 2 -> [1, 3, 5, 7, 9]
np.zeros((2, 3))  # 2x3 array of zeros
np.ones((2, 3))  # 2x3 array of ones
b = [11, 12, 13, 14, 15]
c=np.concatenate((a, [b]), axis=0)  # concatenating arrays along the rows



df = pd.read_csv('sales_data.csv') # Load the dataset
df.head()  # First 5 rows
df.head(10)  # First 10 rows
df.columns  # Get the column names
df.tail()  # Last 5 rows
df.shape   # Get the number of rows and columns
df.info()  # Summary of the DataFrame
df.loc[0]  # Row with label 0
df.iloc[0]  # First row
df[0:5]  # First 5 rows
df.values  # Get the values as a numpy array
df.size #number of elements
c=df.axes # Get the row and column labels
c[0] # row labels
c[1] # column labels
df.describe()  # Summary statistics of the DataFrame
df.dtypes  # Data types of the columns
df['Category'].unique()  # Unique values in the Category column
df['Category'].value_counts()  # Frequency of each unique value in the Category column
df.groupby('Category')['Price'].mean()  # Average price by category
df['Price'].mean()  # Average price
df['Price'].max()  # Maximum price
df['Price'].min()  # Minimum price
df['Price'].std()  # Standard deviation of price
df['Price'].sum()  # Total price
df['Price'].median()  # Median price
df['Price'].mode()  # Mode of price most common value
df['Price'].quantile(0.25)  # 25th percentile of price
df['Price'].quantile(0.75)  # 75th percentile of price
above_50 = df[df['Price']>50]  #saves only the rows where the price is above 50
above_50_and_category_Electrical = df[(df['Price']>50) & (df['Category']=='Electrical')]  #saves only the rows where the price is above 50 and the category is Electrical


#outliers
Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['Price'] < lower_bound) | (df['Price'] > upper_bound)]  #saves the outliers in the price column
df = df[(df['Price'] > lower_bound) & (df['Price'] < upper_bound)]#removing the outliers






#for normalizing the data and visualizing the distribution
plt.figure(figsize=(10, 6))  # Set the figure size
plt.hist(df['Price'], bins=20)  # Histogram of Price
plt.xlabel('Price')  # X-axis label
plt.ylabel('Frequency')  # Y-axis label
plt.title('Price Distribution')  # Title of the plot
plt.show()



#bar chart using seaborn
#takes all the unique values in the category column and counts the frequency of each value
sns.countplot(x='Category', data=df)
plt.show()

#scatter plot visualizing the relationship between two numerical columns

plt.scatter(df['Price'], df['Quantity'])
plt.xlabel('Price')
plt.ylabel('Quantity')
plt.title('Price vs Quantity')
plt.show()







# print(data.isnull().sum())  # check for missing values
# data = data.dropna()  # drop missing values
# print(data.isnull().sum())



# data['Total_Sales'] = data['Quantity'] * data['Price']
# category_sales = data.groupby('Category')['Total_Sales'].sum()
# print(f"Total Sales by Category:{category_sales}")

# MINIMUM_PRICE = 50  # price threshold
# high_price_df = data[data['Price'] > MINIMUM_PRICE]

# print(f"high_price_df{high_price_df}")


# electronics_high_quantity = data[(data['Category'] == 'Electronics') & (data['Quantity'] > 3)]
# print(f"electronics_high_quantity{electronics_high_quantity}")


# for i in range(10,20):
#     print(data.loc[i])
    
# print(data['Category'].unique())

# dummy = pd.get_dummies(data['Category'], prefix='Category', drop_first=True)
# print(dummy)
# data = pd.concat([data, dummy], axis=1)
# print(data.head()) 
# print(data.columns)   


# data['price_category'] = pd.cut(data['Price'], bins=[0, 50, 100, np.inf], labels=['Low', 'Medium', 'High'])

# print(data['Quantity'])

# plt.hist(data['Quantity'], bins=20,edgecolor='black')
# plt.show()
