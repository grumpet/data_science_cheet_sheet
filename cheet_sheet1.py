import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('clubmed_HW2.csv') # Load the dataset
print(df.isnull().sum())  # Check for missing values
df.info()  # Summary of the DataFrame
# status categorical value -> ['single' 'couple' nan 'family'] has 7 missing values
# room_price numerical value -> 3 missing values
# visits2016 numerical value -> 30 missing values
# club_member categorical value -> [False True nan] has 7 missing values


#df = df.dropna()# Drop rows with missing values
df = df.dropna(thresh=2)  # Drop rows with at least 2 missing values
#df_with_status = df.dropna(subset=['status'])  # Drop rows with missing values in the status column
print(df['status'].unique())  # Unique values in the status column
print(df['club_member'].unique())  # Unique values in the club_member column
df['status'] = df['status'].fillna('unkown')  # Fill missing values in the status column with 'unkown'
df = df.dropna(subset=['club_member'])  # Drop rows with missing values in the club_member column
#visits2016 numerical value -> 30 missing values [ 0.  1.  2. nan  3.]
df['visits2016'] = df['visits2016'].mask(df['visits2016'].isna(), 0) # Fill missing values in the visits2016 column with 0
replace_option = df['visits2016'].replace(to_replace=np.nan,value=0)  # Fill missing values in the visits2016 column with 0 saves the result in replace_option
interpolate_option = df['visits2016'].interpolate()  # Fill missing values in the visits2016 column with linear interpolation saves the result in interpolate_option
df['room_price'] = df['room_price'].fillna(df['room_price'].mean())  # Fill missing values in the room_price column with the mean
#mean -> normaly distributed data
#median -> skewed data
#mode -> categorical data
#interpolation -> time series data


#discretization -> converting continuous data into discrete data by creating bins or categories
#to use when we want to convert a continuous variable into a categorical one
df['room_price_bins'] = pd.cut(df['room_price'], bins=3, labels=['low', 'medium', 'high'])  # Discretize the room_price column into 3 bins
#creating a bar plot
count_bins = df['room_price_bins'].value_counts()  # Count of each bin
count_bins.plot(kind='bar')  # Create a bar plot
plt.show()

# creating a cross-tabulation to see the relationship between two categorical variables
cross_tab = pd.crosstab(df['status'], df['club_member'],normalize='index' , margins=True)  # Create a cross-tabulation of status and club_member
#normalize='index' -> normalize the values by row so that the sum of each row is 1 
#margin=True -> add row and column totals
cross_tab.plot(kind='bar', stacked=True)  # Create a stacked bar plot
plt.show()