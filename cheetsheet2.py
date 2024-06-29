import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

df = pd.read_csv('Bridges.csv') # Load the dataset
df['MATERIAL'] = df['MATERIAL'].fillna('unkown')# Fill missing values in the status column with 'unkown'
le = preprocessing.LabelEncoder()
le.fit(df['MATERIAL'])
#['WOOD' 'IRON' 'STEEL' 'unkown'] ->[2 0 1 3]
df['MATERIAL'] = le.transform(df['MATERIAL'])

df.info()  # Summary of the DataFrame
for coloumn in df.columns:
    print(coloumn)
    print(df[coloumn].unique())  # Unique values in the status column
    
df = df.dropna(subset=['PURPOSE']) # Drop rows with missing values in the SPAN column
dummy = pd.get_dummies(df['PURPOSE'] , prefix='PURPOSE')  # Create dummy variables for the PURPOSE column
df = pd.concat([df, dummy], axis=1)  # Concatenate the dummy variables with the original DataFrame
df = df.drop('PURPOSE', axis=1)  # Drop the original PURPOSE column



from category_encoders import OrdinalEncoder
df = df.dropna(subset=['SPAN']) # Drop rows with missing values in the SPAN column
maplist = {'SHORT':0, 'MEDIUM':1, 'LONG':2}
oe = OrdinalEncoder(mapping=[{'col':'SPAN','mapping':maplist}])
df['SPAN'] = oe.fit_transform(df['SPAN']) #span short -> 0, medium -> 1, long -> 2
