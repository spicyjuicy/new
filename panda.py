import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




    
df = pd.read_csv('FB.csv')

df['Day_Name'] = pd.to_datetime(df['Date'])


df['WeekNum'] = df['Day_Name'].dt.week
df['YearNum'] = df['Day_Name'].dt.year
#df = df.drop(columns=['Date','Day_Name'])


#-for column in df:
#    if df[column].dtype == 'float64':
#        print(1)
#        df[column] = df[column]/df[column].max()

print(df.head(25))

print(df.dtypes,len(df.columns))
