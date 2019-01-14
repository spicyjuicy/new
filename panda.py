import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

    
df = pd.read_csv('FB.csv')

df['Day_Name'] = pd.to_datetime(df['Date'])


#df['Volume'] = df['Volume']/df['Volume'].max()
df['WeekNum'] = df['Day_Name'].dt.week
df['YearNum'] = df['Day_Name'].dt.year
df = df.drop(columns=['Date','Day_Name'])


#-for column in df:
#    if df[column].dtype == 'float64':
#        print(1)
#        df[column] = df[column]/df[column].max()

print(df.head(25))
#df.plot(x='Date',y=['Volume','Close'])
#plt.show()

#print(df['Volume'].mean())
print(df.dtypes,len(df.columns))
'''
print(len(df))

df = df.values


print(df.shape)

df = np.split(df,100)

print(df.shape)

#df = df.drop(df.index[:2000000])

#df = np.split(df,100)
#print(df.shape,df[0])

df.plot(x='Timestamp',y='Close')
plt.show()


#86400 minutes in a day
'''