import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

    
df = pd.read_csv('AAPL_daily.csv')

#df['Name'] = -(df['Timestamp'].shift(1) - df['Timestamp'])
#df['Forward'] = df['Close'].shift(-1)

#df = df[df.Name <= 120.0]

df['Day_Name'] = pd.to_datetime(df['Date'])
df['New'] = df['Day_Name'].shift(-1) - df['Day_Name']
#df['New2'] = df['New'].astype('int')


print(df.head(15))

df.plot(x='Date',y=['Volume','Close'])
plt.show()


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