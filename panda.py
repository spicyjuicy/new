import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fill_non_trading(dataframe):
    print(1)
    
def setup_dataframe(file):

    df = pd.read_csv(file)

    df['Day_Name'] = pd.to_datetime(df['Date'])
    print(df['Date'].iloc[0],df['Date'].iloc[-1])
    idx = pd.date_range(df['Date'].iloc[0],df['Date'].iloc[-1])

    df.index = pd.DatetimeIndex(df['Day_Name'])

    df = df.reindex(idx)



    df['Date'] = df.index
    df = df.reset_index(drop=True)
    df['Day_Name'] = pd.to_datetime(df['Date'])
    df['DayNum'] = df['Day_Name'].dt.day
    df['WeekNum'] = df['Day_Name'].dt.week
    df['YearNum'] = df['Day_Name'].dt.year

    #df['Difference'] = df['DayNum'].shift(-1) - df['DayNum']

    #df['rank'] = df.groupby(['YearNum','WeekNum'])['DayNum'].rank()
#df = df.drop(columns=['Date','Day_Name'])


#-for column in df:
#    if df[column].dtype == 'float64':
#        print(1)
#        df[column] = df[column]/df[column].max()

#df.loc[0] = [np.random.randint(-1,1) for n in range(len(df.columns))]

    print(df.head(25))

    print(df.dtypes,len(df.columns))

file = 'FB.csv'

setup_dataframe(file)
