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

    df = df.reindex(idx,fill_value=0.0)

    df['Date'] = df.index
    df = df.reset_index(drop=True)
    df['Day_Name'] = pd.to_datetime(df['Date'])
    df['DayNum'] = df['Day_Name'].dt.day
    df['WeekNum'] = df['Day_Name'].dt.week
    df['YearNum'] = df['Day_Name'].dt.year


    return df

def drop_head(df):
    df = df[df.WeekNum != df.WeekNum.iloc[0]]
    return df

def rank_days(df):

    df['major_index'] = (df.index -2) % 7
    df['rank'] = df.groupby(['YearNum'])['Date'].rank()
    df = df[df.major_index != 6]
    df = df[df.major_index != 0]

    return df

file = 'FB.csv'

df = setup_dataframe(file)
df = drop_head(df)
df = rank_days(df)

print(df.head(50))

print(df.dtypes,len(df.columns))
