import os
import pandas
import pandas as pd


def load_data(filename):
    df = pandas.read_csv(filename)
    tickers = df['Symbol']
    print(tickers.head())
    return tickers


def save_df(df: pd.DataFrame, filename):
    df.to_csv(filename, index=False)


def search_ticker(ticker):
    df = pandas.read_csv(os.path.join("data", "symbols.csv"))
    return search_dataframe(df, 'Symbol', ticker)


def search_dataframe(df, column, value):
    try:
        return df.loc[df[column] == value].index._data[0]
    except:
        return -1
