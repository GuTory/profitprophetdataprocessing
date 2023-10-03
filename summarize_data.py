import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime


def merge_all_data(dir_stocks, dir_etfs):
    df = iterate_through_directories([dir_stocks, dir_etfs])
    df.to_csv(os.path.join("data_sum\\", "data.csv"), index=False)
    print("done.")


def iterate_through_directories(list_of_directories):
    df = pd.DataFrame()
    for element in list_of_directories:
        for filename in os.listdir(element):
            f = os.path.join(element, filename)
            if os.path.isfile(f):
                created_df = extract_ticker_and_create_df(f, filename)
                df = pd.concat([df, created_df])
    return df


def extract_ticker_and_create_df(f, filename):
    ticker = filename.replace(".csv", "")
    return read_data_and_append(ticker, f)


def read_data_and_append(ticker, file):
    df = pd.read_csv(file)
    df['Ticker'] = df.apply(lambda row: ticker, axis=1)
    print(f"{ticker} formatted")
    return df
