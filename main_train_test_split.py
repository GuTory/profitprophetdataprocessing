#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import pandas as pd

from archive_src.scale_data import convert_date_to_float
from sklearn.model_selection import train_test_split

def apply_date_transform(df):
    df['Date'] = df['Date'].apply(
        lambda x: convert_date_to_float(x))
    return df

def read_and_split_all_files(files):
    # file: 'data\etfs\....csv' vagy stocks
    for file in files:
        filename = file[file.rfind("\\") + 1:]

        df = pd.read_csv(file)
        df = apply_date_transform(df)
        df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)
        df_train.to_csv(file.replace("data", "data_train"), index=False)
        df_test.to_csv(file.replace("data", "data_test"), index=False)
        print(f"{filename} printed to csv")



def see_all_files(directory):
    list = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            list.append(f)
    return list

def main():
    etfs = "data\\etfs"
    stocks = "data\\stocks"
    files = see_all_files(etfs) + see_all_files(stocks)
    read_and_split_all_files(files)
    return 0


if __name__ == "__main__":
    main()
