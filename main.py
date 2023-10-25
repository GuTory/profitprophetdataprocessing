#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime as dt
import os
import sys
import tensorflow
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def cropped_data(df, start=dt.datetime(1990, 1, 1), end=dt.datetime(2024, 1, 1)):
    date_filtered_data = df[(df['Date'] > start) & (df['Date'] < end)]
    return date_filtered_data


def save_scaler(scaler, name):
    path = f"helper/{name}.joblib"
    joblib.dump(scaler, open(path, "wb"))


def load_scaler(name):
    path = f"helper/{name}.joblib"
    with open(path, 'rb') as file:
        scaler = joblib.load(file)
        return scaler


def save_model(model):
    path = "helper/model.h5"
    model.save(path, overwrite=True)


def load_model_from_local():
    path = "helper/model.h5"
    return tensorflow.keras.saving.load_model(path)


def search_ticker(ticker):
    try:
        path = f'data/stocks/{ticker}.csv'
        if not os.path.isfile(path):
            path = path.replace('stocks', 'etfs')
        df = pd.read_csv(path, parse_dates=['Date'])
        df["Difference"] = df["Close"].diff()
        cropped = cropped_data(df)
        df = cropped.drop("Date", axis="columns")
        return df

    except ValueError:
        print("inappropriate ticker")


def main():
    try:
        print(tensorflow.__version__)
        ticker = sys.argv[1].upper()
        df = search_ticker(ticker)
        df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)
        scaler = load_scaler("scaler")
        close_scaler = load_scaler("close_scaler")
        scaled_test = scaler.transform(np.array(df_test))

        past = 14

        X_test, y_test = [], []
        for i in range(past, len(scaled_test)):
            X_test.append(scaled_test[i - past:i])
            y_test.append(scaled_test[i])
        X_test, y_test = np.array(X_test), np.array(y_test)
        model = load_model_from_local()
        test_predict = model.predict(X_test)
        test_predict = close_scaler.inverse_transform(test_predict)
        print(test_predict[len(test_predict)-1][0])
    except Exception as e:
        print(f"Wrong parameters: {e}")
    return 0


if __name__ == "__main__":
    main()
