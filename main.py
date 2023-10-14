#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from datetime import datetime as dt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


# Logging a message with printing the current time also
def log(message):
    print(f"[{dt.now()}] --- {message}")


# Splitting the dataframe for Close column
def drop_close_and_split(df: pd.DataFrame):
    X = df.drop("Close", axis=1)
    Y = df["Close"]
    return X, Y


def shift_close(df: pd.DataFrame, shift = 1):
    Y = df["Close"].shift(-shift)
    return df, Y


def main():
    # Defining file path
    train_path = "data_train\\aggregated\\train_aggr_scaled.csv"
    test_path = "data_test\\aggregated\\test_aggr_scaled.csv"

    # Loading dataframes
    log("loading dataframes")
    shift = 1

    df_train = pd.read_csv(train_path)
    X_train, Y_train = shift_close(df_train, shift)
    log(f"describing X_train: \n{X_train.head()}")
    log(f"describing Y_train: \n{Y_train.head()}")

    df_test = pd.read_csv(test_path)
    X_test, Y_test = shift_close(df_test, shift)
    log(f"describing X_train: \n{X_test.head()}")
    log(f"describing Y_train: \n{Y_test.head()}")

    # Creating neural network model
    log(f"creating neural network model input shape={df_train.shape}")
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
    # model.add(Dropout(0.1))
    # model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1))

    # Optimizer and metrics
    optimizer = RMSprop(0.001)
    metrics = ['mae', 'accuracy']
    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=metrics)
    log(model.summary())

    # Training the model
    log("training the model")
    model.fit(
        X_train,
        Y_train,
        epochs=8,
        batch_size=256,
        shuffle=False,
        verbose=2
    )

    # Using testing data to see how precise the model is
    log("evaluating on test data")
    test_error_rate = model.evaluate(
        X_test,
        Y_test,
        verbose=0
    )

    log(f"test error rate was: {test_error_rate}")
    return 0


if __name__ == "__main__":
    main()
