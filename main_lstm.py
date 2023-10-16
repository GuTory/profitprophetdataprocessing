#!/usr/bin/env python
# -*- coding: utf-8 -*-
import keras.callbacks
import pandas as pd
from datetime import datetime as dt
import numpy as np
import tensorflow as tf
from main import log
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import RMSprop, Adam, SGD
import matplotlib.pyplot as plt
from main_train_test_scale import load_scaler

RUN_NAME = "LSTM_32-64-32_all_data"


# Splitting the dataframe for Close column
def create_past_and_future_array(df: pd.DataFrame, future, past):
    log("cropping df to have a faster processing of data")
    # df = df.head(100_000)

    log(df.shape)
    X = []
    Y = []
    for i in range(past, len(df) - future + 1):
        X.append(df.iloc[i - past:i, 0:df.shape[1]])
        Y.append(df["Close"].iloc[i + future - 1:i + future])
    return np.array(X), np.array(Y)


def read_df_and_create_array(path, future, past):
    df = pd.read_csv(path)
    X, Y = create_past_and_future_array(df=df, future=future, past=past)
    log(f"describing X_test: {X.shape}")
    log(f"describing Y_test: {Y.shape}")
    return X, Y


def main():
    # Defining file path
    train_path = "data_train\\aggregated\\train_aggr_scaled.csv"
    test_path = "data_test\\aggregated\\test_aggr_scaled.csv"

    # Loading dataframes
    log("loading dataframes")
    future = 1
    past = 14

    X_test, Y_test = read_df_and_create_array(test_path, future=future, past=past)
    X_train, Y_train = read_df_and_create_array(train_path, future=future, past=past)

    adam = Adam()
    rmsprop = RMSprop()
    sgd = SGD()

    # Creating neural network model
    log(f"creating LSTM model")
    model = Sequential()
    model.add(
        LSTM(
            32,
            activation="relu",
            input_shape=(X_train.shape[1], X_train.shape[2]),
            return_sequences=True,
            name="lstm_1"
        )
    )
    model.add(
        LSTM(
            64,
            activation="relu",
            return_sequences=True,
            name="lstm_2"
        )
    )
    model.add(
        LSTM(
            32,
            activation="relu",
            return_sequences=False,
            name="lstm_3"
        )
    )
    model.add(
        Dense(Y_train.shape[1], name="dense")
    )

    model.compile(optimizer=adam, loss='mse')
    log(model.summary())

    logger = keras.callbacks.TensorBoard(
        log_dir=f"logs/{RUN_NAME}",
        write_graph=True,
        histogram_freq=128
    )

    history = model.fit(
        X_train,
        Y_train,
        epochs=6,
        batch_size=128,
        validation_split=0.2,
        verbose=1,
        callbacks=[logger]
    )

    test_error_rate = model.evaluate(
        X_test,
        Y_test,
        verbose=0
    )


    log("Plotting data")
    plt.plot(history.history["loss"], label="training loss")
    plt.plot(history.history["val_loss"], label="validation loss")
    plt.show()

    scaler = load_scaler("helper\\scaler.pkl")

    """
    model_builder = tf.saved_model.builder.SavedModelBuilder("exported_model")

    inputs = {
        'input': tf.saved_model.utils.build_tensorinfo(model.input)
    }
    outputs = {
        'close': tf.saved_model.utils.build_tensor_info(model.output)
    }
    signature_def = tf.saved_model.signature_def_utils.build_signature_def(
        inputs=inputs,
        outputs=outputs,
        method_name=tf.saved_model.signature_constants.predict
    )
    model_builder.add_meta_graph_and_variables(
        K.get_session(),
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map= {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
        }
    )
    """
    return 0


if __name__ == "__main__":
    main()
