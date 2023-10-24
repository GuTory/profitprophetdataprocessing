#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from datetime import datetime as dt
from keras.models import Sequential
import tensorflow as tf
import sys
import pickle


def save_scaler(scaler):
    path = "helper/scaler.pkl"
    pickle.dump(scaler, open(path, "wb"))


def load_scaler():
    path = "helper/scaler.pkl"
    with open(path, 'rb') as file:
        scaler = pickle.load(file)
        return scaler


def save_model(model: Sequential):
    path = "helper/model.h5"
    model.save(path, overwrite=True)


def load_model():
    path = "helper/model.h5"
    return tf.keras.saving.load_model(path)


def main():
    try:
        ticker = sys.argv[0]

    except ValueError:
        print("no parameters provided")

    return 0


if __name__ == "__main__":
    main()
