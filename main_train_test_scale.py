import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
from datetime import datetime as dt


def load_scaler(path):
    with open(path, 'rb') as file:
        scaler = pickle.load(file)
        print(scaler.scale_)
        print(scaler.min_)


def main():
    # defining important paths
    train = "data_train\\aggregated\\train_aggr.csv"
    train_scaled = "data_train\\aggregated\\train_aggr_scaled.csv"
    test = "data_test\\aggregated\\test_aggr.csv"
    test_scaled = "data_test\\aggregated\\test_aggr_scaled.csv"

    # Loading data and logging timestamp
    print(f"[{dt.now()}] --- loading data")
    df_train = pd.read_csv(train)
    df_test = pd.read_csv(test)

    # Creating scaler with a feature range of 0-1
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Scaling training data with scaler with fit and transform
    print(f"[{dt.now()}] --- fit transform begins")
    scaled_training = pd.DataFrame(scaler.fit_transform(df_train), columns=df_train.columns)

    # Transforming data with already fitted scaler
    print(f"[{dt.now()}] --- fit and transform finished for training data")
    scaled_testing = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns)

    # Saving scaler state
    print(f"[{dt.now()}] --- saving MinMaxScaler()")
    pickle.dump(scaler, open("helper\\scaler.pkl", "wb"))

    # Saving scaled data to csv files
    print(f"[{dt.now()}] --- Saving training and testing data")
    scaled_training.to_csv(train_scaled, index=False)
    scaled_testing.to_csv(test_scaled, index=False)

    print(f"[{dt.now()}] --- process finished")
    return 0


if __name__ == "__main__":
    main()

"""
[09:18:08] --- loading data
[09:18:43] --- fit transform begins
[09:18:47] --- fit and transform finished for training data
[09:18:47] --- saving MinMaxScaler()
[09:18:47] --- Saving training and testing data
[09:24:47] --- process finished
"""
