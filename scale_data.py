import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from convert_tickers import search_dataframe
import joblib


def iterate_through_directory(directory):
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            scale_and_write_to_file(directory, filename)
            return


def scale_and_write_to_file(directory, filename):
    f = os.path.join(directory, filename)
    print(f'reading csv {f}')
    df = pd.read_csv(f)
    print(df.head())
    print('converting dates')
    df['Date'] = df['Date'].apply(
        lambda x: convert_date_to_float(x))
    print('dates converted')
    print(df.head())
    df_ticker = pd.read_csv(os.path.join("data", "symbols.csv"))
    print(df_ticker.head())
    print('applying tickers')
    df['Ticker'] = df['Ticker'].apply(
        lambda x:
            search_dataframe(df_ticker,'Symbol', x)
    )
    print('tickers applied')
    print(df.head())
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    print('Scaler fit-transformed')
    scaled_data_df = pd.DataFrame(scaled_data, columns=df.columns.values)
    print("Note: total_earnings values were scaled by multiplying by {:.10f} and adding {:.6f}"
          .format(scaler.scale_[4], scaler.min_[4]))
    scales = scaler.scale_
    mins = scaler.min_
    joblib.dump(scales, ("data_scaled_sum\\scale_" + filename).replace(".csv", ".gz"))
    joblib.dump(mins, ("data_scaled_sum\\scale_" + filename).replace(".csv", ".gz"))
    scaled_data_df.to_csv(os.path.join(directory.replace("data", "data_scaled"), filename), index=False)


def convert_date_to_float(date_str, reference_date_str='1970-01-01'):
    # Convert the date strings to datetime objects
    date = datetime.strptime(date_str, '%Y-%m-%d')
    reference_date = datetime.strptime(reference_date_str, '%Y-%m-%d')
    # Calculate the difference in days
    days_difference = (date - reference_date).days
    return float(days_difference)
