import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scale_data import convert_date_to_float
from keras.layers.experimental.preprocessing import TextVectorization


def iterate_through_directory(directory):
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            scale_and_write_to_file(directory, filename)
            return


def scale_and_write_to_file(directory, filename):
    print(f"starting to read in {filename}")
    f = os.path.join(directory, filename)
    df = pd.read_csv(f)
    df['Date'] = df['Date'].apply(lambda x: convert_date_to_float(x))
    layer = TextVectorization(output_mode='int', output_sequence_length=1, max_tokens=10000)
    print("adapting layer")
    layer.adapt(df["Ticker"])
    print("layer adoption finished")
    df["Ticker"] = layer(df["Ticker"])
    print("ticker replacement finished")
    print(df.head())
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    scaled_data_df = pd.DataFrame(scaled_data, columns=df.columns.values)
    print("Note: Close values were scaled by multiplying by {:.10f} and adding {:.6f}"
          .format(scaler.scale_[4], scaler.min_[4]))
    scaled_data_df.to_csv(os.path.join(directory.replace("data_sum", "data_sum_scaled"), filename), index=False)
    print(f"finished writing to {filename}")
