import pandas as pd

from main_train_test_split import see_all_files


def aggregate_files_and_apply_ticker(files):
    df = pd.DataFrame()
    df_ticker = pd.DataFrame()
    for i, file in enumerate(files):
        ticker = file[file.rfind("\\") + 1:].replace(".csv", "")
        is_stock = False
        if "stock" in file:
            is_stock = True
        # A dátum már át van alakítva
        df_part = pd.read_csv(file)
        # ha stock akkor 0 egyébként 1
        df_part['Is Stock'] = 0 if is_stock else 1
        df_part["Ticker"] = i
        df_ticker = df_ticker._append({'Ticker': ticker, 'Index': i}, ignore_index=True)
        df = pd.concat([df, df_part])
    print(f'loop ended shape: {df.shape}')
    return df, df_ticker


def print_aggregated_to_file(df, target):
    df.to_csv(target, index=False)


def main():
    train = see_all_files("data_train\\etfs") + see_all_files("data_train\\stocks")
    test = see_all_files("data_test\\etfs") + see_all_files("data_test\\stocks")

    test_df, test_tickers = aggregate_files_and_apply_ticker(test)
    print("tests read")
    train_df, train_tickers = aggregate_files_and_apply_ticker(train)
    print(f"train and test df-s are loaded: {train_df.shape}, {test_df.shape}")
    if train_tickers.equals(test_tickers):
        print("train and test tickers are identical, as expected")
    train_aggregated_filepath = "data_train\\aggregated\\train_aggr.csv"
    test_aggregated_filepath = "data_test\\aggregated\\test_aggr.csv"

    print_aggregated_to_file(train_df, train_aggregated_filepath)
    print_aggregated_to_file(test_df, test_aggregated_filepath)
    print_aggregated_to_file(train_tickers, "helper\\tickers.csv")
    print('csv-s printed')
    return 0


if __name__ == "__main__":
    main()
