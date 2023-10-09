#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from scale_data import iterate_through_directory
from convert_tickers import search_ticker, load_data, save_df


def main():
    etfs = "data\\etfs"
    stocks = "data\\stocks"

    # iterate_through_directory(stocks)
    # iterate_through_directory(etfs)

    # Merge data
    # merge_all_data(stocks, etfs)

    iterate_through_directory("data_sum\\")
    f = os.path.join("data\\", 'symbols_valid_meta.csv')
    index = search_ticker('A')
    print(index)
    return 0


if __name__ == "__main__":
    main()
