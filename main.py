#!/usr/bin/env python
# -*- coding: utf-8 -*-
from scale_data import iterate_through_directory
from summarize_data import merge_all_data

def main():
    etfs = "data\\etfs"
    stocks = "data\\stocks"

    # iterate_through_directory(stocks)
    # iterate_through_directory(etfs)
    merge_all_data(stocks, etfs)
    return 0


if __name__ == "__main__":
    main()
