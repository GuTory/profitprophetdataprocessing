#!/usr/bin/env python
# -*- coding: utf-8 -*-
from scale_data import iterate_through_directory


def main():
    etfs = "data\\etfs"
    stocks = "data\\stocks"

    iterate_through_directory(stocks)
    iterate_through_directory(etfs)
    return 0


if __name__ == "__main__":
    main()
