# This is a part of the code copied from Data Processing.
# I will be using both of them to compare the effects of adding the
# extracted trends vs raw stock data + MACD.

# for the raw data version, it will calculate 10 days ahead.


import datetime
from datetime import timedelta
import numpy as np
import numpy.linalg as la
import pandas as pd
import json
import os

import sklearn.preprocessing
import yfinance as yf
import matplotlib.pyplot as plt


def grab(ticker, path):
    dat = yf.Ticker(ticker)
    dat_hist = dat.history(period="max")

    # Save file to json in case we need it later.  This prevents us from having to re-download it every time.
    dat_hist.to_json(path)


#grab("TSLA", "TSLA.json")
#grab("^DJI", "DowJones.json")#
#grab("^IXIC", "NASDAQ.json")

# grab per interval:
# yf.download (tickers = , period = , interval = )
# to_json


DJIAfile = pd.read_json("DowJones.json")

NDXfile = pd.read_json("NASDAQ.json")

TargetFile = pd.read_json("TSLA.json")
TargetFile = TargetFile.astype("float64")

logTestDF = pd.read_json("TSLA.json").astype("float64")
logTestDF = logTestDF.drop(columns=["Dividends", "Stock Splits"])


print(logTestDF.shape)
logTestDF["NDX open"] = [0.00] * logTestDF.shape[0]
logTestDF["NDX high"] = [0.0] * logTestDF.shape[0]
logTestDF["NDX low"] = [0.0] * logTestDF.shape[0]
logTestDF["NDX close"] = [0.0] * logTestDF.shape[0]
logTestDF["NDX amount"] = [0.0] * logTestDF.shape[0]
logTestDF["DJIA open"] = [0.0] * logTestDF.shape[0]
logTestDF["DJIA high"] = [0.0] * logTestDF.shape[0]
logTestDF["DJIA low"] = [0.0] * logTestDF.shape[0]
logTestDF["DJIA close"] = [0.0] * logTestDF.shape[0]
logTestDF["DJIA amount"] = [0.0] * logTestDF.shape[0]

logTestDF = logTestDF.astype("float64")

_idx = 0
while _idx != logTestDF.shape[0]:
    curIndex = logTestDF.iloc[_idx].name

    # print(curIndex)

    # print(logTestDF.iloc[_idx]["NDX open"])
    # print(NDXfile.loc[curIndex]["Open"])
    try:
        logTestDF.loc[curIndex, "NDX open"] = NDXfile.loc[curIndex]["Open"]
        logTestDF.loc[curIndex, "NDX high"] = NDXfile.loc[curIndex]["High"]
        logTestDF.loc[curIndex, "NDX low"] = NDXfile.loc[curIndex]["Low"]
        logTestDF.loc[curIndex, "NDX close"] = NDXfile.loc[curIndex]["Close"]
        logTestDF.loc[curIndex, "NDX amount"] = NDXfile.loc[curIndex]["Volume"]
        logTestDF.loc[curIndex, "DJIA open"] = DJIAfile.loc[curIndex]["Open"]
        logTestDF.loc[curIndex, "DJIA close"] = DJIAfile.loc[curIndex]["Close"]
        logTestDF.loc[curIndex, "DJIA high"] = DJIAfile.loc[curIndex]["High"]
        logTestDF.loc[curIndex, "DJIA low"] = DJIAfile.loc[curIndex]["Low"]
        logTestDF.loc[curIndex, "DJIA amount"] = DJIAfile.loc[curIndex]["Volume"]

    except:
        print(f"one or more errors happened at location {_idx} with date {curIndex}")
    # print(logTestDF.loc[_idx, "NDX open"])

    _idx += 1
for i in range(logTestDF.shape[0]):
    for j in range(logTestDF.shape[1]):
        # no neg, +1 to prevent neg
        logTestDF.iloc[i, j] = (np.log10(logTestDF.iloc[i, j] + 1))

print(la.cond(logTestDF.to_numpy()))
print(logTestDF.shape)
logTestDF.to_csv("with_additional_dat.csv")
