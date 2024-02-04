import yfinance as yf

#msft = yf.Ticker("MSFT")
#msft_hist = msft.history(period="max")
import os
import pandas as pd
import numpy as np

DATA_PATH = "msft_data.json"
"""
data = pd.read_json(DATA_PATH)
# we only care about 10 years after that is from 2000
# therefore we take last 1 year of working data
print(data.head())

data["target"] = data["High"]
data["target"] = data["High"].shift(1)
data["feature"] = list(data.rolling(180))

print(data.head())
print(data.tail())
print(data.iloc[0]["feature"])

features = data.iloc[365]["feature"]
print("features:")
print(features)
featureArray = np.zeros((180, 7))
for i in range(features.shape[0]):
    featureArray[i][0] = features.iloc[i]["Open"]
    featureArray[i][1] = features.iloc[i]["High"]
    featureArray[i][2] = features.iloc[i]["Low"]
    featureArray[i][3] = features.iloc[i]["Close"]
    featureArray[i][4] = features.iloc[i]["Volume"]
    featureArray[i][5] = features.iloc[i]["Dividends"]
    featureArray[i][6] = features.iloc[i]["Stock Splits"]

print(featureArray)
data.to_json("processedPrices.json")

newpath = "processedPrices.json"
"""



"""
import torch
from torch.utils.data import Dataset
class CustomStockDataset(Dataset):
    def __init__(self, csv_file, discardLoc, transform=None):
        self._data = pd.read_json(csv_file)  # Make sure to import pandas
        self._transform = transform

        self._data["target"] = self._data["High"]
        self._data["target"] = self._data["High"].shift(1)
        self._data["feature"] = list(self._data.rolling(180))

        # throw away last bit of data

        print("Dataset shape:" + str(self._data.shape))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):  # idx is an integer

        # takes previous 6 months / 1 year / something and compiles it into a row Array
        # features: 2D Array of 180 rows / 7 cols
        # label: price of the next item

        sample = self._data.iloc[idx]
        # You may need to modify this part based on your specific CSV format
        _features = sample["feature"]

        featureArray = np.zeros((180, 7))
        for i in range(_features.shape[0]):
            featureArray[i][0] = _features.iloc[i]["Open"]
            featureArray[i][1] = _features.iloc[i]["High"]
            featureArray[i][2] = _features.iloc[i]["Low"]
            featureArray[i][3] = _features.iloc[i]["Close"]
            featureArray[i][4] = _features.iloc[i]["Volume"]
            featureArray[i][5] = _features.iloc[i]["Dividends"]
            featureArray[i][6] = _features.iloc[i]["Stock Splits"]
        _features = featureArray

        label = sample["target"]

        if self._transform:
            _features = self._transform(_features)

        return torch.from_numpy(_features), label


ds = CustomStockDataset(DATA_PATH, 360)
ft, tg = ds.__getitem__(360)
print(ft)
print(ft.shape)
print("target="+str(tg))
"""
import random
def rnd_train_test(k, source):
    _train = source
    _test = []

    for i in range(k):
        _test.append(_train.pop(random.randint(0, len(_train))))
    return _train, _test

print(rnd_train_test(5, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]))