import datetime

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


grab("TSLA", "TSLA.json")
grab("^DJI", "DowJones.json")
grab("^IXIC", "NASDAQ.json")


DJIAfile = pd.read_json("DowJones.json")

NDXfile = pd.read_json("NASDAQ.json")

TargetFile = pd.read_json("TSLA.json")
TargetFile = TargetFile.astype("float64")

logTestDF = pd.read_json("TSLA.json").astype("float64")
logTestDF = logTestDF.drop(columns=["Dividends", "Stock Splits"])

"""
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
"""


# feature to be done: Use similar stocks to determine the stock in question

# find best buy/sell points, given previous N months of data (Default 6)

# Rank buy points: rate the amount of rising days from this point on to the next local max?

# 缠论解法?
# label : predict when the CURRENT trend of level n ends.

# a recursive approach with a recursive DS that is leveled.
# class

# also implement FeatureList class
class RecFeature:

    # begin - begin time of this feature
    # end - end time of this feature
    # level - on which level? if begin == end then level = 0
    # can create a recFeature from two recFeatures
    # StartDate = the lesser startDate of the 2 recFeature
    # endDate: = tne greater EndDate of the 2 recFeature

    # type; It can either be a
    # | "Node" | "Increasing Trend" | "Decreasing Trend"
    # with representations 0, 1, 2 respectively

    # momentum ; it is a number that encodes the momentum of a Node.
    # only Nodes have momentum. the longer a Node is the more momentum it has.

    # subnodelist : it is a list that only contains subnodes of level strictly
    # level -1.
    # For base case (a point) the level == 0.

    # it is a recursive DS. For every new node on a higher level,
    # it is always constructed from smaller nodes.

    def __init__(self, dateBegin, dateEnd, type: int, momentum, level: int, subnodes: list):
        self.dateBegin = dateBegin
        self.dateEnd = dateEnd
        self.type = type
        self.level = level
        self.momentum = momentum
        self.subnodes = subnodes

    def add_subnode(self, subnode):
        self.subnodes.append(subnode)

    def delete_subnode(self, subnode):
        self.subnodes.remove(subnode)

    # automatically determines type; if
    @staticmethod
    def MakeNew(self, subnodelist: list):
        if len(subnodelist) == 0:
            raise ValueError("ERROR: subnode list has length zero")

        hierarchy = subnodelist[0].level;
        for i in subnodelist:
            if i.level != hierarchy:
                raise TypeError("ERROR: subnode list has different leveled subnodes")

        dateBegin = subnodelist[0].dateBegin
        dateEnd = subnodelist[0].dateEnd
        for i in subnodelist:
            if i.dateBegin < dateBegin:
                dateBegin = i.dateBegin
            if i.dateEnd > dateEnd:
                dateEnd = i.dateEnd

        return RecFeature(dateBegin=dateBegin, dateEnd=dateEnd, level=hierarchy + 1, subnodes=subnodelist)


# 1. preprocessing
# Task: Implement the following algo
# algo: given a sequence:list[list] of dimension 2 * N of the form
# can also be dimension 3 x N? [high, low, dir]
# where dir == 1 -> growing -> that day open < close
# dir == 0 -> decreasing -> that day close < open
# [high1, low1],[high2, low2], [high3, low3]...
# input : [ [high, low] , [high, low] ] -> 2 dataframe columns

fileToAnalyze = "TSLA.json"

# identify type 1: growing sequence
# All arrays have high(n-1) < high(n) < high(n+1) and
# low(n-1) < low(n) < low(n+1); mark n as inc sequence

# identify type 2: decreasing sequence
# All arrays have high(n-1) > high(n) > high(n+1) and
# low(n-1) > low(n) > low(n+1); mark n as dec sequence

# identify type 3: peak
# Array have sequence high(n-1) < High(n) > high(n+1)
# And low(n-1) < low(n) > low(n+1); mark n as peak

# identify type 4: valley
# Array have sequence high(n-1) > High(n) < high(n+1)
# And low(n-1) > low(n) < low(n+1); mark n as valley

# use stack. put items sequentially in stack, identify while iterating.

# process all Inclusion type relations with a stack data structure.
# Inclusion:

# Type 1: Left inclusion
# If high(n-1) < high(n) and low(n-1) < low(n) then nth includes n-1th

# Type 2: Right inclusion
# If high(n-1) > high(n) and low(n-1) > low(n) then n-1th includes nth

# Inclusion processing algo:
# if n-2th is a growing sequence, we set the included value as:
# high of included = max(high(n-1), high(n))
# low of included = max(low(n-1), low(n))
# push onto stack

# else if n-2th is a decreasing sequence, we set included value as:
# high of included = min(high(n-1), high(n))
# low of included = min(low(n-1), low(n))
# push onto stack

# next entry, access stack as if processed data is original


inputdf = pd.read_json(fileToAnalyze)
high = "High"
low = "Low"


# assert a High and a Low column exists
# returns a list of vertices at index, and type
# completed
# this processes at level 0. Zeroth level of the architecture
# = vertices
def processLayer1(df: pd.DataFrame):
    # this forms a pd dataframe of shape
    # |High    | low   |
    # ------------------
    # |highval | lowval|
    # ...
    dfToAnalyze = pd.concat([df[high], df[low]], axis=1)
    height = dfToAnalyze.shape[0]

    extractedFeaturesHigh = []
    extractedFeaturesLow = []
    extractedFeaturesType = []
    extractedFeaturesIndex = []

    if height < 3:
        return []

    # inspects i-1, i, i+1.
    # Peak is labeled with 1, valley is labeled with -1
    for i in range(1, height - 1):
        # peak type:
        if (dfToAnalyze[high].iloc[i - 1] < dfToAnalyze[high].iloc[i]) and (
            dfToAnalyze[high].iloc[i] > dfToAnalyze[high].iloc[i + 1]) and (
            dfToAnalyze[low].iloc[i - 1] < dfToAnalyze[low].iloc[i]) and (
            dfToAnalyze[low].iloc[i] > dfToAnalyze[low].iloc[i + 1]):

            extractedFeaturesHigh.append(dfToAnalyze[high].iloc[i])
            extractedFeaturesLow.append(dfToAnalyze[low].iloc[i])
            extractedFeaturesType.append(1)
            extractedFeaturesIndex.append(df.index[i])

        # valley type:
        if (dfToAnalyze[high].iloc[i - 1] >
            dfToAnalyze[high].iloc[i] and dfToAnalyze[high].iloc[i] <
            dfToAnalyze[high].iloc[i + 1]) and (dfToAnalyze[low].iloc[i - 1] >
            dfToAnalyze[low].iloc[i] and dfToAnalyze[low].iloc[i] <
            dfToAnalyze[low].iloc[i + 1]):

            extractedFeaturesHigh.append(dfToAnalyze[high].iloc[i])
            extractedFeaturesLow.append(dfToAnalyze[low].iloc[i])
            extractedFeaturesType.append(-1)
            extractedFeaturesIndex.append(df.index[i])

    rawDF = {
        "VTHigh" : extractedFeaturesHigh,
        "VTLow"  : extractedFeaturesLow,
        "VTType" : extractedFeaturesType,
        "VTDate" : extractedFeaturesIndex
    }

    return pd.DataFrame(rawDF)

"""
dat = processLayer1(inputdf)
dataSegment = []
plt.plot(inputdf.index[len(inputdf)-200 : len(inputdf)] , inputdf[high][len(inputdf)-200 : len(inputdf)], "ro-")
plt.plot(dat["VTDate"][len(dat)-30:len(dat)], dat["VTHigh"][len(dat)-30:len(dat)], "bo--")
plt.plot(dat["VTDate"][len(dat)-30:len(dat)], dat["VTLow"][len(dat)-30:len(dat)], "go--")

plt.xlabel("Date")
plt.ylabel("High Price")
plt.title("TSLA vertices")
plt.xticks(rotation=45)
plt.show()
"""

# 2. identifying trends
# increasing trend -> a trend that starts at index n, and ends at index m if:
# n is at a valley, and m is at a peak, and n-m > 3, and high(m) > low(n).

# decreasing trend -> a trend that starts at index n, and ends at index m if:
# n is at a peak, and m is at a valley, and n-m > 3, and high(n) > low(m).

# sliding window algo / DP
# sliding window
# for i:
# if i is low:
#   iterate forward until a desired pos is found or end of list is reached
# if i is high:
#   do the same but for High conditions



# Trends are ALWAYS continuous. Perhaps not always differentiable.
# And always follow the rule that an increasing sequence is followed
# by a decreasing sequence.

# 3. increasing/ decreasing sequence list

# from a list seqs of k sequences, seqs[0], seqs[1], seqs[2] must have
# an overlap. recall that a sequence starts from an index and ends at an index.
# that is, given a list [h1, l1], [h2, l2], [h3, l3], ...
# intersect( [h1, l1] , [h2, l2], [h3, l3]) != null set.
# set mathematics / simple O(1) sol of comparing high lows.

# Identify: 1. Increasing sequence list
# increasing sequence lists fulfill the conditions above and
# starts and ends with an increasing sequence.

# Identify: 2. Decreasing sequence list
# increasing sequence lists fulfill the conditions above and
# starts and ends with an decreasing sequence.

# sequence lists have 2n+1 items where n in [0+].
# sequence lists are always continuous.

# 4. nodes, transitions


"""
# use normalized method - loses data. The data on Maximum is immediately lost
# Lower cond.

from sklearn.preprocessing import  normalize
from sklearn import preprocessing

scaler = sklearn.preprocessing.MinMaxScaler()

targetNormalized = TargetFile
targetNormalized["Open"] = scaler.fit_transform(targetNormalized[["Open"]])
targetNormalized["High"] = scaler.fit_transform(targetNormalized[["High"]])
targetNormalized["Low"] = scaler.fit_transform(targetNormalized[["Low"]])
targetNormalized["Close"] = scaler.fit_transform(targetNormalized[["Close"]])
targetNormalized["Volume"] = scaler.fit_transform(targetNormalized[["Volume"]])

targetNormalized = targetNormalized.drop(columns=["Dividends", "Stock Splits"])

targetNormalized.to_csv("TSLA_Normalized.csv")
targetNormalized.to_json("TSLA_Normalized.json")
print(targetNormalized.head())

print(la.cond(targetNormalized.to_numpy()))
"""
