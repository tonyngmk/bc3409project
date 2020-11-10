#!/usr/bin/env python
# -*- coding: utf-8 -*- 


import yfinance as yf
import pandas as pd
import numpy as np

def main():
    stocks = yf.Tickers(['FB','AMZN','NFLX','GOOGL','AAPL'])
    time_period = 'max'#can change this
    df = stocks.history(period = time_period)
    df = df['Close']
    df = df.reset_index()
    df = df[df.notna().all(axis = 1)]
    df.to_csv("prices.csv", index=False)
    
    stocks = ['FB','AMZN','NFLX','GOOGL','AAPL']
    masterlist = []
    for each in stocks:
        x = yf.Ticker(each)
        y = x.recommendations
        y['stock_symbol'] = each
        y = y.reset_index()
        masterlist.append(y)
    dfRecommendations = pd.concat(masterlist,axis= 0)
    dfRecommendations["Date"] = pd.to_datetime(dfRecommendations["Date"]) # convert string to datetime format
    dfRecommendations.to_csv('recommendations.csv',index = False)
            
if __name__ == '__main__':
    main()