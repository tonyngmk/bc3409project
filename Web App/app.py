import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import plotly.figure_factory as ff
import yfinance as yf

@st.cache(persist = True)
def load_data():
    stocks = yf.Tickers(['FB','AMZN','NFLX','GOOGL','AAPL'])
    time_period = 'max'#can change this
    df = stocks.history(period = time_period)
    df = df['Close']
    df = df.reset_index()
    df = df[df.notna().all(axis = 1)]
    # df.to_csv("prices.csv", index=False)
    # df = pd.read_csv("prices.csv")
    df["Date"] = pd.to_datetime(df["Date"]) # convert string to datetime format
    
    # stocks = ['FB','AMZN','NFLX','GOOGL','AAPL']
    # masterlist = []
    # for each in stocks:
        # x = yf.Ticker(each)
        # y = x.recommendations
        # y['stock_symbol'] = each
        # y = y.reset_index()
        # masterlist.append(y)
    # dfRecommendations = pd.concat(masterlist,axis= 0)
    # dfRecommendations["Date"] = pd.to_datetime(dfRecommendations["Date"]) # convert string to datetime format
    # dfRecommendations.to_csv('recommendations.csv',index = False)
    dfRecommendations = pd.read_csv("recommendations.csv")# Read from cache to reduce time taken
    dfRecommendations["Date"] = pd.to_datetime(dfRecommendations["Date"]) # convert string to datetime format
    dfRecommendations["Date"] = dfRecommendations["Date"].dt.date
    return df, dfRecommendations

st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/NTU%20Logo.png', width = 750)
st.write('''<h1 align=center><font color='Blue'>BC3409</font> - 
<font color='red'>AI in Accounting & Finance</font>''', unsafe_allow_html=True)

st.write("<h2 align=center>FAANG Web App</h2>", unsafe_allow_html=True)
'''This web app provides a platform to visualize live data of all 5 FAANG stocks from Yahoo Finance.'''

df, dfRecommendations = load_data()
faangDict = {"Facebook": "FB", "Amazon":"AMZN", "Apple":"AAPL", "Netflix":"NFLX", "Google":"GOOGL"}
invFaangDict = {"FB": "Facebook", "AMZN":"Amazon", "AAPL":"Apple", "NFLX":"Netflix", "GOOGL":"Google"}
faangTuple = ("Facebook", "Amazon", "Apple", "Netflix", "Google")
def filterRec(df, col):
    return df["stock_symbol"] == col


stock = st.radio("Select stock:", faangTuple)
# if stock == "Facebook":
"### Price Plot"
st.area_chart(df.set_index("Date")[faangDict[stock]])
"### Percentage Change"
dfPctChg = {"day{}".format(i):pd.concat([df["Date"], df[faangDict[stock]].pct_change(periods=i)], axis=1) for i in [1, 5, 30, 90, 180]}
latestDate = pd.DataFrame({"{} Day".format(i): dfPctChg["day{}".format(i)].iloc[-1][faangDict[stock]]*100 for i in [1, 5, 30, 90, 180]}, 
                 index = ["Latest Date: {}".format(dfPctChg["day1"].iloc[-1]["Date"].strftime("%d/%m/%Y"))])
latestDate
"#### 1-Day % Change"
day1 = dfPctChg["day1"].set_index("Date").dropna()
day1.columns = ["1-Day Percentage Change"]
st.line_chart(day1)

"#### 5-Day % Change"
day5 = dfPctChg["day5"].set_index("Date").dropna()
day5.columns = ["5-Day Percentage Change"]
st.line_chart(day5)

"#### 30-Day % Change"
day30 = dfPctChg["day30"].set_index("Date").dropna()
day30.columns = ["30-Day Percentage Change"]
st.line_chart(day30)

"#### 90-Day % Change"
day90 = dfPctChg["day90"].set_index("Date").dropna()
day90.columns = ["90-Day Percentage Change"]
st.line_chart(day90)

"#### 180-Day % Change"
day180 = dfPctChg["day180"].set_index("Date").dropna()
day180.columns = ["180-Day Percentage Change"]
st.line_chart(day180)

if st.checkbox("Wrangle Multiple Stocks"):
    "### Inspect Dataset"
# nrows = st.slider("Select number of rows:", df.iloc[0]["Date"].to_pydatetime(), df.iloc[-1]["Date"].to_pydatetime())
    start, end = st.select_slider("Select date range:", options = df["Date"].dt.strftime("%d/%m/%Y").values.tolist(), value = ("02/01/2015", "01/06/2020"))
    start = datetime.datetime.strptime(start, "%d/%m/%Y")
    end = datetime.datetime.strptime(end, "%d/%m/%Y")
    stockInspect = st.multiselect("Select stocks:", faangTuple, faangTuple)
    dfPriceFiltered = df.set_index("Date").loc[start:end, [faangDict[i] for i in stockInspect]]
    "#### Price Dataset"
    dfPriceFiltered
    "#### Recommendations Dataset"
    dfRecFiltered = pd.concat([dfRecommendations[dfRecommendations["stock_symbol"] == faangDict[i]] for i in stockInspect])
    dfRecFiltered = dfRecFiltered.set_index("Date")
    dfRecFiltered
    "#### Price Plot"
    st.line_chart(dfPriceFiltered)
    "#### Distribution Plot"
    fig = ff.create_distplot([dfPriceFiltered[i].values.tolist() for i in dfPriceFiltered], stockInspect)
    st.plotly_chart(fig, use_container_width=True)
    # st.bar_chart(dfRecFiltered["To Grade"].value_counts(ascending=True))

    

hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

myWatermark = """
            <style>
            footer:after {
            content:'Tony Ng'; 
            visibility: visible;
            display: block;
            position: relative;
            #background-color: red;
            padding: 5px;
            top: 2px;
            }
            </style>
            """
st.markdown(myWatermark, unsafe_allow_html=True)