from pyexpat import XML_PARAM_ENTITY_PARSING_ALWAYS
from matplotlib.pyplot import close
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import plotly
import plotly.graph_objects as go
from pandas_datareader import data as wb
import requests
import hvplot.pandas
import base64
from pandas.tseries.offsets import DateOffset

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import classification_report, plot_precision_recall_curve



st.title("""Predicting Price""")
st.markdown("Our project is to select stocks and sectors and predict future prices by using SVM model and LSTM model")



#function calling local css sheet
def local_css(file_name):
    with open(file_name) as f:
        st.sidebar.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

#local css sheet
local_css("style.css")

#ticker search feature in sidebar
st.sidebar.subheader("""Ticker""")
selected_stock = st.sidebar.text_input("Enter the ticker you want to explore...", "GOOG")
button_clicked = st.sidebar.button("GO")
if button_clicked == "GO":
    main()

#main function
def main():
    st.subheader("""Daily adjusted closing price for """ + selected_stock)
    #get data on searched ticker
    stock_data = yf.Ticker(selected_stock)
    #get historical data for searched ticker
    stock_df = stock_data.history(period='1d', start='2018-01-01', end=None)
    #print line chart with daily closing prices for searched ticker
    col1, col2 = st.columns(2)
    col1.line_chart(stock_df.Close)
    col2.dataframe(stock_df.Close)

if __name__ == "__main__":
    main()


#creating training and test splits
@st.cache(persist = True)
def split(df):
    y = stock_df['signal'].copy()
    X = df.drop(columns = ["signal"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)
    
    return X_train, X_test, y_train, y_test

def plot_metrics (metrics_list):
    if "Precision-Recall Curve" in metrics_list:
        st.subheader("Precision-Recall Curve")
        plot_precision_recall_curve(model, X_test, y_test)
        st.pyplot()

st.sidebar.subheader("""Choose Classifier""")
classifier = st.sidebar.selectbox("Classifier", ("SVM"))

#SP 500

st.title ('S&P 500 App')
st.markdown("""
This app retrieves the list of the **S&P 500** from Wikipedia
* **Data Source:** [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)
""")

st.sidebar.header('User Input Features')

#Web scraping of S&P 500 data
@st.cache
def load_data():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = pd.read_html(url, header = 0)
    df = html[0]
    return df

df = load_data()
sector = df.groupby('GICS Sector')

# Sidebar - Sector selection
sorted_sector_unique = sorted( df['GICS Sector'].unique() )
selected_sector = st.sidebar.multiselect('Sector', sorted_sector_unique, sorted_sector_unique)

# Filtering data
df_selected_sector = df[ (df['GICS Sector'].isin(selected_sector)) ]

st.header('Display Companies in Selected Sector')
st.write('Data Dimension: ' + str(df_selected_sector.shape[0]) + ' rows and ' + str(df_selected_sector.shape[1]) + ' columns.')
st.dataframe(df_selected_sector)

# Download S&P500 data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="SP500.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df_selected_sector), unsafe_allow_html=True)

# https://pypi.org/project/yfinance/

data = yf.download(
        tickers = list(df_selected_sector[:10].Symbol),
        period = "ytd",
        interval = "1d",
        group_by = 'ticker',
        auto_adjust = True,
        prepost = True,
        threads = True,
        proxy = None
    )


# Plot Closing Price of Query Symbol
def price_plot(symbol):
  df = pd.DataFrame(data[symbol].Close)
  df['Date'] = df.index
  plt.plot(df.Date, df.Close, color='skyblue', alpha=0.8)
  plt.xticks(rotation=90)
  plt.title(symbol, fontweight='bold')
  plt.xlabel('Date', fontweight='bold')
  plt.ylabel('Closing Price', fontweight='bold')
  st.set_option('deprecation.showPyplotGlobalUse', False)
  return st.pyplot()

num_company = st.sidebar.slider('Number of Companies', 1, 10)

if st.button('Show Plots'):
    st.header('Stock Closing Price')
    for i in list(df_selected_sector.Symbol)[:num_company]:
        price_plot(i)