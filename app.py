import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import  ExponentialSmoothing
# %matplotlib inline
import seaborn as sns
sns.set()

import streamlit as st
import pickle
import warnings
warnings.filterwarnings('ignore')

# loading the Model
forecast_model = pickle.load(open('forecast_holt_winter_mul.pkl','rb'))

# print title of web app
st.title("Stock Market Analysis and Prediction")

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')

# Reading the Stock Data
data=pd.read_csv("Tata_Motors_stock.csv")

data['Date'] = pd.to_datetime(data['Date'])
data['Symbol'] = data['Symbol'].astype('category')
data['Series'] = data['Series'].astype('category')

# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

# create checkbox
if st.checkbox('Show data'):
    st.subheader('Data from 2012 - 2022')
    st.write(data)

# dislay graph of open and close column
st.subheader('Graph of Close & Open:-')
fig = plt.figure(figsize = (20,7))
st.line_chart(data[["Open","Close"]])
st.pyplot(fig)

# display plot of volume column in datasets
st.subheader('Graph of Volume:-')
fig = plt.figure(figsize = (20,7))
st.line_chart(data['Volume'])
st.pyplot(fig)

# displaying time vs Closing Price
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize = (20,7))
st.line_chart(data['Close'])
st.pyplot(fig)

# Decomposition of Time Series Data
#st.subheader('Decomposition of Time Series Data')
# time_components_365 = seasonal_decompose(data['Close'], period= 365, model = 'additive')
# st.line_chart(time_components_365)

st.subheader('Closing Price vs Time chart with 365MA')
fig = plt.figure(figsize = (20,7))
plt.plot(data['Close'])
data['Close'].rolling(365).mean().plot(label= 'MA 365' )
st.pyplot(fig)

st.subheader('Forecast for Next 100 Days')
fig = plt.figure(figsize=(20,8))
data['Close'].plot(label = 'Original Series')
forecast_model.forecast(100).plot(label='Forecast for Next 100 Days')
st.pyplot(fig)

st.subheader('Original Vs Predicted')
fig = plt.figure(figsize=(20,8))
data['Close'].plot(label = 'Original Share Price')
forecast_model.predict(start= data['Close'].index[0],end = data['Close'].index[-1]).plot(label='Predicted Share price')
st.pyplot(fig)