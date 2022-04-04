
# load library
import streamlit as st
import pandas as pd
from statsmodels.tsa.holtwinters import  ExponentialSmoothing
import matplotlib.pyplot as plt

# print title of web app
st.title("Stock Market Analysis and Prediction")
dataset = ('Tata_Motors_stock','Wipro_stock','SBI_stock')
option = st.selectbox('Select dataset for prediction',dataset)
DATA_URL =("C:/Users/vinis/Stock_Market_Analysis/"+option+'.csv')

def load_data():
    data = pd.read_csv(DATA_URL)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data=load_data()

# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

# create checkbox
if st.checkbox('Show data'):
    st.subheader('Data from 2012 - 2022')
    st.write(data)


# dislay graph of open and close column
st.subheader('Graph of Close & Open:-')
st.line_chart(data[["Open","Close"]])



# display plot of volume column in datasets
#st.subheader('Graph of Volume:-')
#st.line_chart(data['Volume'])

#st.subheader('Closing Price vs Time chart')
#fig = plt.figure(figsize = (12,6))
#plt.plot(data.Close)
#st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = data.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(data.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA and 200MA')
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(data.Close,'b')
st.pyplot(fig)

x = int(len(data)*0.8)
train_data = data['Close'][0:x]
test_data = data['Close'][x:]

winter_model_mul = ExponentialSmoothing(train_data,seasonal='mul', trend='mul', seasonal_periods= 365).fit()
winter_predict_mul = winter_model_mul.predict(start = test_data.index[0], end = test_data.index[-1] )
 


st.subheader('Prediction vs Orginal')
fig2 = plt.figure(figsize = (12,6))
plt.plot(test_data,'b',Label = 'Orginal Price')
plt.plot(winter_predict_mul,'r',Label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

forecast_model = ExponentialSmoothing(data['Close'], seasonal='mul',trend ='mul', seasonal_periods=365).fit()
forecast_model.forecast(100)

st.subheader('Forecast for Next 100 Days')
fig3 = plt.figure(figsize=(12,6))
data['Close'].plot(label = 'Original Series')
forecast_model.forecast(100).plot(label='Forecast for Next 100 Days')
plt.legend()
st.pyplot(fig3)





