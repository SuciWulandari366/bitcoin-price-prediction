import streamlit as st
import pandas as pd
import lightgbm as lgb

model = lgb.Booster(model_file='model_bitcoin (5).txt')

st.title("Prediksi Harga Bitcoin 🪙")
st.write("Masukkan data hari ini untuk memprediksi harga Bitcoin besok")

open_price = st.number_input("Open Price")
high = st.number_input("High")
low = st.number_input("Low")
volume = st.number_input("Volume")
close_lag1 = st.number_input("Close Hari Sebelumnya")

if st.button("Prediksi"):
    data = pd.DataFrame([[open_price, high, low, volume, close_lag1]],
                        columns=['Open', 'High', 'Low', 'Volume', 'Close_lag1'])
    
    # Samakan urutan fitur
    data = data[model.feature_name()]
    
    pred = model.predict(data)
    
    st.success(f"Prediksi Harga Bitcoin Besok: {pred[0]:,.2f}")
