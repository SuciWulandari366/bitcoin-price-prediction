import streamlit as st
import pandas as pd
import lightgbm as lgb
from datetime import datetime

# Load model
model = lgb.Booster(model_file='model_bitcoin (5).txt')

# UI
st.title("Prediksi Harga Bitcoin 🪙")
st.write("Masukkan data hari ini untuk memprediksi harga Bitcoin besok")

# Input user (dibuat simpel)
open_price = st.number_input("Open Price")
high = st.number_input("High")
low = st.number_input("Low")
volume = st.number_input("Volume")
close_lag1 = st.number_input("Close Hari Sebelumnya")
marketcap = st.number_input("Market Cap")

if st.button("Prediksi"):
    
    # Ambil waktu sekarang
    now = datetime.now()
    day_of_week = now.weekday()
    month = now.month
    year = now.year

    # Buat data lengkap (12 fitur)
    data = pd.DataFrame([{
        'Open': open_price,
        'High': high,
        'Low': low,
        'Volume': volume,

        'Close_lag1': close_lag1,
        'Close_lag2': close_lag1,   # disederhanakan
        'Close_lag3': close_lag1,

        'day_of_week': day_of_week,
        'month': month,
        'year': year,

        'MarketCap': marketcap,
        'MarketCap_lag1': marketcap
    }])

    try:
        # Samakan urutan fitur sesuai model
        data = data[model.feature_name()]

        # Prediksi
        pred = model.predict(data)

        # Output
        st.success(f"Prediksi Harga Bitcoin Besok: {pred[0]:,.2f}")

    except Exception as e:
        st.error("Terjadi error saat prediksi 😢")
        st.write(e)
