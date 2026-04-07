import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load('model_bitcoin.pkl')

st.title("Prediksi Harga Bitcoin 💰")

open_price = st.number_input("Open Price", value=0.0)
volume = st.number_input("Volume", value=0.0)

if st.button("Prediksi"):
    data = pd.DataFrame([[open_price, volume]], columns=['Open', 'Volume'])
    
    pred = model.predict(data)
    
    st.success(f"Hasil Prediksi: {pred[0]:,.2f}")
