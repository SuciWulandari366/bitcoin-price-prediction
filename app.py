import streamlit as st
import joblib
import pandas as pd

model = joblib.load('model_bitcoin.pkl')

st.title("Prediksi Harga Bitcoin 💰")

# Input
open_price = st.number_input("Open Price")
high = st.number_input("High")
low = st.number_input("Low")
volume = st.number_input("Volume")

if st.button("Prediksi"):
    data = pd.DataFrame([[open_price, high, low, volume]],
                        columns=['Open', 'High', 'Low', 'Volume'])
    
    pred = model.predict(data)
    
    st.success(f"Hasil Prediksi: {pred[0]:,.2f}")
