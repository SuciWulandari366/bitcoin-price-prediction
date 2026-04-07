import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('model_bitcoin.pkl')

# Judul
st.title("Prediksi Harga Bitcoin 💰")

st.write("Masukkan data untuk prediksi")

# Input
open_price = st.number_input("Open Price", value=0.0)
volume = st.number_input("Volume", value=0.0)

# Tombol prediksi
if st.button("Prediksi"):
    data = np.array([[open_price, volume]])
    pred = model.predict(data)
    
    st.success(f"Prediksi Harga Bitcoin: {pred[0]:,.2f}")