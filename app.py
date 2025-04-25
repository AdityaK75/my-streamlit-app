import streamlit as st
import numpy as np
import pickle
from keras.models import load_model

# Load model and scaler
model = load_model("/Users/adityakanagalekar/Downloads/lstm_model.h5", compile=False)
with open("/Users/adityakanagalekar/Downloads/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Air Pollution Forecasting App 🌫️")
st.write("Predict PM2.5 concentration based on meteorological data using an LSTM model.")

# Define inputs (11-hour window * 7 features)
st.subheader("Input Latest Hour's Data:")
dew = st.number_input("Dew Point (°C)", value=2.0)
temp = st.number_input("Temperature (°C)", value=14.0)
press = st.number_input("Pressure (hPa)", value=1016.0)
wnd_dir = st.selectbox("Wind Direction", options=["NE", "SE", "NW", "cv"])
wnd_spd = st.number_input("Wind Speed", value=5.3)
snow = st.number_input("Snow (hours)", value=0.0)
rain = st.number_input("Rain (hours)", value=0.0)

# Map wind_dir to numeric
wnd_map = {"NE": 0, "SE": 1, "NW": 2, "cv": 3}
wnd_dir = wnd_map[wnd_dir]

# For now, simulate 11 identical hours for demonstration
input_features = [[dew, temp, press, wnd_dir, wnd_spd, snow, rain]] * 11
input_scaled = scaler.transform(input_features)
X_input = np.array(input_scaled).reshape(1, 11, 7)

if st.button("Predict PM2.5"):
    prediction = model.predict(X_input)[0][0]
    st.success(f"Predicted PM2.5 Level: {round(prediction * 1000, 2)} µg/m³ (scaled back approximation)")
