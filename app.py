import streamlit as st
import numpy as np
import pickle
from keras.models import load_model

# Load the trained LSTM model and the scaler
model = load_model("best_model.h5", compile=False)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit App Title
st.title("🌫️ AI-Based Air Pollution Forecasting System")
st.subheader("Predict PM2.5 Concentration near Chemical Plants")

st.markdown("""
This app uses an **LSTM neural network** trained on **meteorological data** to predict hourly **PM2.5 pollution levels**.
""")

# Input Section
st.header("Enter Current Meteorological Data:")

col1, col2 = st.columns(2)

with col1:
    dew = st.number_input("Dew Point (°C)", min_value=-50.0, max_value=50.0, value=2.0)
    temp = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, value=14.0)
    press = st.number_input("Atmospheric Pressure (hPa)", min_value=900.0, max_value=1100.0, value=1016.0)

with col2:
    wnd_dir = st.selectbox("Wind Direction", options=["NE", "SE", "NW", "cv"])
    wnd_spd = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=600.0, value=5.3)
    snow = st.number_input("Snowfall Hours", min_value=0.0, max_value=24.0, value=0.0)
    rain = st.number_input("Rainfall Hours", min_value=0.0, max_value=24.0, value=0.0)

# Map wind_dir to numerical value
wnd_map = {"NE": 0, "SE": 1, "NW": 2, "cv": 3}
wnd_dir = wnd_map[wnd_dir]

# Prepare the input feature for 11 past hours (replicate the latest data)
input_features = [[dew, temp, press, wnd_dir, wnd_spd, snow, rain]] * 11
input_scaled = scaler.transform(input_features)
X_input = np.array(input_scaled).reshape(1, 11, 7)

# Prediction Button
if st.button("🔮 Predict PM2.5 Concentration"):
    prediction_scaled = model.predict(X_input)[0][0]
    prediction = prediction_scaled * 1000  # Rescale to approximate original pollution units

    st.success(f"Predicted PM2.5 Level: **{round(prediction, 2)} µg/m³**")

    if prediction > 150:
        st.error("⚠️ Warning: Very High Pollution Level! Immediate Action Needed.")
    elif prediction > 75:
        st.warning("⚠️ Moderate Pollution Detected. Exercise caution.")
    else:
        st.success("✅ Pollution levels are within safe limits.")
    
    st.markdown("---")
    st.caption("Note: Prediction assumes steady meteorological conditions over the last 11 hours.")
