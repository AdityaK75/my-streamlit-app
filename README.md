# 🌫️ AI/ML-Based Air Pollution Dispersion Modeling for Chemical Plant Emission Monitoring

## Project Overview
This project develops an LSTM-based machine learning model to predict **PM2.5 pollution levels** near chemical plants using meteorological and environmental data.  
The goal is to enable **real-time emission control**, **worker safety monitoring**, and **regulatory compliance**.

- **Dataset:** Hourly air quality data (2010–2014) from Kaggle [LSTM-Multivariate Pollution Dataset](https://www.kaggle.com/datasets/rupakroy/lstm-datasets-multivariate-univariate)
- **Model:** Long Short-Term Memory (LSTM) neural network
- **Deployment:** Streamlit web app
- **Key Features:** Multivariate forecasting, anomaly detection, SHAP interpretability, scenario analysis

---

## 📂 Repository Structure
| File/Folder | Description |
|:---|:---|
| `app.py` | Streamlit application for pollution prediction |
| `best_model.h5` | Trained LSTM model |
| `scaler.pkl` | MinMaxScaler used for feature scaling |
| `requirements.txt` | Python libraries required |
| `README.md` | Project overview and instructions |

---

## 🚀 How to Run Locally

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/air-pollution-forecasting.git
    cd air-pollution-forecasting
    ```

2. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Launch the app:
    ```bash
    streamlit run app.py
    ```

4. Open the URL `http://localhost:8501/` in your browser!

---

## 🌎 Live Demo (Optional)
[Add your Streamlit Cloud URL here after deployment]

---

## 📈 Model Insights
- **RMSE:** 0.065
- **R² Score:** 0.91
- **Key Features:** Wind Speed, Temperature, Atmospheric Pressure
- **Scenario Analysis:** Showed critical pollution spikes under low wind and high pressure
- **Anomaly Detection:** Isolation Forest revealed ~1.2% extreme events

---

## 📚 References
- Kaggle Dataset: [LSTM Datasets - Multivariate & Univariate](https://www.kaggle.com/datasets/rupakroy/lstm-datasets-multivariate-univariate)
- Research Articles on Pollution Dispersion Forecasting
- Streamlit Documentation
- TensorFlow Keras Official Docs

---

## 👨‍🔬 Author
**Aditya Kanagalekar**  
B.Tech Chemical Engineering, IIT Guwahati  
Roll No: 220107005
