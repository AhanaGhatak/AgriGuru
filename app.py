import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from deep_translator import GoogleTranslator

# Language translation setup
languages = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Marathi": "mr",
    "Tamil": "ta"
}

# 🌐 Language selection
selected_lang = st.selectbox("🌐 Select Language", list(languages.keys()))
target_lang = languages[selected_lang]

# Translation function
def _(text):
    if target_lang == "en":
        return text
    try:
        return GoogleTranslator(source='en', target=target_lang).translate(text)
    except:
        return text
st.set_page_config(page_title="AgriGuru Lite", layout="centered")

st.title("🌾 AgriGuru Lite – Smart Farming Assistant")

# ---------------- LOAD PRODUCTION DATA ----------------
@st.cache_data
def load_production_data():
    return pd.read_csv("crop_production.csv")

try:
    prod_df = load_production_data()

    state_filter = st.selectbox("🌍 Select State", prod_df["State_Name"].dropna().unique())
    district_filter = st.selectbox("🏞️ Select District", prod_df[prod_df["State_Name"] == state_filter]["District_Name"].dropna().unique())
    season_filter = st.selectbox("🗓️ Select Season", prod_df["Season"].dropna().unique())

    st.markdown(f"### 📍 Selected Region: **{district_filter}, {state_filter}** | Season: **{season_filter}**")
except FileNotFoundError:
    st.warning("Please upload `crop_production.csv`.")

# ---------------- WEATHER FORECAST ----------------
st.subheader("🌦️ 5-Day Weather Forecast")
weather_api_key = "0a16832edf4445ce698396f2fa890ddd"

# Map district to valid city name (basic fallback)
district_to_city = {
    "MALDAH": "Malda",
    "BARDHAMAN": "Bardhaman",
    "NADIA": "Krishnanagar",
    "24 PARAGANAS NORTH": "Barasat",
    "24 PARAGANAS SOUTH": "Diamond Harbour",
    "HOWRAH": "Howrah",
    "KOLKATA": "Kolkata"
    # Add more as needed
}

def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={weather_api_key}&units=metric"
    res = requests.get(url)
    if res.status_code == 200:
        return res.json()['list'][:5]
    return None

if district_filter:
    city_query = district_to_city.get(district_filter.upper(), district_filter)
    forecast = get_weather(city_query)
    if forecast:
        for day in forecast:
            st.write(f"{day['dt_txt']} | 🌡️ {day['main']['temp']}°C | {day['weather'][0]['description']}")
    else:
        st.warning("⚠️ Weather unavailable. Try entering a nearby city manually.")

# ---------------- SOIL TYPE BUTTONS ----------------
st.subheader("🧱 Explore Suitable Crops by Soil Type")

soil_crop_map = {
    "Alluvial": ["Rice", "Sugarcane", "Wheat", "Jute"],
    "Black": ["Cotton", "Soybean", "Sorghum"],
    "Red": ["Millets", "Groundnut", "Potato"],
    "Laterite": ["Cashew", "Tea", "Tapioca"],
    "Sandy": ["Melons", "Pulses", "Groundnut"],
    "Clayey": ["Rice", "Wheat", "Lentil"],
    "Loamy": ["Maize", "Barley", "Sugarcane"]
}

soil_col1, soil_col2, soil_col3 = st.columns(3)
with soil_col1:
    if st.button("Alluvial"):
        st.info("🌾 Suitable Crops: " + ", ".join(soil_crop_map["Alluvial"]))
with soil_col2:
    if st.button("Black"):
        st.info("🌾 Suitable Crops: " + ", ".join(soil_crop_map["Black"]))
with soil_col3:
    if st.button("Red"):
        st.info("🌾 Suitable Crops: " + ", ".join(soil_crop_map["Red"]))

soil_col4, soil_col5, soil_col6 = st.columns(3)
with soil_col4:
    if st.button("Laterite"):
        st.info("🌾 Suitable Crops: " + ", ".join(soil_crop_map["Laterite"]))
with soil_col5:
    if st.button("Sandy"):
        st.info("🌾 Suitable Crops: " + ", ".join(soil_crop_map["Sandy"]))
with soil_col6:
    if st.button("Clayey"):
        st.info("🌾 Suitable Crops: " + ", ".join(soil_crop_map["Clayey"]))

if st.button("Loamy"):
    st.info("🌾 Suitable Crops: " + ", ".join(soil_crop_map["Loamy"]))

st.divider()

# ---------------- USER INPUT FOR ML ----------------
st.markdown("### 📥 Enter Soil and Climate Data (for ML Prediction)")
n = st.number_input("Nitrogen", min_value=0.0)
p = st.number_input("Phosphorous", min_value=0.0)
k = st.number_input("Potassium", min_value=0.0)
temp = st.number_input("Temparature (°C)", min_value=0.0)
humidity = st.number_input("Humidity (%)", min_value=0.0)
moisture = st.number_input("Moisture (%)", min_value=0.0)

# ---------------- ML MODEL: FILTERED BY DISTRICT CROPS ----------------
st.subheader("🌿 ML-Powered Crop Recommendation (Filtered by District)")

@st.cache_data
def load_soil_dataset():
    df = pd.read_csv("data_core.csv")
    le = LabelEncoder()
    df["soil_encoded"] = le.fit_transform(df["Soil Type"])
    features = ["Nitrogen", "Phosphorous", "Potassium", "Temparature", "Humidity", "Moisture", "soil_encoded"]
    X = df[features]
    y = df["Crop Type"]
    model = RandomForestClassifier()
    model.fit(X, y)
    return model, le, df

try:
    soil_model, soil_encoder, soil_df = load_soil_dataset()
    soil_input = st.selectbox("🧪 Select Soil Type for ML", soil_df["Soil Type"].unique())

    if st.button("Predict Best Crops in District"):
        encoded_soil = soil_encoder.transform([soil_input])[0]
        input_data = [[n, p, k, temp, humidity, moisture, encoded_soil]]

        # Get all unique crops from the district
        district_crops = prod_df[
            (prod_df["District_Name"] == district_filter) &
            (prod_df["State_Name"] == state_filter)
        ]["Crop"].dropna().unique()

        # Get ML predicted probabilities for all crops
        proba = soil_model.predict_proba(input_data)[0]
        labels = soil_model.classes_
        crop_scores = {label: prob for label, prob in zip(labels, proba)}

        # Sort only crops that are grown in the district
        recommended = [(crop, crop_scores[crop]) for crop in district_crops if crop in crop_scores]
        recommended = sorted(recommended, key=lambda x: x[1], reverse=True)[:5]

        if recommended:
            st.success("✅ Top Recommended Crops Grown in Your District:")
            for crop, score in recommended:
                st.write(f"🌱 **{crop}** – Confidence: {score:.2f}")
        else:
            st.warning("❌ No matching crops from prediction found in this district.")
except FileNotFoundError:
    st.warning("Please upload `data_core.csv`.")
