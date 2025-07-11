import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from deep_translator import GoogleTranslator

# ---------------- LANGUAGE TRANSLATION SETUP ----------------
languages = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Marathi": "mr",
    "Tamil": "ta"
}

selected_lang = st.sidebar.selectbox("🌐 Select Language", list(languages.keys()))
target_lang = languages[selected_lang]

# Translate helper function
def _(text):
    if target_lang == "en":
        return text
    try:
        return GoogleTranslator(source='en', target=target_lang).translate(text)
    except:
        return text

# ---------------- UI SETUP ----------------
st.set_page_config(page_title="AgriGuru Lite", layout="centered")
st.title(_("🌾 AgriGuru Lite – Smart Farming Assistant"))

# ---------------- LOAD PRODUCTION DATA ----------------
@st.cache_data
def load_production_data():
    return pd.read_csv("crop_production.csv")

try:
    prod_df = load_production_data()
    state_options = prod_df["State_Name"].dropna().unique()
    state_filter = st.selectbox(_("🌍 Select State"), state_options)

    district_options = prod_df[prod_df["State_Name"] == state_filter]["District_Name"].dropna().unique()
    district_filter = st.selectbox(_("🏜️ Select District"), district_options)

    season_options = prod_df["Season"].dropna().unique()
    season_filter = st.selectbox(_("🗓️ Select Season"), season_options)

    st.markdown(f"### 📍 {_('Selected Region')}: **{district_filter}, {state_filter}** | {_('Season')}: **{season_filter}**")
except FileNotFoundError:
    st.warning(_("Please upload `crop_production.csv`."))

# ---------------- WEATHER FORECAST ----------------
st.subheader(_("🌦️ 5-Day Weather Forecast"))
weather_api_key = "0a16832edf4445ce698396f2fa890ddd"

district_to_city = {
    "MALDAH": "Malda",
    "BARDHAMAN": "Bardhaman",
    "NADIA": "Krishnanagar",
    "24 PARAGANAS NORTH": "Barasat",
    "24 PARAGANAS SOUTH": "Diamond Harbour",
    "HOWRAH": "Howrah",
    "KOLKATA": "Kolkata"
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
            st.write(f"{day['dt_txt']} | 🌡️ {day['main']['temp']} °C | {_(day['weather'][0]['description'])}")
    else:
        st.warning(_("Weather unavailable. Try entering a nearby city manually."))

# ---------------- SOIL TYPE BUTTONS ----------------
st.subheader(_("🥱 Explore Suitable Crops by Soil Type"))

soil_crop_map = {
    "Alluvial": ["Rice", "Sugarcane", "Wheat", "Jute"],
    "Black": ["Cotton", "Soybean", "Sorghum"],
    "Red": ["Millets", "Groundnut", "Potato"],
    "Laterite": ["Cashew", "Tea", "Tapioca"],
    "Sandy": ["Melons", "Pulses", "Groundnut"],
    "Clayey": ["Rice", "Wheat", "Lentil"],
    "Loamy": ["Maize", "Barley", "Sugarcane"]
}

cols = st.columns(3)
soil_types = list(soil_crop_map.keys())
for i, soil in enumerate(soil_types):
    with cols[i % 3]:
        if st.button(_(soil)):
            crops = ", ".join([_(crop) for crop in soil_crop_map[soil]])
            st.info(_("🌾 Suitable Crops: ") + crops)

st.divider()

# ---------------- USER INPUT FOR ML ----------------
st.markdown("### 📥 " + _("Enter Soil and Climate Data (for ML Prediction)"))
n = st.number_input(_("Nitrogen"), min_value=0.0)
p = st.number_input(_("Phosphorous"), min_value=0.0)
k = st.number_input(_("Potassium"), min_value=0.0)
temp = st.number_input(_("Temparature (°C)"), min_value=0.0)
humidity = st.number_input(_("Humidity (%)"), min_value=0.0)
moisture = st.number_input(_("Moisture (%)"), min_value=0.0)

# ---------------- ML MODEL: FILTERED BY DISTRICT CROPS ----------------
st.subheader(_("🌿 ML-Powered Crop Recommendation (Filtered by District)"))

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
    soil_input = st.selectbox(_("🧪 Select Soil Type for ML"), soil_df["Soil Type"].unique())

    if st.button(_("Predict Best Crops in District")):
        encoded_soil = soil_encoder.transform([soil_input])[0]
        input_data = [[n, p, k, temp, humidity, moisture, encoded_soil]]

        district_crops = prod_df[
            (prod_df["District_Name"] == district_filter) &
            (prod_df["State_Name"] == state_filter)
        ]["Crop"].dropna().unique()

        proba = soil_model.predict_proba(input_data)[0]
        labels = soil_model.classes_
        crop_scores = {label: prob for label, prob in zip(labels, proba)}

        recommended = [(crop, crop_scores[crop]) for crop in district_crops if crop in crop_scores]
        recommended = sorted(recommended, key=lambda x: x[1], reverse=True)[:5]

        if recommended:
            st.success(_("✅ Top Recommended Crops Grown in Your District:"))
            for crop, score in recommended:
                st.write(f"🌱 **{_(crop)}** – {_('Confidence')}: {score:.2f}")
        else:
            st.warning(_("No matching crops from prediction found in this district."))
except FileNotFoundError:
    st.warning(_("Please upload `data_core.csv`."))
