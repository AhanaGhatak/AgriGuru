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

st.set_page_config(page_title="AgriGuru Lite", layout="centered")

# 🌐 Language selection in sidebar
selected_lang = st.sidebar.selectbox("🌐 Select Language", list(languages.keys()))
target_lang = languages[selected_lang]

# Translation functions
def translate_to_en(text):
    if target_lang == "en":
        return text
    try:
        return GoogleTranslator(source=target_lang, target="en").translate(text)
    except:
        return text

def translate_from_en(text):
    if target_lang == "en":
        return text
    try:
        return GoogleTranslator(source="en", target=target_lang).translate(text)
    except:
        return text

def _(text):
    return translate_from_en(text)

st.title(_("\U0001F33E AgriGuru Lite – Smart Farming Assistant"))

# ---------------- LOAD PRODUCTION DATA ----------------
@st.cache_data
def load_production_data():
    return pd.read_csv("crop_production.csv")

try:
    prod_df = load_production_data()

    states = prod_df["State_Name"].dropna().unique()
    states_translated = [translate_from_en(s) for s in states]
    state_sel_trans = st.selectbox(_("🌍 Select State"), states_translated)
    state_filter = translate_to_en(state_sel_trans)

    districts = prod_df[prod_df["State_Name"] == state_filter]["District_Name"].dropna().unique()
    districts_translated = [translate_from_en(d) for d in districts]
    district_sel_trans = st.selectbox(_("🏜️ Select District"), districts_translated)
    district_filter = translate_to_en(district_sel_trans)

    season_options = ["Kharif", "Rabi", "Zaid"]
    season_options_translated = [translate_from_en(s) for s in season_options]
    season_trans = st.selectbox(_("🗓️ Select Season"), season_options_translated)
    season_filter = translate_to_en(season_trans)

    st.markdown(_(f"### 📍 Selected Region: **{district_filter}, {state_filter}** | Season: **{season_filter}**"))
except FileNotFoundError:
    st.warning(_("Please upload `crop_production.csv`."))

# ---------------- WEATHER FORECAST ----------------
st.subheader(_("\U0001F326️ 5-Day Weather Forecast"))
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

if 'district_filter' in locals():
    city_query = district_to_city.get(district_filter.upper(), district_filter)
    forecast = get_weather(city_query)
    if forecast:
        for day in forecast:
            st.write(f"{day['dt_txt']} | 🌡️ {day['main']['temp']} °C | {_(day['weather'][0]['description'])}")
    else:
        st.warning(_("\u26a0\ufe0f Weather unavailable. Try entering a nearby city manually."))

# ---------------- SOIL TYPE BUTTONS ----------------
st.subheader(_("\U0001F9F1 Explore Suitable Crops by Soil Type"))

soil_crop_map = {
    "Alluvial": ["Rice", "Sugarcane", "Wheat", "Jute"],
    "Black": ["Cotton", "Soybean", "Sorghum"],
    "Red": ["Millets", "Groundnut", "Potato"],
    "Laterite": ["Cashew", "Tea", "Tapioca"],
    "Sandy": ["Melons", "Pulses", "Groundnut"],
    "Clayey": ["Rice", "Wheat", "Lentil"],
    "Loamy": ["Maize", "Barley", "Sugarcane"]
}

soil_types = list(soil_crop_map.keys())
for i in range(0, len(soil_types), 3):
    cols = st.columns(3)
    for j in range(3):
        if i + j < len(soil_types):
            with cols[j]:
                soil_type = soil_types[i + j]
                if st.button(_(soil_type)):
                    translated_crops = [_(c) for c in soil_crop_map[soil_type]]
                    st.info("\U0001F33E " + _(f"Suitable Crops: {', '.join(translated_crops)}"))

st.divider()

# ---------------- USER INPUT FOR ML ----------------
st.markdown(_("### 📅 Enter Soil and Climate Data (for ML Prediction)"))
n = st.number_input(_("Nitrogen"), min_value=0.0, key="n_input")
p = st.number_input(_("Phosphorous"), min_value=0.0, key="p_input")
k = st.number_input(_("Potassium"), min_value=0.0, key="k_input")
temp = st.number_input(_("Temparature (°C)"), min_value=0.0, key="temp_input")
humidity = st.number_input(_("Humidity (%)"), min_value=0.0, key="humidity_input")
moisture = st.number_input(_("Moisture (%)"), min_value=0.0, key="moisture_input")

# ---------------- ML MODEL: FILTERED BY DISTRICT CROPS ----------------
st.subheader(_("\U0001F33F ML-Powered Crop Recommendation (Filtered by District)"))

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
    soil_opts = soil_df["Soil Type"].unique()
    soil_opts_translated = [translate_from_en(s) for s in soil_opts]
    soil_sel_trans = st.selectbox(_("\U0001F9EA Select Soil Type for ML"), soil_opts_translated)
    soil_input = translate_to_en(soil_sel_trans)

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
            st.success(_("\u2705 Top Recommended Crops Grown in Your District:"))
            for crop, score in recommended:
                st.write(f"\U0001F331 **{_(crop)}** – {_("Confidence")}: {score:.2f}")
        else:
            st.warning(_("\u274c No matching crops from prediction found in this district."))
except FileNotFoundError:
    st.warning(_("Please upload `data_core.csv`."))

