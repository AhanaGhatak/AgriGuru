import streamlit as st 
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from deep_translator import GoogleTranslator

# Language translation setup
languages = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Marathi": "mr",
    "Tamil": "ta"
}

st.set_page_config(page_title="AgriGuru Lite", layout="centered")

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

st.title(_("🌾 AgriGuru Lite – Smart Farming Assistant"))

# ---------------- WEATHER FORECAST ----------------
st.subheader(_("🌦️ 5-Day Weather Forecast"))
api_key = "0a16832edf4445ce698396f2fa890ddd"

if location:
    # Translate location to English for API call
    try:
        location_en = GoogleTranslator(source=target_lang, target='en').translate(location)
    except:
        location_en = location  # fallback if translation fails

    forecast = get_weather(location_en)

def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"
    res = requests.get(url)
    if res.status_code == 200:
        return res.json()['list'][:5]
    return None

if location:
    forecast = get_weather(location)
    if forecast:
        for day in forecast:
            st.write(f"{day['dt_txt']} | 🌡️ {day['main']['temp']}°C | {_(day['weather'][0]['description'])}")
    else:
        st.warning(_("Couldn't fetch weather. Please check the city name."))

# ---------------- RULE-BASED CROP RECOMMENDATION ----------------
st.subheader(_("🧠 Rule-Based Crop Recommendation"))

season = st.selectbox(_("Select the Crop Season"), ["Kharif", "Rabi", "Zaid"])
soil = st.selectbox(_("Select Soil Type"), ["Alluvial", "Black", "Red", "Laterite", "Sandy", "Clayey"])

def recommend_crops(season, soil):
    if season == "Kharif" and soil == "Alluvial":
        return ["Paddy", "Maize", "Jute"]
    elif season == "Rabi" and soil == "Black":
        return ["Wheat", "Barley", "Gram"]
    elif season == "Zaid":
        return ["Watermelon", "Cucumber", "Bitter Gourd"]
    else:
        return ["Millets", "Pulses", "Sunflower"]

if season and soil:
    rule_based = recommend_crops(season, soil)
    st.success(_("Recommended Crops: ") + ", ".join(rule_based))

# ---------------- ML-BASED CROP RECOMMENDATION ----------------
st.subheader(_("🤖 ML-Based Crop Recommendation (via CSV + Random Forest)"))

@st.cache_data
def load_crop_data():
    return pd.read_csv("Crop_recommendation.csv")

df = load_crop_data()

X = df.drop("label", axis=1)
y = df["label"]

model = RandomForestClassifier()
model.fit(X, y)

# Crop-to-Season Mapping
crop_seasons = {
    "rice": "Kharif", "maize": "Kharif", "jute": "Kharif", "cotton": "Kharif",
    "kidneybeans": "Kharif", "pigeonpeas": "Kharif", "blackgram": "Kharif", 
    "mothbeans": "Kharif", "mungbean": "Kharif",

    "wheat": "Rabi", "gram": "Rabi", "lentil": "Rabi", "chickpea": "Rabi",
    "grapes": "Rabi", "apple": "Rabi", "orange": "Rabi", "pomegranate": "Rabi",

    "watermelon": "Zaid", "muskmelon": "Zaid", "cucumber": "Zaid",

    "banana": "All Season", "mango": "All Season", "papaya": "All Season",
    "coconut": "All Season", "coffee": "All Season"
}

st.markdown(_("*Enter Soil and Climate Data for ML Prediction*"))
n = st.number_input(_("Nitrogen (N)"), min_value=0.0)
p = st.number_input(_("Phosphorus (P)"), min_value=0.0)
k = st.number_input(_("Potassium (K)"), min_value=0.0)
temp = st.number_input(_("Temperature (°C)"), min_value=0.0)
humidity = st.number_input(_("Humidity (%)"), min_value=0.0)
ph = st.number_input(_("Soil pH"), min_value=0.0)
rainfall = st.number_input(_("Rainfall (mm)"), min_value=0.0)

if st.button(_("Predict Best Crop")):
    input_data = [[n, p, k, temp, humidity, ph, rainfall]]
    prediction = model.predict(input_data)
    predicted_crop = prediction[0]
    season = crop_seasons.get(predicted_crop, "Unknown")
    st.success(f"🌱 { _('Predicted Crop') }: *{predicted_crop}* ({ _(season) } {_('season')})")
