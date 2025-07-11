import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from transformers import MarianMTModel, MarianTokenizer

# Set Streamlit page config
st.set_page_config(page_title="AgriGuru Lite", layout="centered")
st.title("🌾 AgriGuru Lite – Smart Farming Assistant")

# -------- LANGUAGE TRANSLATION SETUP --------
language_map = {
    "English": "en", "Hindi": "hi", "Bengali": "bn", "Tamil": "ta",
    "Telugu": "te", "Marathi": "mr", "Gujarati": "gu", "Punjabi": "pa"
}

src_lang = "en"
target_lang_name = st.selectbox("🌐 Choose Language", list(language_map.keys()))
target_lang = language_map[target_lang_name]

# Translation function
@st.cache_resource
def load_translation_model(src='en', tgt='hi'):
    model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate_text(text, tgt):
    if tgt == "en":
        return text  # No translation
    try:
        tokenizer, model = load_translation_model('en', tgt)
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        translated = model.generate(**inputs)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    except:
        return text  # fallback

def _(text):
    return translate_text(text, target_lang)

# -------- WEATHER FORECAST --------
st.subheader(_("🌦️ 5-Day Weather Forecast"))
api_key = "0a16832edf4445ce698396f2fa890ddd"  # Replace with your OpenWeatherMap API Key

location = st.text_input(_("Enter your City/District (for weather)"))

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

# -------- RULE-BASED CROP RECOMMENDATION --------
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

# -------- ML-BASED CROP RECOMMENDATION --------
st.subheader(_("🤖 ML-Based Crop Recommendation (via CSV + Random Forest)"))

@st.cache_data
def load_crop_data():
    return pd.read_csv("Crop_recommendation.csv")

df = load_crop_data()

X = df.drop("label", axis=1)
y = df["label"]

model = RandomForestClassifier()
model.fit(X, y)

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
    st.success(f"🌱 { _('Predicted Crop') }: **{predicted_crop}** ({ _(season) } {_('season')})")


