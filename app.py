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

st.set_page_config(page_title="AgriGuru Lite", layout="wide")
st.markdown("""
    <style>
    .stButton button { width: 100%; border-radius: 10px; font-weight: bold; }
    .stSelectbox > div { border-radius: 10px; }
    .block-container { padding-top: 2rem; }
    .css-18e3th9 { background: linear-gradient(to right, #dce35b, #45b649); padding: 1rem; border-radius: 12px; }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #2f4f4f; }
    </style>
""", unsafe_allow_html=True)

# Sidebar: Language selection
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/606/606807.png", width=80)
st.sidebar.title("🌐 Language Settings")
selected_lang = st.sidebar.selectbox("Select Language", list(languages.keys()))
target_lang = languages[selected_lang]

# Translation with caching
translator_cache = {}
def _(text):
    if target_lang == "en":
        return text
    if (text, target_lang) in translator_cache:
        return translator_cache[(text, target_lang)]
    try:
        translated = GoogleTranslator(source='en', target=target_lang).translate(text)
        translator_cache[(text, target_lang)] = translated
        return translated
    except:
        return text

st.markdown(f"<h1 style='text-align: center; color: #006400;'>{_('🌾 AgriGuru Lite – Smart Farming Assistant')}</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 2px solid #228B22;'>", unsafe_allow_html=True)

@st.cache_data
def load_production_data():
    return pd.read_csv("crop_production.csv")

# Hardcoded most common crop by state
most_common_crop = {
    'Andhra Pradesh': 'Rice', 'Arunachal Pradesh': 'Maize', 'Assam': 'Rice', 'Bihar': 'Wheat',
    'Chhattisgarh': 'Rice', 'Goa': 'Coconut', 'Gujarat': 'Cotton', 'Haryana': 'Wheat',
    'Himachal Pradesh': 'Apple', 'Jharkhand': 'Rice', 'Karnataka': 'Ragi', 'Kerala': 'Coconut',
    'Madhya Pradesh': 'Wheat', 'Maharashtra': 'Sugarcane', 'Manipur': 'Rice', 'Meghalaya': 'Potato',
    'Mizoram': 'Rice', 'Nagaland': 'Maize', 'Odisha': 'Rice', 'Punjab': 'Wheat', 'Rajasthan': 'Pearl Millet',
    'Sikkim': 'Maize', 'Tamil Nadu': 'Rice', 'Telangana': 'Maize', 'Tripura': 'Rice',
    'Uttar Pradesh': 'Wheat', 'Uttarakhand': 'Wheat', 'West Bengal': 'Rice', 'Delhi': 'Wheat'
}

try:
    prod_df = load_production_data()
    col1, col2, col3 = st.columns(3)
    with col1:
        states = sorted(prod_df["State_Name"].dropna().unique())
        state_display = [_(s) for s in states]
        selected_state_display = st.selectbox(_("🌍 Select State"), state_display)
        selected_state = states[state_display.index(selected_state_display)]

    with col2:
        districts = sorted(prod_df[prod_df["State_Name"] == selected_state]["District_Name"].dropna().unique())
        district_display = [_(d) for d in districts]
        selected_district_display = st.selectbox(_("🏜 Select District"), district_display)
        selected_district = districts[district_display.index(selected_district_display)]

    with col3:
        seasons = sorted(prod_df["Season"].dropna().unique())
        season_display = [_(s) for s in seasons]
        selected_season_display = st.selectbox(_("🗓 Select Season"), season_display)
        selected_season = seasons[season_display.index(selected_season_display)]

    st.markdown(f"<h4 style='color:#2E8B57;'>📍 {('Selected Region')}: <b>{selected_district}, {selected_state}</b> | {('Season')}: <b>{selected_season}</b></h4>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning(_("⚠ Please upload crop_production.csv."))

st.markdown("### ⛅ " + _("Weather Forecast"))
weather_api_key = "0a16832edf4445ce698396f2fa890ddd"

def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={weather_api_key}&units=metric"
    res = requests.get(url)
    if res.status_code == 200:
        return res.json()['list'][:5]
    return None

if 'selected_district' in locals():
    try:
        district_en = GoogleTranslator(source=target_lang, target='en').translate(selected_district)
    except:
        district_en = selected_district

    forecast = get_weather(district_en)
    if forecast:
        for day in forecast:
            st.info(f"🗕 {day['dt_txt']} | 🌡 {day['main']['temp']} °C | ☁ {_(day['weather'][0]['description'])}")
    else:
        st.warning(_("⚠ Weather unavailable. Try a nearby city."))

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown(f"### 📊 { _('Enter Soil and Climate Data (for ML Prediction)') }")
col1, col2, col3 = st.columns(3)
with col1:
    n = st.number_input(_("Nitrogen"), min_value=0.0, key="n")
    p = st.number_input(_("Phosphorous"), min_value=0.0, key="p")
with col2:
    k = st.number_input(_("Potassium"), min_value=0.0, key="k")
    temp = st.number_input(_("Temparature (°C)"), min_value=0.0, key="temp")
with col3:
    humidity = st.number_input(_("Humidity (%)"), min_value=0.0, key="humidity")
    moisture = st.number_input(_("Moisture (%)"), min_value=0.0, key="moisture")

@st.cache_data
def load_soil_dataset():
    df = pd.read_csv("data_core.csv")
    le = LabelEncoder()
    df["soil_encoded"] = le.fit_transform(df["Soil Type"])
    features = ["Nitrogen", "Phosphorous", "Potassium", "Temparature", "Humidity", "Moisture", "soil_encoded"]
    X = df[features]
    y = df["Crop Type"]
    model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
    model.fit(X, y)
    return model, le, df

try:
    soil_model, soil_encoder, soil_df = load_soil_dataset()
    soil_display = [_(s) for s in soil_df["Soil Type"].unique()]
    selected_soil_display = st.selectbox(_("🧪 Select Soil Type for ML"), soil_display)
    selected_soil = soil_df["Soil Type"].unique()[soil_display.index(selected_soil_display)]

    budget = st.number_input(_("💰 Enter Your Budget (INR/tonne)"), min_value=0.0)

    if st.button(_("🌱 Predict Best Crops in District")):
        encoded_soil = soil_encoder.transform([selected_soil])[0]
        input_data = [[n, p, k, temp, humidity, moisture, encoded_soil]]

        district_crops = prod_df[
            (prod_df["District_Name"] == selected_district) &
            (prod_df["State_Name"] == selected_state)
        ]["Crop"].dropna().unique()

        proba = soil_model.predict_proba(input_data)[0]
        labels = soil_model.classes_
        crop_scores = {label: prob for label, prob in zip(labels, proba)}

        price_df = soil_df[["Crop Type", "Production (tonnes)"]].dropna()
        price_df["Clean Price"] = price_df["Production (tonnes)"].replace({
            r',': '', r'[^0-9.]': '', r'-+': ''
        }, regex=True).astype(str)
        price_df["Clean Price"] = pd.to_numeric(price_df["Clean Price"], errors='coerce')

        price_map = price_df.dropna().drop_duplicates("Crop Type").set_index("Crop Type")["Clean Price"].to_dict()

        recommended = [(crop, crop_scores[crop], price_map[crop])
                       for crop in district_crops
                       if crop in crop_scores and crop in price_map and price_map[crop] <= budget]

        # Add most common crop even if over budget
        if selected_state in most_common_crop:
            common_crop = most_common_crop[selected_state]
            if common_crop in crop_scores and common_crop in price_map:
                if common_crop not in [c[0] for c in recommended]:
                    recommended.insert(0, (common_crop, crop_scores[common_crop], price_map[common_crop]))

        recommended = sorted(recommended, key=lambda x: x[1], reverse=True)[:5]

        if recommended:
            st.success(_("✅ Top Recommended Crops (Priority to Common Crop):"))
            for crop, score, price in recommended:
                st.write(f"🌿 {_(crop)} — ₹{price:.0f}/tonne")
        else:
            st.warning(_("❌ No crops found within your budget."))
except FileNotFoundError:
    st.warning(_("⚠ Please upload data_core.csv."))
