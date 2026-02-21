import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import json
import os
import datetime
import pandas as pd
import re
from fpdf import FPDF

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="Gau-Raksha AI Pro",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if 'language' not in st.session_state: st.session_state.language = 'English'

# --- 2. BREED DATA MAPPING (Location & State) ---
breed_meta = {
    # NORTH INDIA
    "Sahiwal": [30.9, 74.0, "Punjab"], "Hariana": [29.0, 76.0, "Haryana"], 
    "Mewati": [27.9, 77.0, "Haryana/Rajasthan"], "Rathi": [28.0, 73.3, "Rajasthan"], 
    "Nagori": [27.2, 73.7, "Rajasthan"], "Tharparkar": [26.9, 70.9, "Rajasthan"],
    "Ponwar": [28.6, 79.8, "Uttar Pradesh"], "Kherigarh": [27.9, 80.7, "Uttar Pradesh"], 
    "Gangatiri": [25.3, 83.0, "Uttar Pradesh/Bihar"], "Belahi": [30.7, 76.7, "Haryana"], 
    "Badri": [30.0, 79.0, "Uttarakhand"], "Himachali Pahari": [31.1, 77.1, "Himachal Pradesh"],
    "Ladakhi": [34.1, 77.5, "Ladakh"], "Siri": [27.3, 88.6, "Sikkim"], 
    "Lakhimi": [26.2, 92.9, "Assam"], "Thutho": [26.1, 94.5, "Nagaland"], 
    "Bachaur": [26.6, 85.5, "Bihar"], "Purnea": [25.7, 87.4, "Bihar"],

    # WEST INDIA
    "Gir": [21.1, 70.8, "Gujarat"], "Kankrej": [23.2, 69.6, "Gujarat"], 
    "Red Sindhi": [25.3, 68.3, "Gujarat/Pakistan"], "Dangi": [20.9, 73.8, "Gujarat"], 
    "Nari": [24.8, 72.8, "Rajasthan/Gujarat"], "Dagri": [22.8, 74.2, "Gujarat"],
    "Malvi": [23.5, 75.8, "Madhya Pradesh"], "Nimari": [21.8, 76.3, "Madhya Pradesh"], 
    "Kenkatha": [24.8, 80.0, "Uttar Pradesh/MP"],

    # CENTRAL & SOUTH INDIA
    "Ongole": [15.5, 80.0, "Andhra Pradesh"], "Krishna Valley": [16.5, 74.8, "Karnataka"], 
    "Deoni": [18.4, 77.0, "Maharashtra"], "Khillar": [17.6, 75.9, "Maharashtra"], 
    "Red Kandhari": [19.1, 77.3, "Maharashtra"], "Gaolao": [20.7, 78.6, "Maharashtra"],
    "Kokan Kapila": [17.0, 73.3, "Maharashtra/Goa"], "Shweta Kapila": [15.2, 74.0, "Goa"],
    "Amritmahal": [13.5, 75.8, "Karnataka"], "Hallikar": [12.3, 76.6, "Karnataka"], 
    "Malnad Gidda": [13.5, 75.0, "Karnataka"], "Kangayam": [11.0, 77.5, "Tamil Nadu"], 
    "Bargur": [11.8, 77.5, "Tamil Nadu"], "Umblachery": [10.7, 79.8, "Tamil Nadu"],
    "Pulikulam": [9.8, 78.5, "Tamil Nadu"], "Punganur": [13.3, 78.5, "Andhra Pradesh"], 
    "Poda Thurpu": [16.5, 78.5, "Telangana"], "Vechur": [9.5, 76.5, "Kerala"],

    # EAST INDIA
    "Binjharpuri": [20.5, 85.5, "Odisha"], "Ghumusari": [19.5, 84.5, "Odisha"], 
    "Khariar": [20.3, 82.5, "Odisha"], "Motu": [18.3, 81.5, "Odisha"], 
    "Kosali": [21.2, 81.6, "Chhattisgarh"]
}
DEFAULT_LOCATION = [20.5937, 78.9629]

# --- 3. TRANSLATIONS ---
translations = {
    'English': {
        'app_name': "Gau-Raksha AI",
        'status_on': "AI Engine Active", 'status_off': "AI Engine Offline",
        'title': "Gau-Raksha AI", 'subtitle': "Indigenous Cattle Identification System",
        'upload_label': "Upload a cow image here", 'upload_help': "Supported formats: JPG, PNG",
        'analyze': "Identify Breed", 'input_cap': "Input Image",
        'results': "Analysis Results", 'milk': "Milk (L/Day)",
        'fat': "Fat %", 'value': "Market Price (₹)",
        'diet_label': "🥗 Best Food / Diet",
        'calc_title': "💰 ROI Calculator", 'map_title': "🗺️ Native Region",
        'status_title': "📉 Conservation", 'report_btn': "Download PDF Report",
        'desc_header': "Description", 'top3': "Alternative Matches (Top 3)",
        'wait_msg': "👆 Upload an image to start.",
        'analyzing': "🧬 Sequencing Genotypes (TTA Enabled)...",
        'conf': "Confidence", 'profit': "Est. Monthly Income", 'loss': "Loss",
        'month': "Month", 'cost': "Cost (₹/Mo)", 'price_l': "Milk Price (₹/L)",
        'safe': "Safe", 'endangered': "Endangered", 'vulnerable': "Vulnerable"
    },
    'Hindi': {
        'app_name': "गौ-रक्षा एआई",
        'status_on': "एआई इंजन सक्रिय", 'status_off': "एआई इंजन बंद",
        'title': "गौ-रक्षा एआई", 'subtitle': "स्वदेशी नस्ल पहचान प्रणाली",
        'upload_label': "यहाँ गाय की फोटो अपलोड करें", 'upload_help': "समर्थित प्रारूप: JPG, PNG",
        'analyze': "नस्ल पहचानें", 'input_cap': "इनपुट छवि",
        'results': "विश्लेषण परिणाम", 'milk': "दूध (लीटर/दिन)",
        'fat': "वसा %", 'value': "बाजार मूल्य (₹)",
        'diet_label': "🥗 सर्वोत्तम आहार / चारा",
        'calc_title': "💰 लाभ कैलकुलेटर", 'map_title': "🗺️ मूल क्षेत्र",
        'status_title': "📉 संरक्षण स्थिति", 'report_btn': "रिपोर्ट डाउनलोड करें (PDF)",
        'desc_header': "विवरण", 'top3': "अन्य संभावित नस्लें",
        'wait_msg': "👆 शुरू करने के लिए एक छवि अपलोड करें।",
        'analyzing': "🧬 जीनोटाइप का विश्लेषण...",
        'conf': "विश्वास", 'profit': "अनुमानित मासिक आय", 'loss': "हानि",
        'month': "महीना", 'cost': "लागत (₹/महीना)", 'price_l': "दूध की कीमत (₹/लीटर)",
        'safe': "सुरक्षित", 'endangered': "लुप्तप्राय", 'vulnerable': "कमजोर"
    },
    'Gujarati': {
        'app_name': "ગૌ-રક્ષા એઆઈ",
        'status_on': "એઆઈ એન્જિન સક્રિય", 'status_off': "એઆઈ એન્જિન બંધ",
        'title': "ગૌ-રક્ષા એઆઈ", 'subtitle': "સ્વદેશી ગાય ઓળખ સિસ્ટમ",
        'upload_label': "અહીં ગાયનો ફોટો અપલોડ કરો", 'upload_help': "સપોર્ટેડ ફોર્મેટ્સ: JPG, PNG",
        'analyze': "જાતિ ઓળખો", 'input_cap': "ઇનપુટ ઇમેજ",
        'results': "પરિણામો", 'milk': "દૂધ (લિટર/દિવસ)",
        'fat': "ચરબી %", 'value': "કિંમત (₹)",
        'diet_label': "🥗 શ્રેષ્ઠ ખોરાક / આહાર",
        'calc_title': "💰 નફો કેલ્ક્યુલેટર", 'map_title': "🗺️ જાતિ વિસ્તાર",
        'status_title': "📉 સંરક્ષણ", 'report_btn': "રિપોર્ટ ડાઉનલોડ કરો (PDF)",
        'desc_header': "વર્ણન", 'top3': "અન્ય શક્યતાઓ",
        'wait_msg': "👆 શરૂ કરવા માટે ઇમેજ અપલોડ કરો.",
        'analyzing': "🧬 જીનોટાઇપ વિશ્લેષણ...",
        'conf': "વિશ્વાસ", 'profit': "માસિક આવક", 'loss': "નુકસાન",
        'month': "મહિનો", 'cost': "ખર્ચ (₹/મહિનો)", 'price_l': "દૂધની કિંમત (₹/લિટર)",
        'safe': "સુરક્ષિત", 'endangered': "લુપ્તપ્રાય", 'vulnerable': "જોખમી"
    }
}
def get_text(key): return translations[st.session_state.language][key]

# --- 4. CSS STYLING ---
bg_color = "#0E1117"
text_color = "#FAFAFA"
card_bg = "#262730"
border_color = "#464855"

st.markdown(f"""
    <style>
    .stApp {{ background-color: {bg_color} !important; color: {text_color} !important; }}
    p, h1, h2, h3, h4, label, li, span {{ color: {text_color} !important; }}
    .stButton>button {{
        background-color: #FF4B4B; color: white !important; 
        border-radius: 8px; font-weight: bold; height: 3em; width: 100%; border: none;
    }}
    div[data-testid="stMetric"] {{
        background-color: {card_bg} !important;
        border: 1px solid {border_color};
        padding: 10px; border-radius: 10px;
        text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        height: auto !important; min-height: 100px;
    }}
    div[data-testid="stMetricValue"] {{ 
        color: {text_color} !important; font-weight: bold; font-size: 1.1rem !important; 
        white-space: normal !important; line-height: 1.4 !important;
    }}
    button[data-baseweb="tab"] {{
        background-color: transparent !important; color: {text_color} !important; font-weight: 600;
    }}
    div[data-baseweb="tab-highlight"] {{ background-color: #FF4B4B !important; }}
    div[data-baseweb="select"] > div, div[data-baseweb="select"], label[data-testid="stWidgetLabel"] {{
        cursor: pointer !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- 5. LOGIC & DATA LOADING ---
@st.cache_resource
def load_resources():
    if not os.path.exists('cow_model.h5'): return None, None, None
    model = tf.keras.models.load_model('cow_model.h5')
    with open('class_indices.json', 'r') as f: class_indices = json.load(f)
    class_names = {int(k): v for k, v in class_indices.items()}
    breed_data = {}
    if os.path.exists('breed_data.json'):
        with open('breed_data.json', 'r', encoding='utf-8') as f: breed_data = json.load(f)
    return model, class_names, breed_data

def parse_range_mean(value_str):
    try:
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", str(value_str))
        if not numbers: return 0
        return sum(map(float, numbers)) / len(numbers)
    except: return 0

model, class_names, breed_info = load_resources()

# --- 6. SMART DATA LOOKUP (FIXES CAPITALIZATION ERRORS) ---
def get_breed_data(breed_name, database):
    # 1. Try Exact Match
    if breed_name in database:
        return database[breed_name], breed_name
    
    # 2. Try Title Case (motu -> Motu)
    if breed_name.title() in database:
        return database[breed_name.title()], breed_name.title()
        
    # 3. Try removing underscores (red_sindhi -> Red Sindhi)
    cleaned = breed_name.replace("_", " ").title()
    if cleaned in database:
        return database[cleaned], cleaned
        
    # 4. Fallback: Search Case-Insensitive
    for key in database.keys():
        if key.lower() == breed_name.replace("_", " ").lower():
            return database[key], key
            
    # 5. Give up
    return {}, breed_name.title()

# --- 7. PDF GENERATOR ---
def create_pdf(breed, score, info, img):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Gau-Raksha AI - Diagnostic Report", ln=True, align='C')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 5, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
    pdf.line(10, 25, 200, 25)
    
    # Image
    img_path = "temp_report_img.png"
    try:
        img.save(img_path)
        pdf.image(img_path, x=55, y=30, w=100)
    except: pass
    
    pdf.set_y(115)
    pdf.set_fill_color(220, 230, 255)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, f"DETECTED BREED: {breed} ({score:.1f}%)", ln=True, align='C', fill=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 10)
    col_w_1, col_w_2, row_h = 70, 120, 7
    pdf.cell(col_w_1, row_h, "Parameter", 1, 0, 'C')
    pdf.cell(col_w_2, row_h, "Value", 1, 1, 'C')
    pdf.set_font("Arial", '', 10)
    
    def clean(text): return str(text).replace("₹", "Rs. ").encode('latin-1', 'replace').decode('latin-1')
    data = [
        ("Milk Yield (Daily)", info.get('milk_yield', 'N/A')),
        ("Fat Percentage", info.get('fat_percentage', 'N/A')),
        ("Market Value", info.get('market_value', 'N/A')),
        ("Best Diet", info.get('diet', 'N/A')),
        ("Climate Tolerance", info.get('climate_tolerance', 'N/A'))
    ]
    for param, val in data:
        pdf.cell(col_w_1, row_h, param, 1, 0)
        pdf.cell(col_w_2, row_h, clean(val), 1, 1)
        
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, "Breed Description:", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.multi_cell(0, 5, clean(info.get('description', 'No description available.')))
    pdf.set_y(-10)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, "Gau-Raksha AI | Hackathon Project 2026", 0, 0, 'C')
    return pdf.output(dest='S').encode('latin-1')

# --- 8. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2395/2395796.png", width=80)
    st.title(get_text('app_name'))
    lang = st.selectbox("🌐 Language", ['English', 'Hindi', 'Gujarati'])
    if lang != st.session_state.language:
        st.session_state.language = lang
        st.rerun()
    st.markdown("---")
    if model: st.success(f"🟢 {get_text('status_on')}")
    else: st.error(f"🔴 {get_text('status_off')}")

# --- 9. MAIN LAYOUT ---
st.title(f"🐄 {get_text('title')}")
st.markdown(f"*{get_text('subtitle')}*")

if not model:
    st.warning("⚠️ Model not detected. Run `python train_model.py`.")
    st.stop()

uploaded_file = st.file_uploader(get_text('upload_label'), type=["jpg", "png", "jpeg"], help=get_text('upload_help'))

if uploaded_file:
    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        img_obj = Image.open(uploaded_file).convert('RGB')
        st.image(img_obj, caption=get_text('input_cap'), use_container_width=True)
        analyze_btn = st.button(f"🚀 {get_text('analyze')}")

    if analyze_btn or 'analyzed_breed' in st.session_state:
        if analyze_btn:
            with st.spinner(get_text('analyzing')):
                img = img_obj.resize((224, 224))
                
                # --- TTA PREDICTION ---
                img_1 = np.array(img)
                img_1 = np.expand_dims(img_1, axis=0)
                img_flip = ImageOps.mirror(img)
                img_2 = np.array(img_flip)
                img_2 = np.expand_dims(img_2, axis=0)
                
                p1 = model.predict(img_1)
                p2 = model.predict(img_2)
                avg_pred = (p1 + p2) / 2.0
                
                top_3_indices = np.argsort(avg_pred[0])[-3:][::-1]
                
                # RAW PREDICTION FROM MODEL (e.g., "motu" or "red_sindhi")
                raw_pred_name = class_names[top_3_indices[0]]
                
                # SMART LOOKUP: Finds the correct Key in JSON ("Motu", "Red Sindhi")
                found_data, correct_name = get_breed_data(raw_pred_name, breed_info)
                
                # Save to session state
                st.session_state.analyzed_breed = correct_name 
                st.session_state.analyzed_score = avg_pred[0][top_3_indices[0]] * 100
                st.session_state.analyzed_info = found_data # Save the found data directly
                
                # Process Top 3 for display (Title Case them)
                st.session_state.top_3_data = []
                for i in top_3_indices:
                    r_name = class_names[i]
                    # Use same smart lookup for these to get nice formatting
                    _, nice_name = get_breed_data(r_name, breed_info)
                    st.session_state.top_3_data.append((nice_name, avg_pred[0][i]*100))

        breed = st.session_state.analyzed_breed
        score = st.session_state.analyzed_score
        # Use the data found by the Smart Lookup
        info = st.session_state.analyzed_info 

        with col2:
            clr = "#28a745" if score > 70 else "#ffc107"
            if score < 40: clr = "#dc3545"
            st.markdown(f"""
            <div style="background:{clr};padding:15px;border-radius:10px;text-align:center;color:white;font-weight:bold;margin-bottom:15px;">
                {breed} • {score:.1f}% {get_text('conf')}
            </div>
            """, unsafe_allow_html=True)

            with st.expander(f"📌 {get_text('top3')}", expanded=True):
                for b_name, b_score in st.session_state.top_3_data:
                    st.progress(int(b_score))
                    st.caption(f"{b_name}: {b_score:.1f}%")

            m1, m2 = st.columns(2)
            m1.metric(get_text('milk'), info.get('milk_yield', 'N/A'))
            m2.metric(get_text('fat'), info.get('fat_percentage', 'N/A'))
            
            # DIET DISPLAY
            st.markdown(f"**{get_text('diet_label')}**")
            st.info(info.get('diet', 'Information not available in database.'))
            
            price_val = info.get('market_value', 'N/A')
            st.markdown(f"""
            <div style="background-color:{card_bg}; border:1px solid {border_color}; padding:15px; border-radius:10px; text-align:center; margin-top:10px;">
                <p style="margin:0; opacity:0.8; font-size:0.9rem;">{get_text('value')}</p>
                <h4 style="margin:5px 0 0 0; color:{text_color}; font-size:1.1rem; white-space: normal;">{price_val}</h4>
            </div>
            """, unsafe_allow_html=True)

    if 'analyzed_breed' in st.session_state:
        st.markdown("---")
        t1, t2, t3, t4 = st.tabs([f"🗺️ {get_text('map_title')}", f"💰 {get_text('calc_title')}", f"📉 {get_text('status_title')}", "📄 Report"])

        with t1:
            coords = breed_meta.get(breed, [20.0, 78.0, "India"])
            state_name = coords[2] if len(coords) > 2 else "Native Region"
            st.write(f"### 📍 {state_name}")
            map_df = pd.DataFrame({'lat': [coords[0]], 'lon': [coords[1]]})
            st.map(map_df, zoom=5, size=5000, color='#FF4B4B')

        with t2:
            avg_milk = parse_range_mean(info.get('milk_yield', '0'))
            price = st.slider(get_text('price_l'), 30, 100, 60)
            maint = st.number_input(get_text('cost'), value=3000)
            profit = (avg_milk * price * 30) - maint
            color = "green" if profit > 0 else "red"
            res_text = get_text('profit') if profit > 0 else get_text('loss')
            st.markdown(f"<h3 style='color:{color} !important'>{res_text}: ₹{abs(profit):,.0f} / {get_text('month')}</h3>", unsafe_allow_html=True)

        with t3:
            endangered = ["Vechur", "Punganur", "Krishna Valley", "Red Kandhari", "Ponwar"]
            vulnerable = ["Sahiwal", "Red Sindhi", "Tharparkar"]
            
            if breed in endangered:
                status = get_text('endangered')
                s_color = "red"
            elif breed in vulnerable:
                status = get_text('vulnerable')
                s_color = "orange"
            else:
                status = get_text('safe')
                s_color = "green"
            
            st.markdown(f"### **Status:** <span style='color:{s_color}'>{status}</span>", unsafe_allow_html=True)

        with t4:
            st.write("### 📄 Diagnostic Report")
            pdf_data = create_pdf(breed, score, info, img_obj)
            st.download_button(
                label=f"📥 {get_text('report_btn')}",
                data=pdf_data,
                file_name=f"Gau-Raksha_Report_{breed}.pdf",
                mime="application/pdf"
            )

else:
    st.info(get_text('wait_msg'))