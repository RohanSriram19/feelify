import streamlit as st
from fer import FER
import cv2
import numpy as np
from PIL import Image
import random

# Set page config with emoji
st.set_page_config(page_title="🧠 Feelify", layout="centered")

# Inject custom CSS for background + boxes
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #f2f7fd, #fdf2f6);
    }
    .stApp {
        background: linear-gradient(135deg, #fceef3 0%, #e9f0fc 100%);
    }
    .mood-box {
        background-color: #e3fcef;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #b8e2d6;
        margin-bottom: 1rem;
    }
    .affirmation-box {
        background-color: #fff6e5;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ffe8b3;
    }
    </style>
""", unsafe_allow_html=True)

# Affirmations
affirmations = {
    "happy": ["Keep smiling, it suits you!", "Happiness looks great on you!", "Joy is contagious – spread it!"],
    "sad": ["It's okay to feel down sometimes.", "You're stronger than you think.", "Better days are coming."],
    "angry": ["Breathe in, breathe out. You’ve got this.", "Let it go – peace is power.", "Frustration means you care."],
    "neutral": ["Stay calm, stay centered.", "Neutral is a good place to reset.", "Balance is strength."],
    "surprise": ["Something unexpected? Embrace the moment!", "Stay curious – life’s full of surprises!"],
    "fear": ["Courage isn’t the absence of fear, but action in its face.", "You are safe and in control."],
    "disgust": ["Not everything will be pretty, but you are resilient.", "Refocus your energy."]
}

emoji_map = {
    "happy": "😄", "sad": "😢", "angry": "😠", "neutral": "😐",
    "surprise": "😲", "fear": "😨", "disgust": "🤢"
}

# Title
st.markdown("<h1 style='text-align: center;'>🧠 Feelify</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem;'>Discover how you feel, one selfie at a time.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📸 Upload a selfie (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Your uploaded image", use_container_width=True)

    # Convert image
    img_array = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    detector = FER()  # No MTCNN for stability
    result = detector.top_emotion(img_bgr)

    if result and result[0] is not None:
        emotion, score = result

        st.markdown(f"""
            <div class='mood-box'>
                <h3>Detected Mood: {emotion.capitalize()} {emoji_map.get(emotion, '')}</h3>
                <p><b>Confidence:</b> {round(score * 100, 2)}%</p>
            </div>
        """, unsafe_allow_html=True)

        if emotion in affirmations:
            st.markdown(f"""
                <div class='affirmation-box'>
                    💬 <i>{random.choice(affirmations[emotion])}</i>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("😕 Couldn't detect a face. Try a clearer, front-facing selfie.")
