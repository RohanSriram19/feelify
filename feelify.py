import streamlit as st
from fer import FER
import cv2
import numpy as np
from PIL import Image
import random

# Affirmations based on mood
affirmations = {
    "happy": ["Keep smiling, it suits you!", "Happiness looks great on you!", "Joy is contagious – spread it!"],
    "sad": ["It's okay to feel down sometimes.", "You're stronger than you think.", "Better days are coming."],
    "angry": ["Breathe in, breathe out. You’ve got this.", "Let it go – peace is power.", "Frustration means you care."],
    "neutral": ["Stay calm, stay centered.", "Neutral is a good place to reset.", "Balance is strength."],
    "surprise": ["Something unexpected? Embrace the moment!", "Stay curious – life’s full of surprises!"],
    "fear": ["Courage isn’t the absence of fear, but action in its face.", "You are safe and in control."],
    "disgust": ["Not everything will be pretty, but you are resilient.", "Refocus your energy."]
}

# Emoji map
emoji_map = {
    "happy": "😄",
    "sad": "😢",
    "angry": "😠",
    "neutral": "😐",
    "surprise": "😲",
    "fear": "😨",
    "disgust": "🤢"
}

# Streamlit setup
st.set_page_config(page_title="Feelify - Mood Detection App", page_icon="🧠")
st.title("🧠 Feelify: Understand Your Mood Through AI")

# Accept common image types, including uppercase extensions
uploaded_file = st.file_uploader(
    "Upload a selfie (JPG, JPEG, PNG)", 
    type=["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Your uploaded image", use_container_width=True)

    # Convert image to BGR for FER
    img_array = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    detector = FER()

    result = detector.top_emotion(img_bgr)

    st.write("Debug:", result)  # Show what FER returns

    if result is not None and result[0] is not None:
        emotion, score = result
        st.subheader(f"Detected Mood: **{emotion.capitalize()}** {emoji_map.get(emotion, '')}")
        st.text(f"Confidence: {round(score * 100, 2)}%")
        if emotion in affirmations:
            st.markdown(f"💬 *{random.choice(affirmations[emotion])}*")
    else:
        st.warning("😕 Couldn't detect a face. Try a clearer, front-facing selfie.")
