import streamlit as st
import numpy as np
import cv2
from PIL import Image
from fer_custom import FER

st.set_page_config(page_title="Feelify: Understand Your Mood", page_icon="🧠")

st.title("🧠 Feelify: Understand Your Mood Through AI")
st.markdown("Upload a selfie (JPG, JPEG, PNG)")

uploaded_file = st.file_uploader(" ", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display uploaded image
    st.image(image, caption="Your uploaded image", use_container_width=True)

    # Run emotion detection
    detector = FER(mtcnn=True)
    results = detector.detect_emotions(image)

    if results:
        emotion_scores = results[0]["emotions"]
        emotion = max(emotion_scores, key=emotion_scores.get)
        emoji_map = {
            "happy": "😄",
            "angry": "😠",
            "disgust": "🤢",
            "fear": "😨",
            "sad": "😢",
            "surprise": "😲",
            "neutral": "😐"
        }
        st.subheader(f"Detected Mood: **{emotion.capitalize()}** {emoji_map.get(emotion, '')}")
    else:
        st.warning("😕 Couldn't detect a face. Try a clearer photo.")
