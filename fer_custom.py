import cv2
import numpy as np
from mtcnn import MTCNN
from keras.models import load_model
import os

class FER:
    def __init__(self, mtcnn=True):
        self._detector = MTCNN() if mtcnn else None
        model_path = os.path.join("models", "fer2013_mini_XCEPTION.hdf5")
        self._model = load_model(model_path, compile=False)
        self._target_size = self._model.input_shape[1:3]
        self._emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    def detect_emotions(self, img):
        if self._detector is None:
            return []

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self._detector.detect_faces(img_rgb)
        results = []

        for face in faces:
            (x, y, w, h) = face["box"]
            face_img = img[y:y+h, x:x+w]
            face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_gray, self._target_size)
            face_array = np.expand_dims(np.expand_dims(face_resized, -1), 0)
            face_array = face_array / 255.0

            prediction = self._model.predict(face_array, verbose=0)[0]
            emotions = {label: float(pred) for label, pred in zip(self._emotion_labels, prediction)}
            results.append({
                "box": [x, y, w, h],
                "emotions": emotions
            })

        return results
