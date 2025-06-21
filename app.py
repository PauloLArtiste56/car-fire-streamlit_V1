import streamlit as st
import tempfile
import cv2
from ultralytics import YOLO
import os

st.set_page_config(page_title="Détection de feu - Vidéo")
st.title("🎥 Détection de feu de voiture dans une vidéo")

model = YOLO("best.pt")

uploaded_video = st.file_uploader("📹 Importez une vidéo", type=["mp4", "avi", "mov", "mkv"])
conf_threshold = st.slider("Seuil de confiance", 0.0, 1.0, 0.25, 0.01)

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    st.info("🔍 Analyse en cours...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(frame, conf=conf_threshold)
        annotated_frame = results[0].plot()

        stframe.image(annotated_frame, channels="BGR", use_container_width=True)

    cap.release()

    import time

    time.sleep(2)
    try:
        os.unlink(video_path)
        st.success("✅ Analyse terminée et fichier temporaire supprimé.")
    except PermissionError:
        st.warning("⚠️ Impossible de supprimer le fichier temporaire (toujours en cours d'utilisation).")
