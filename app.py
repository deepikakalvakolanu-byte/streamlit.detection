import streamlit as st
import cv2
import numpy as np
from PIL import Image

# App title
st.set_page_config(page_title="Face Detection App", layout="centered")
st.title("🧠 Human Face Identification App")
st.write("Upload an image and the app will detect human faces.")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Upload image
uploaded_file = st.file_uploader(
    "Upload an Image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    st.subheader("📸 Uploaded Image Preview")
    st.image(image, use_column_width=True)

    # Convert image to OpenCV format
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw rectangles and labels
    for (x, y, w, h) in faces:
        cv2.rectangle(
            img_array, (x, y), (x + w, y + h), (0, 255, 0), 2
        )
        cv2.putText(
            img_array,
            "Human Face Identified",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Display result
    st.subheader("✅ Face Detection Result")
    st.image(img_array, use_column_width=True)

    # Face count
    st.success(f"Number of faces detected: {len(faces)}")

