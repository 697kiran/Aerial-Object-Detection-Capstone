import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

st.set_page_config(
    page_title="Aerial Surveillance AI",
    page_icon="🚁",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #f4f4f4; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; font-weight: bold;}
    div[data-testid="stMetricValue"] { font-size: 1.5rem; }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.title("🔧 Control Panel")
app_mode = st.sidebar.selectbox("Select Module", ["Classification (Bird vs Drone)", "Object Detection (YOLO)"])

st.sidebar.markdown("---")
st.sidebar.info("Model Configuration")

# DEFAULT TO YOUR SPECIFIC FILENAMES
cls_model_name = st.sidebar.text_input("Classifier File", "transfer_legacy.h5")
yolo_model_name = st.sidebar.text_input("YOLO File", "best_yolo.pt")

# --- LOADERS ---
@st.cache_resource
def load_cls_model(path):
    if os.path.exists(path):
        try:
            return tf.keras.models.load_model(path)
        except Exception as e:
            st.error(f"Error loading Keras model: {e}")
            return None
    return None

@st.cache_resource
def load_yolo_model(path):
    if YOLO_AVAILABLE and os.path.exists(path):
        try:
            return YOLO(path)
        except Exception as e:
            st.error(f"Error loading YOLO model: {e}")
            return None
    return None

def preprocess_cls_image(image):
    # Resize to 224x224 as required by MobileNetV2
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- MAIN UI ---
st.title("🚁 Aerial Object Classification & Detection")
st.markdown("### AI Solution for Surveillance & Wildlife Monitoring")

uploaded_file = st.file_uploader("Upload Aerial Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Input Image", use_container_width=True)

    with col2:
        # === CLASSIFICATION MODE ===
        if app_mode == "Classification (Bird vs Drone)":
            st.subheader("🔍 Classification Results")

            # Check if model file exists before loading
            if not os.path.exists(cls_model_name):
                st.error(f"❌ File not found: `{cls_model_name}`")
                st.warning("Please ensure `best_transfer_model.h5` is in this folder.")
            else:
                model = load_cls_model(cls_model_name)

                if st.button("Analyze Image", type="primary"):
                    if model:
                        with st.spinner("Processing pixels..."):
                            processed_img = preprocess_cls_image(image)
                            pred = model.predict(processed_img)
                            confidence = pred[0][0]

                            # Logic: 0=Bird, 1=Drone
                            # (Adjust this if your specific training run swapped them)
                            if confidence > 0.5:
                                label = "DRONE DETECTED"
                                conf_val = confidence
                                color = "red"
                                icon = "🛸"
                                msg = "Unidentified aerial object in restricted zone."
                            else:
                                label = "BIRD DETECTED"
                                conf_val = 1 - confidence
                                color = "green"
                                icon = "🦅"
                                msg = "Biological activity detected. No threat."

                            st.markdown(f"## :{color}[{icon} {label}]")
                            st.metric("Confidence Score", f"{conf_val*100:.2f}%")
                            st.progress(int(conf_val*100))
                            st.info(msg)

        # === DETECTION MODE ===
        elif app_mode == "Object Detection (YOLO)":
            st.subheader("🎯 Object Detection Results")

            if not YOLO_AVAILABLE:
                st.error("❌ `ultralytics` not installed. Run `pip install ultralytics`.")
            elif not os.path.exists(yolo_model_name):
                st.error(f"❌ File not found: `{yolo_model_name}`")
                st.warning("Please ensure `best_yolo.pt` is in this folder.")
            else:
                yolo = load_yolo_model(yolo_model_name)

                if st.button("Detect Objects", type="primary"):
                    if yolo:
                        with st.spinner("Running YOLO Inference..."):
                            # Run prediction
                            results = yolo.predict(image, conf=0.4) # conf=0.4 filters weak predictions

                            # Plot
                            res_plotted = results[0].plot()
                            res_img = Image.fromarray(res_plotted[..., ::-1]) # Fix RGB/BGR
                            st.image(res_img, caption="YOLO Output", use_container_width=True)

                            # Stats
                            boxes = results[0].boxes
                            if len(boxes) > 0:
                                st.success(f"✅ Detected {len(boxes)} object(s).")
                            else:
                                st.warning("No objects detected with high confidence.")

st.divider()
st.caption("Capstone Project | Deep Learning")