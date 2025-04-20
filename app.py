# Libraries
import streamlit as st
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import requests
import tempfile
import os

# Load Function (requires plant_disease_data.csv)
def load_disease_data():
    df = pd.read_csv("plant_disease_data.csv")
    disease_db = {}     # empty dictionary
    for i, row in df.iterrows():
        if row['Disease'] != 'N/A':
            disease_db[row['Disease']] = {
                "symptoms": row['Symptoms'],
                "treatment": row['Treatment']
            }
    return disease_db, df['Class Name'].unique().tolist()
disease_db, class_name = load_disease_data()





# Model Load Function
@st.cache_resource(show_spinner="⚙️ Loading AI model...", ttl=24*3600)
def load_model():
    # Model URL and local cache path
    model_url = "https://github.com/Aniket1409/Plant-Disease-Scanner/releases/download/v1.0.0/model.keras"
    cache_dir = Path(tempfile.gettempdir()) / "plant_disease_models"
    cache_dir.mkdir(exist_ok=True)
    model_path = cache_dir / "model.keras"
    
    try:
        # Download if not cached
        if not model_path.exists():
            with st.spinner('Downloading model (this may take a few minutes)...'):
                response = requests.get(model_url, stream=True)
                response.raise_for_status()
                
                # Write to temporary file first
                temp_path = model_path.with_suffix('.tmp')
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Verify download completed
                if temp_path.stat().st_size == 0:
                    raise ValueError("Downloaded file is empty")
                
                # Rename temp file to final name
                temp_path.rename(model_path)
        
        # Load model
        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully!")
        return model
        
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        # Try to clean up corrupted files
        if 'temp_path' in locals() and temp_path.exists():
            temp_path.unlink()
        if model_path.exists():
            model_path.unlink()
        return None

# Load the model
model = load_model()

if model is None:
    st.error("Failed to load model. The application cannot continue.")
    st.stop()





# Prediction Function
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert to batch
    prediction = model.predict(input_arr)   # Predict on array
    return np.argmax(prediction), prediction # Both index and prediction array

st.title("Plant Disease Scanner")

# Camera Input, Drag & Drop Uploader
img_file_buffer = st.camera_input("Take a photo of the leaf")
if img_file_buffer:
    test_image = img_file_buffer
else:
    test_image = st.file_uploader("Upload a plant leaf photo: ", type=['jpg','jpeg','png'],
                                key="uploader", accept_multiple_files=False,
                                help="Take or upload a clear photo of a plant leaf")

# Image Error Handling
if test_image is not None:
    try: Image.open(test_image); st.image(test_image, use_column_width=True)
    except UnidentifiedImageError:
        st.error("⚠️ Invalid image"); test_image = None  # Clear the invalid upload
    except Exception as e:
        st.error(f"⚠️ Error reading image: {str(e)}"); test_image = None

# Predict Button with loading spinner, confidence meter, result (green), error (red)
if st.button("Predict Disease"):
    if test_image is None:
        st.warning("⚠️ Please upload an image first!")
    else:
        with st.spinner("Analyzing the image..."):
            try:
                result_index, predictions = model_prediction(test_image)
                confidence = np.max(predictions) * 100
                st.success(f"🌱 {class_name[result_index]}")
                st.progress(int(confidence))
                st.caption(f"Confidence: {confidence:.1f}%")                
                if result_index is not None:
                    with st.expander(f"ℹ️ About {class_name[result_index]}"):
                        disease_info = disease_db.get(class_name[result_index], 
                            {"symptoms": "Information not available",
                            "treatment": "Consult an agricultural expert"})
                        st.subheader("Symptoms")
                        st.write(disease_info["symptoms"])
                        st.subheader("Treatment")
                        st.write(disease_info["treatment"])

            except TypeError:
                st.error("⚠️ Invalid file format - please upload an image")
            except UnidentifiedImageError:
                st.error("⚠️ Corrupt image - please try another file")
            except Exception as e:
                st.error(f"⚠️ Prediction failed: {str(e)}")
