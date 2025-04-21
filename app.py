# Libraries
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

# Load Function (requires plant_disease_data.csv)
def load_disease_data():
    df = pd.read_csv("plant_disease_data.csv")
    disease_db = {}     # empty dictionary
    for i, row in df.iterrows():
        if row['Disease'] != 'N/A':
            disease_db[row['Class Name']] = {
                "disease": row['Disease'],
                "symptoms": row['Symptoms'],
                "treatment": row['Treatment']
            }
    return disease_db, df['Class Name'].unique().tolist()
disease_db, class_name = load_disease_data()

# Model Function
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert to batch
    prediction = model.predict(input_arr)   # Predict on array
    return np.argmax(prediction), prediction # Both index and prediction array

st.title("Plant Disease Scanner")

# Drag & Drop Uploader
test_image = st.file_uploader("Upload a plant leaf photo: ", type=['JPG','jpg','jpeg','png'],
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

            except UnidentifiedImageError:
                st.error("⚠️ Corrupt image - please try another file")
            except Exception as e:
                st.error(f"⚠️ Prediction failed: {str(e)}")