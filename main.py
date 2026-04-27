import streamlit as st
import tensorflow as tf
import numpy as np
import json

# Load disease info
with open("disease_info.json") as f:
    disease_data = json.load(f)

# Load model only once (fast)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_model.keras")

model = load_model()

# Prediction function
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home","About","Disease Recognition"])

# Home Page
if(app_mode=="Home"):
    st.header("CROP DISEASE DETECTION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Upload a leaf image and get disease detection + solution instantly.
    """)

# About Page
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
    This dataset contains 87K images of plant leaves across 38 classes.
    Used CNN (TensorFlow) for classification.
    """)

# Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")

    test_image = st.file_uploader("Choose an Image:")

    # Show Image
    if st.button("Show Image"):
        if test_image is not None:
            st.image(test_image, use_column_width=True)
        else:
            st.warning("Please upload an image first!")

    # Predict
    if st.button("Predict"):
        if test_image is not None:   # correct else placement
            with st.spinner("Model is Predicting..."):

                result_index = model_prediction(test_image)

                class_name = [
                    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
                    'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
                    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
                    'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy',
                    'Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
                    'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
                    'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy',
                    'Potato___Early_blight','Potato___Late_blight','Potato___healthy',
                    'Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew',
                    'Strawberry___Leaf_scorch','Strawberry___healthy',
                    'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold',
                    'Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite',
                    'Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy'
                ]

                predicted = class_name[result_index]

                # Clean name
                clean_name = predicted.replace("___"," - ").replace("_"," ")

                st.success(f"🌿 Disease Detected: {clean_name}")

                # Solutions
                st.subheader("Solution")
                st.write(disease_data[predicted]["solution"])

                st.subheader("Prevention")
                st.write(disease_data[predicted]["prevention"])

                st.subheader("Organic Remedy")
                st.write(disease_data[predicted]["organic"])

        else:
            st.warning("Please upload an image first!")