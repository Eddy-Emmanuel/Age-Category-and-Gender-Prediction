import io
import pathlib
import numpy as np
import streamlit as st
from fastai.vision.all import *

def Main(age_learner, gender_learner):
    st.title(body="GENDER AND AGE CATEGORY PREDICTION")

    uploaded_image = st.file_uploader(label="Upload headshot image of yourself.")
    
    if uploaded_image is not None:
        byte_img = PILImage.create(io.BytesIO(uploaded_image.read()))
        rbg_image = byte_img.convert("RGB")
        RESIZED_IMAGE = rbg_image.resize(size=(224, 224))
        image = np.array(RESIZED_IMAGE)
        
        # Make prediction
        age_pred, _, _ = age_learner.predict(image)
        gender_pred, _, _ = gender_learner.predict(image)
        
        # Display Image and prediction
        st.image(image=image, caption="Output")
        
        # Display prediction
        st.write(f"Gender: {gender_pred}", unsafe_allow_html=False)
        st.write(f"Age Category: {age_pred.title()}", unsafe_allow_html=False)
    
if __name__ == "__main__":
    proxipath = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        age_learner = load_learner(r"model\age_learner.pkl")
        gender_learner = load_learner(r"model\gender_model.pkl")
        print("Sucessfully Loaded model")
    finally:
        pathlib.PosixPath = proxipath
        
    Main(age_learner, gender_learner)
