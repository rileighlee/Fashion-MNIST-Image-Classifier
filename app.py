import streamlit as st
import tensorflow as tf

@st.cache(allow_output_mutation=True, hash_funcs={tf.keras.Model: id})
def load_model():
    model_path = os.path.join(MODEL_DIR, "model.h5")
    model = tf.keras.models.load_model(model_path)
    return model

st.write("# Clothes Detection System")
file = st.file_uploader("Choose clothes photo from computer", type=["jpg", "png"])

import cv2
from PIL import Image,ImageOps
import numpy as np
def import_and_predict(image_data,model):
    size=(64,64)
    image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    model = load_model()  # Load the trained model
    prediction = model.predict(np.array([np.array(image.resize((28, 28)))/255.0]))
    predicted_class = class_names[np.argmax(prediction)]
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    string = "OUTPUT : " + predicted_class
    st.success(string)
