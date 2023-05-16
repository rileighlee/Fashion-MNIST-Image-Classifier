import streamlit as st
import tensorflow as tf
import builtins
import pathlib

@st.cache(allow_output_mutation=True, hash_funcs={builtins.function: id})
def load_model():
    model = tf.keras.models.load_model('model.h5')
    return model

# Workaround for a Streamlit bug related to the __file__ attribute
# See https://github.com/streamlit/streamlit/issues/2433 for more details
def patch_pathlib():
    try:
        pathlib.Path(__file__)
    except NameError:
        pathlib.Path(__file__).resolve().__class__ = pathlib.PosixPath

patch_pathlib()

model = load_model()

st.write("""
# Clothes Detection System
""")

file = st.file_uploader("Choose clothes photo from computer", type=["jpg", "png"])

from PIL import Image, ImageOps
import numpy as np

def import_and_predict(image_data, model):
    size = (64, 64)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    string = "OUTPUT : " + class_names[np.argmax(prediction)]
    st.success(string)
