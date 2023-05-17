import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

model = keras.models.load_model('model.h5')

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def preprocess_image(image):
    image = image.resize((28, 28))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image
  
def app():
    st.title('Fashion MNIST Image Classifier')
    file = st.file_uploader('Upload a Fashion MNIST image', type=['jpg', 'png'])
    
    if file is not None:
        image = Image.open(file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        image = preprocess_image(image)
        prediction = model.predict(image)
        class_index = np.argmax(prediction)
        class_label = class_names[class_index]
        st.write('The image is ', class_label)
        
if __name__ == '__main__':
    app()
