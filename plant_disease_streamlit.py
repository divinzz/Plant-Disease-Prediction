import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/plant_disease_model.h5" 

model = tf.keras.models.load_model(model_path)

class_indices = json.load(open(f"{working_dir}/class indices.json"))


def load_img(image_path, target_size=(224, 224)):

  img = Image.open(image_path)
  img = img.resize(target_size)
  img_array = np.array(img)
  img_array = np.expand_dims(img_array, axis=0)
  img_array = img_array.astype('float32') / 255.
  return img_array


def predict_image(model, image_path, class_indices):
  preprocessed_img = load_img(image_path)
  predictions = model.predict(preprocessed_img)
  prediction_class_index = np.argmax(predictions, axis=1)[0]
  prediction_class_name = class_indices[str(prediction_class_index)]
  return prediction_class_name

st.title('☘️ Plant Disease Classifier ☘️')

uploaded_image = st.file_uploader("upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
  image  = Image.open(uploaded_image)
  col1, col2 = st.columns(2)

  with col1:
    resized_img = image.resize((150, 150))
    st.image(resized_img)

  with col2:
    if st.button('Classify'):
      prediction = predict_image(model, uploaded_image, class_indices)
      st.success(f"Prediction: **{str(prediction)}**")

    
     