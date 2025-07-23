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

class_descriptions = {
    "Apple___Apple_scab": "Apple scab is a fungal disease that causes dark, sunken lesions on apple leaves, leading to early leaf drop and deformed, unattractive fruit. Over time, it can severely reduce apple quality, impacting the overall yield and marketability of the fruit.",
    "Apple___Black_rot": "Black rot is a bacterial disease characterized by the appearance of circular, dark lesions on apple leaves and fruit. This infection can cause premature fruit drop, resulting in significant yield loss and reduced fruit quality, making apples unsuitable for sale.",
    "Apple___Cedar_apple_rust": "Cedar apple rust is a fungal disease that causes yellow-orange spots on the leaves of apple trees, often followed by the formation of raised orange pustules. This disease can interfere with fruit development, leading to a decreased apple yield and lower fruit quality.",
    "Apple___healthy": "Healthy apple trees display vibrant green leaves, free from any discoloration, spots, or lesions. The absence of disease symptoms signals robust growth and a well-maintained, thriving apple tree.",
    "Blueberry___healthy": "Healthy blueberry plants have strong, dark green leaves with no signs of disease, pest infestation, or physical damage. The plants grow vigorously, supporting a high potential for fruit production and overall vitality.",
    "Cherry_(including_sour)___Powdery_mildew": "Powdery mildew is a fungal disease that appears as white, powdery spots on cherry leaves. This disease can cause leaf deformation and weaken the plant by hindering photosynthesis, potentially leading to reduced growth and fruit production.",
    "Cherry_(including_sour)___healthy": "Healthy cherry trees exhibit green, fresh leaves, free from any disease or pest damage. The absence of infections indicates that the tree is strong and capable of producing high-quality fruit.",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot": "Cercospora leaf spot, or gray leaf spot, is a fungal disease that causes dark, circular spots with gray centers on corn leaves. This condition leads to early leaf drop, diminishing the plant’s ability to photosynthesize and reducing the overall corn yield.",
    "Corn_(maize)___Common_rust_": "Common rust is a fungal infection that forms reddish-brown pustules on corn leaves. These pustules disrupt the leaf surface, impairing photosynthesis and weakening the plant. As a result, the plant's growth is stunted, reducing overall yield.",
    "Corn_(maize)___Northern_Leaf_Blight": "Northern leaf blight is a fungal disease that causes large, long, grayish lesions on corn leaves. The disease leads to premature leaf death, reduces photosynthetic capacity, and ultimately results in a lower corn yield.",
    "Corn_(maize)___healthy": "Healthy corn plants have lush, vibrant green leaves that are free from visible diseases or physical damage. This indicates that the plant is growing optimally and capable of producing a healthy corn crop.",
    "Grape___Black_rot": "Black rot is a destructive fungal disease that causes dark lesions on grape leaves and shriveled, decaying fruit. It can quickly spread and severely affect grapevine health, reducing the quantity and quality of grapes produced.",
    "Grape___Esca_(Black_Measles)": "Esca, also known as Black Measles, is a fungal infection that results in dark streaks on grapevines, often leading to premature leaf drop. It can dramatically affect grape quality and yield, especially during humid, wet conditions.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Isariopsis leaf spot is a fungal disease that manifests as round spots on grape leaves. These spots reduce the plant’s ability to perform photosynthesis and can weaken the vine, ultimately affecting grape yield and fruit quality.",
    "Grape___healthy": "Healthy grapevines display vibrant green leaves that are free from disease, pest damage, or deformities. This signals that the vine is strong and capable of producing high-quality grapes.",
    "Orange___Haunglongbing_(Citrus_greening)": "Huanglongbing, also known as citrus greening, is a bacterial disease that causes severe yellowing of citrus leaves, resulting in poor fruit development. Infected trees may eventually suffer from fruit drop, drastically reducing yield and fruit quality.",
    "Peach___Bacterial_spot": "Bacterial spot is an infection that causes dark lesions on peach leaves, often leading to defoliation and reduced fruit quality. This disease can severely affect the growth and overall health of peach trees, resulting in lower yield and marketable fruit.",
    "Peach___healthy": "Healthy peach trees have lush, green leaves that show no signs of infection, damage, or disease. This indicates that the tree is growing well and capable of producing strong, high-quality fruit.",
    "Pepper,_bell___Bacterial_spot": "Bacterial spot is a disease that causes water-soaked, dark spots on bell pepper leaves. This infection weakens the plant, reducing its growth potential and impairing the fruit production, ultimately resulting in a lower yield and quality of peppers.",
    "Pepper,_bell___healthy": "Healthy bell pepper plants have deep green leaves with no visible disease symptoms, discoloration, or damage. The plant is vigorous, supporting the growth of high-quality peppers.",
    "Potato___Early_blight": "Early blight is a fungal disease that forms dark, concentric lesions on potato leaves. This disease can reduce the photosynthetic capacity of the plant, leading to lower yields and potentially affecting the overall quality of the potato crop.",
    "Potato___Late_blight": "Late blight is a fungal disease that causes dark lesions to appear on potato leaves, rapidly spreading through the plant. This disease can cause swift crop loss, significantly impacting both the quantity and quality of the potato harvest.",
    "Potato___healthy": "Healthy potato plants display vibrant green leaves with no visible signs of disease, damage, or pests. The plant is growing vigorously and is on track to produce a healthy, high-quality potato crop.",
    "Raspberry___healthy": "Healthy raspberry plants have green, disease-free leaves that indicate strong growth and vitality. The plant is in optimal condition to produce a bountiful crop of high-quality raspberries.",
    "Soybean___healthy": "Healthy soybean plants show strong, green leaves with no signs of disease or pests. The plant’s growth is unhindered, which supports a healthy soybean crop and maximizes yield potential.",
    "Squash___Powdery_mildew": "Powdery mildew is a fungal infection that creates white, powdery spots on squash leaves. These spots cause leaf deformation and weaken the plant’s ability to grow and produce fruit, ultimately reducing the squash yield.",
    "Strawberry___Leaf_scorch": "Leaf scorch is a condition where strawberry leaves develop brown, crispy tips due to disease or environmental stress. It can be caused by fungal infections or dry conditions, which affect plant growth and fruit production.",
    "Strawberry___healthy": "Healthy strawberry plants have vibrant green leaves, free from disease, damage, or stress. The plant is robust and thriving, promoting strong growth and abundant fruit production.",
    "Tomato___Bacterial_spot": "Bacterial spot is a disease that causes water-soaked, dark lesions to appear on tomato leaves. These lesions eventually lead to leaf drop, weakening the plant and reducing both yield and fruit quality.",
    "Tomato___Early_blight": "Early blight is a fungal disease that causes circular dark spots to form on tomato leaves. As the disease progresses, it impairs photosynthesis, reduces plant vitality, and negatively impacts tomato production.",
    "Tomato___Late_blight": "Late blight is a fungal disease that produces large, dark lesions on tomato leaves. This disease spreads rapidly and can lead to plant death, devastating the tomato crop and causing significant yield loss.",
    "Tomato___Leaf_Mold": "Leaf mold is a fungal infection that causes yellowing and distortion of tomato leaves, weakening the plant’s overall health. This disease reduces the plant’s ability to grow properly and impacts fruit production.",
    "Tomato___Septoria_leaf_spot": "Septoria leaf spot is a fungal disease that causes small, dark lesions on tomato leaves. These lesions reduce the plant’s photosynthetic capacity, leading to weakened plants and diminished fruit production.",
    "Tomato___Spider_mites_Two-spotted_spider_mite": "Spider mites, particularly the two-spotted spider mite, cause yellowing and stippling on tomato leaves. This pest infestation weakens the plant by damaging its leaves, reducing its overall health and productivity.",
    "Tomato___Target_Spot": "Target spot is a fungal disease that causes dark, circular lesions with a yellow border on tomato leaves. It reduces photosynthesis and weakens the plant, leading to decreased yield and potential fruit quality loss.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato yellow leaf curl virus is a viral infection that causes tomato leaves to yellow and curl. This stunts plant growth and dramatically reduces fruit production and quality, severely impacting the crop.",
    "Tomato___Tomato_mosaic_virus": "Tomato mosaic virus is a viral disease that produces a mosaic-like pattern of light and dark green on tomato leaves. It weakens the plant, affecting its overall health, yield, and fruit quality, making the tomatoes unmarketable.",
    "Tomato___healthy": "Healthy tomato plants have lush, green, disease-free leaves. The plant is free from infections, pests, or damage, signaling that it is growing optimally and capable of producing high-quality tomatoes."
}


def load_img(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = img_array.astype('float32') / 255.  
    return img_array

def predict_image(model, image, class_indices, class_descriptions):
    preprocessed_img = load_img(image)
    predictions = model.predict(preprocessed_img)
    prediction_class_index = np.argmax(predictions, axis=1)[0]
    prediction_class_name = class_indices[str(prediction_class_index)]
    prediction_confidence = predictions[0][prediction_class_index] * 100  # Confidence percentage
    prediction_description = class_descriptions.get(prediction_class_name, "Description not available for this class.")
    return prediction_class_name, prediction_confidence, prediction_description

# Set the page config
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="wide")

st.markdown("<h1 style='text-align: center;'>☘️ Plant Disease Classifier ☘️</h1>", unsafe_allow_html=True)


uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)

    col1, col2 = st.columns([1, 3])                                        # 30% and 70% width ratio
    
    # Display the image in the left column (30%)

    with col1:
        resized_img = image.resize((300, 300))  
        st.image(resized_img, caption="Uploaded Image")

    with col2:
        if st.button('Classify'):
            prediction, confidence, description = predict_image(model, image, class_indices, class_descriptions)
            st.success(f"Prediction: **{prediction}**")
            st.write(f"Description: **{description}**")
