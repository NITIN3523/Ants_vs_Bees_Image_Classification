import streamlit as st
from utils import set_background
from PIL import Image
import torch
from classifier import classify
from io import BytesIO
import base64
from Model import CNN

st.set_page_config(
    page_title='Ants and Bees Classification',
    layout='centered'
)

set_background('utils/bg.jpg')

st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 50px;
        color: #f0f0f0;
        font-weight: bold;        
        margin-top: -65px;
    }
    .header {
        text-align: center;
        font-size: 35px;
        color: #87cefa;
        margin-top: -15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">Ants and Bees Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="header">Upload an image to classify it as an Ant and Bee</div>', unsafe_allow_html=True)

file = st.file_uploader('',type = ['jpg','jpeg','png','jfif'])


model = torch.load('Model/aunt_vs_bee.pth',map_location=torch.device('cpu'))

class_names = {0:'Bee',1:'Ant'}

if file is not None:
    
    image = Image.open(file).convert('RGB')
    
    prediction, score = classify(image, model, class_names)
    
    bufferd = BytesIO()
    image.save(bufferd, format='PNG')
    img_base64 = base64.b64encode(bufferd.getvalue()).decode()

    # Display classification results with reduced gap and no extra space
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; align-items: center;">        
        <img src="data:image/png;base64,{img_base64}" style="width: 350px; height: 320px; object-fit: cover;">
        <div style="font-size:40px; font-weight:bold; margin-left: 20px; color:#FFFFFF;">
            <p> <strong> Result: {prediction} </strong> </p>
            <p style="margin-top:-10px;"> <strong> Score: {score}% </strong> </p>
        </div>        
        </div>
        """,
        unsafe_allow_html=True
    )