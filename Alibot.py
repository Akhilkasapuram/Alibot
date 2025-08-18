import streamlit as st
import openai
import os
import base64
from dotenv import load_dotenv
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch

# Load API key from environment
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.title("Interior Design Chatbot (Web Search + Image Understanding)")

user_input = st.text_input("Ask your design question:")
uploaded_img = st.file_uploader("Upload a room photo", type=["jpg", "jpeg", "png"])

# BLIP-2 for scene description
@st.cache_resource
def load_blip2():
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
    return processor, model

def get_scene_description(image):
    processor, model = load_blip2()
    inputs = processor(image, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    description = processor.decode(outputs[0], skip_special_tokens=True)
    return description

if uploaded_img:
    image = Image.open(uploaded_img)
    st.image(image, caption="Uploaded Room Photo", use_column_width=True)
    
    # Get scene description
    with st.spinner("Analyzing room layout and camera angle..."):
        scene_desc = get_scene_description(image)
        st.write(f"**Scene Analysis:** {scene_desc}")

if user_input:
    messages = [
        {"role": "system", "content": "You are a helpful interior-design assistant."}
    ]
    if uploaded_img:
        img_bytes = uploaded_img.read()
        img_type = uploaded_img.type or "image/png"
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_input},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{img_type};base64,{img_b64}"}
                }
            ]
        })
    else:
        messages.append({"role": "user", "content": user_input})
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=300
    )
    st.write(response.choices[0].message.content)
