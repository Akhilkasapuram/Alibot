import streamlit as st
import openai
import os
import base64
from dotenv import load_dotenv
from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
import numpy as np

# Load API key from environment
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.title("Interior Design Chatbot (Web Search + Image Understanding)")

user_input = st.text_input("Ask your design question:")
uploaded_img = st.file_uploader("Upload a room photo", type=["jpg", "jpeg", "png"])

# Function to segment room image using SAM
def segment_room_image(image):
    # Convert PIL image to numpy array for SAM
    image_array = np.array(image)
    
    # For now, just return basic info about the image
    # SAM integration will be added step by step
    height, width = image_array.shape[:2]
    return f"Image dimensions: {width}x{height} pixels. Room segmentation will be processed here."

if uploaded_img:
    image = Image.open(uploaded_img)
    st.image(image, caption="Uploaded Room Photo", use_column_width=True)
    
    # Process the image with SAM
    segmentation_info = segment_room_image(image)
    st.write(segmentation_info)

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
