import streamlit as st
import openai
import os
import base64
import requests
from dotenv import load_dotenv

# Load API keys from environment
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")

st.title("🏠 Interior Design Chatbot (Text + Image Generation)")

user_input = st.text_input("Ask your design question:")
uploaded_img = st.file_uploader("Upload a room photo", type=["jpg", "jpeg", "png"])

# Function to generate images using Hugging Face
def generate_design_image(prompt):
    API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    enhanced_prompt = f"interior design, {prompt}, modern, well-lit, professional photography"
    
    payload = {
        "inputs": enhanced_prompt,
        "parameters": {
            "num_inference_steps": 20,
            "guidance_scale": 7.5
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            return response.content
        elif response.status_code == 503:
            return "MODEL_LOADING"  # Model is starting up
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Request failed: {str(e)}")
        return None

if uploaded_img:
    st.image(uploaded_img, caption="Uploaded Room Photo", use_container_width=True)

if user_input:
    messages = [
        {"role": "system", "content": "You are a helpful interior design assistant."}
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
    
    # Get text response from GPT-4o
    with st.spinner("Analyzing and generating design advice..."):
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=300
        )
        
        text_advice = response.choices[0].message.content
        st.write("## 💭 Design Advice:")
        st.write(text_advice)
    
    # Generate design image
    if HF_TOKEN:
        with st.spinner("Generating design visualization... (may take 1-2 minutes first time)"):
            image_result = generate_design_image(user_input)
            
            if image_result == "MODEL_LOADING":
                st.warning("🔄 Hugging Face model is starting up. Please wait 1-2 minutes and try again.")
            elif image_result:
                st.write("## 🎨 Generated Design:")
                st.image(image_result, caption="AI Generated Design", use_container_width=True)
            else:
                st.warning("⚠️ Image generation temporarily unavailable. The text advice above is still accurate!")
    else:
        st.info("Add HUGGINGFACE_API_KEY to enable image generation.")
