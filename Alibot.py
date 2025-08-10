import streamlit as st
import openai
import os
import base64
from dotenv import load_dotenv

# Load API key from environment
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.title("Interior Design Chatbot (Web Search + Image Understanding)")

user_input = st.text_input("Ask your design question:")

uploaded_img = st.file_uploader("Upload a room photo", type=["jpg", "jpeg", "png"])

if uploaded_img:
    st.image(uploaded_img, caption="Uploaded Room Photo", use_column_width=True)

if user_input:
    messages = [
        {"role": "system", "content": "You are a helpful interior-design assistant."}
    ]

    if uploaded_img:
        img_bytes = uploaded_img.read()
        img_type = uploaded_img.type or "image/png"  # "image/jpeg" or "image/png"
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
        model="gpt-4o",            # GPT-4o: supports both images and web search
        messages=messages,
        max_tokens=300
    )
    st.write(response.choices[0].message.content)
