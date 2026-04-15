import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

st.title("🖼️ Text to Image Generator")

@st.cache_resource
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    
    return pipe

prompt = st.text_input("Enter your prompt:")

if st.button("Generate Image"):
    if prompt:
        try:
            pipe = load_model()
            with st.spinner("Generating image... (may take time ⏳)"):
                image = pipe(prompt, num_inference_steps=20).images[0]
                st.image(image)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Enter a prompt!")
