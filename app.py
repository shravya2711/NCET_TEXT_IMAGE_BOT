import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

st.set_page_config(page_title="Text to Image Generator")
st.title("🖼️ Text → Image Generator (Local Model)")

@st.cache_resource
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

prompt = st.text_input("Enter your prompt:")

if st.button("Generate Image"):
    if prompt:
        pipe = load_model()
        with st.spinner("Generating..."):
            image = pipe(prompt).images[0]
            st.image(image, caption="Generated Image", use_column_width=True)

            # Download option
            import io
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            st.download_button(
                label="Download Image",
                data=buf.getvalue(),
                file_name="generated.png",
                mime="image/png"
            )
    else:
        st.warning("Please enter a prompt!")
