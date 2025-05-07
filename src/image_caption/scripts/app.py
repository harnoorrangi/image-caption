import streamlit as st
from PIL import Image

from image_caption.scripts.predict_model import load_model, predict_caption
from image_caption.scripts.utils import load_hydra_config


@st.cache_resource
def load_model_cached():
    cfg = load_hydra_config("vit_gpt2")
    model_name = cfg.train_params.hugging_face_model_id
    return load_model(model_name)


st.title("Image Captioning using Vit and Gpt2")
st.write("Upload an image, and the model will generate a caption for it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load model and generate caption
    with st.spinner("Generating caption..."):
        model, image_processor, tokenizer, device = load_model_cached()
        caption = predict_caption(model, image_processor, tokenizer, device, uploaded_file)

    # Display the caption
    st.success("Caption generated!")
    st.write(f"**Caption:** {caption}")
