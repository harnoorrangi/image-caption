import streamlit as st
from PIL import Image

from image_caption.scripts.predict_model import load_model, predict_caption
from image_caption.scripts.utils import load_hydra_config

# Page config for a fancy look
st.set_page_config(
    page_title="Image Captioning using ViT and GPT-2",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .caption-box {
        background-color: #f0f2f6;
        border-left: 6px solid #1f77b4;
        padding: 15px;
        border-radius: 4px;
        margin-top: 20px;
        color: #333333;
    }
    .caption-box h3 {
        color: #1f77b4;
    }
    .caption-box p {
        color: #333333;
        font-size: 1.1rem;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 10px;
        padding: 0.6em 1.5em;
        font-size: 1rem;
    }
    .stButton>button:hover {
        background-color: #185a9d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar with instructions
with st.sidebar:
    st.header("ℹ️ About this App")
    st.write(
        "Generate human-like captions for your images using a pretrained ViT + GPT-2 model hosted on Hugging Face."
    )
    st.divider()
    st.info("Upload an image and click **Generate Caption** to see the result.")
    st.divider()
    st.write("---")
    st.write("Made with ❤️ by Harnoor Rangi")


@st.cache_resource
def load_model_cached():
    # Load config and model once
    cfg = load_hydra_config("vit_gpt2")
    model_name = cfg.train_params.hugging_face_model_id
    return load_model(model_name)


# Main interface
st.markdown("# 🖼️ Image Captioning using ViT and GPT-2")
st.write("Drop an image below, then click **Generate Caption** to get a witty description.")

uploaded_file = st.file_uploader("Select an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Center the image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Your Uploaded Image", use_column_width=True)

    # Generate button
    generate = st.button("Generate Caption")

    if generate:
        model, image_processor, tokenizer, device = load_model_cached()
        with st.spinner("🤖 Generating caption..."):
            caption = predict_caption(model, image_processor, tokenizer, device, uploaded_file)

        # Display caption in styled box
        st.markdown(
            f'<div class="caption-box"><h3>✨ Caption:</h3><p>{caption}</p></div>',
            unsafe_allow_html=True,
        )
        st.balloons()
