import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import json
import os
from io import BytesIO

# --- Import the function from Pixelate.py ---
try:
    from Pixelate import pixelate
except ImportError as e:
    st.error(f"Could not find Pixelate.py: {e}")
    st.stop()

# --- Helper: Load Palettes ---
@st.cache_data
def load_palettes(filepath="palettes.json"):
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r") as f:
        return json.load(f)

# --- UI Configuration ---
st.set_page_config(page_title="Pixelator", layout="wide")
st.title("Pixelator")

palettes_data = load_palettes()

# Initialize session state to store the pixelated result
if "processed_img" not in st.session_state:
    st.session_state.processed_img = None

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Pixelisation Settings")
    selected_palette = st.selectbox("Palette", options=list(palettes_data.keys()))
    
    block_size = st.number_input("Block Size", min_value=1, max_value=128, value=16)
    method = st.radio("Method", ["mode", "mean"], horizontal=True)
    
    # Conditional UI: Blend Strength only appears if Apply Edges is True
    apply_edges = st.toggle("Apply Edges", value=False)
    
    # Logic for conditional slider
    blend_strength = 0.5  # Default value if hidden
    if apply_edges:
        blend_strength = st.slider("Edge Strength", 0.0, 0.9, 0.5, 0.1)

    # The Process Button
    st.divider()
    submit_button = st.button("Run", use_container_width=True, type="primary")
    
    st.divider()
    
    st.header("Filters")
    # These affect the image instantly after it has been pixelated
    brightness = st.slider("Brightness", 0.0, 2.0, 1.0, 0.05)
    saturation = st.slider("Saturation", 0.0, 2.0, 1.0, 0.05)

# --- Main App Logic ---
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original")
        st.image(img, use_container_width=True)
        
    with col2:
        st.subheader("Processed")
        
        # Action: Run Heavy Pixelation Logic
        if submit_button:
            with st.spinner("Processing pixelation..."):
                img_array = np.array(img)
                # Pass parameters to your Pixelate.py function
                result_arr = pixelate(
                    img_array, 
                    selected_palette, 
                    block_size, 
                    method, 
                    apply_edges, 
                    blend_strength
                )
                # Store in session state so we don't re-run this on every slider move
                st.session_state.processed_img = Image.fromarray(result_arr.astype(np.uint8))

        # Render the result if it exists
        if st.session_state.processed_img is not None:
            # Start with the stored pixelated image
            enhanced_img = st.session_state.processed_img
            
            # Apply Real-time Enhancements using PIL
            if brightness != 1.0:
                enhanced_img = ImageEnhance.Brightness(enhanced_img).enhance(brightness)
            
            if saturation != 1.0:
                enhanced_img = ImageEnhance.Color(enhanced_img).enhance(saturation)
            
            st.image(enhanced_img, use_container_width=True)
            
            # Prepare download
            buf = BytesIO()
            enhanced_img.save(buf, format="PNG")
            st.download_button(
                label="Download Pixelated Image",
                data=buf.getvalue(),
                file_name=f"pixelated_{selected_palette}.png",
                mime="image/png",
                use_container_width=True
            )
        else:
            st.info("Click 'Process Pixelation' to generate the base image.")
else:
    st.session_state.processed_img = None
    st.info("Please upload an image.")