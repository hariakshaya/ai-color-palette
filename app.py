import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import webcolors

def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def get_palette(image, n_colors=5):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (200, 200))
    img = img.reshape((-1, 3))

    kmeans = KMeans(n_clusters=n_colors, random_state=0)
    kmeans.fit(img)

    colors = kmeans.cluster_centers_.astype(int)
    return colors

st.title("🎨 AI Color Palette Generator")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    colors = get_palette(image, n_colors=5)

    st.subheader("Extracted Colors")
    cols = st.columns(5)
    for i, col in enumerate(cols):
        rgb = tuple(colors[i])
        hex_color = '#%02x%02x%02x' % rgb
        name = closest_color(rgb)
        col.markdown(
            f"<div style='background-color:{hex_color};width:100%;height:100px;border-radius:10px'></div><p style='text-align:center;'>{name}</p>",
            unsafe_allow_html=True,
        )
