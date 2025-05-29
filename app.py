import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw
import rasterio
from PIL import Image
import os
from pyproj import Transformer

clicked = st.button("Get pixel coordinates")

# Load image and bounds
tif_path = "rgb_fast_sam_test.tif"
with rasterio.open(tif_path) as src:
    img = src.read()  # shape: (3, H, W)
    bounds = src.bounds
    transform = src.transform
    width = src.width
    height = src.height
    profile = src.profile

# Save as temporary PNG if not already
tmp_path = os.path.join("/tmp", "overlay.png")
if not os.path.exists(tmp_path):
    # Convert to (H, W, 3) and save
    img_rgb = img.transpose(1, 2, 0)
    Image.fromarray(img_rgb).save(tmp_path)

# Define map and bounds
image_bounds = [[56.00887809, -2.79804176], [56.03862098, -2.72650399]]
center = [56.02437988, -2.76394275]

# Build Folium map
m = folium.Map(location=center, zoom_start=13)
folium.raster_layers.ImageOverlay(tmp_path, bounds=image_bounds, opacity=1).add_to(m)

draw = Draw(
    draw_options={
        "polyline": False,
        "polygon": False,
        "circle": False,
        "rectangle": False,
        "circlemarker": False,
        "marker": True  # Enable only point (marker)
    },
    edit_options={"edit": True}
)
draw.add_to(m)

# Streamlit app
st.title("Draw points on the image")


output = st_folium(m, width=700, height=500, pixelated=True)

with st.sidebar:
    st.header("Captured Points")
    if clicked:
        points = output.get("all_drawings")
        coords = [p["geometry"]["coordinates"] for p in points]
        st.write(coords)