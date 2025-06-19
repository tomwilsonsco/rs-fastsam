import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw
import rasterio as rio
from PIL import Image
from pathlib import Path
from shapely.geometry import box
from rasterio.transform import xy
from ultralytics import FastSAM, SAM
import numpy as np
import cv2
from shapely.geometry import Polygon, mapping
import geopandas as gpd
from pyproj import Transformer

# → Page config
st.set_page_config(
    page_title="🛰️ Satellite Image Segmentation",
    layout="wide",
    initial_sidebar_state="expanded",
)
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Satellite Image Segmentation")

with st.expander("❓ How to use"):
    st.markdown(
        """
    1. Use the draw marker icon (top left of the map) to place points on image features you wish to segment.  
    **Alternatively**, Zoom in on the image to the extent you wish to segment.
    2. Choose **FastSAM** or **MobileSAM** from the sidebar.  
    3. Wait for the segmentation predictions to draw.  
    4. (Optional) Download the predictions as a GeoJSON file.  
    5. Delete points or predictions and run again as needed.
    """
    )


TIF_PATH = Path("data") / "rgb_fast_sam_test.tif"

if "initialized" not in st.session_state:
    st.session_state["show_segmentation"] = False
    st.session_state["segmentation_done"] = False
    st.session_state["initialized"] = True
    st.session_state["out"] = {}
    st.session_state["points"] = []
    st.session_state["map"] = None


def to_pixel_coordinates(coordinates, profile):
    """Convert list of [lon, lat] coordinates to pixel coordinates.

    For each [lon, lat] coordinate:
    - Convert from EPSG:4326 (WGS84) to the image CRS (profile['crs']).
    - Convert world coordinates to pixel coordinates using the inverted affine transform.
    - Exclude any points that fall outside the image bounds defined by profile['width'] and profile['height'].

    Returns a list of [col, row] pixel coordinates.
    """
    transformer = Transformer.from_crs("EPSG:4326", profile["crs"], always_xy=True)

    if not coordinates:
        return None

    pixel_coords = []
    for coord in coordinates:
        lon, lat = coord[0], coord[1]
        x, y = transformer.transform(lon, lat)
        # Convert world coordinates (x, y) to pixel coordinates
        col, row = ~profile["transform"] * (x, y)
        col, row = int(round(col)), int(round(row))

        # Check if the pixel lies within the image bounds
        if 0 <= col < profile["width"] and 0 <= row < profile["height"]:
            pixel_coords.append([col, row])

    return pixel_coords


@st.cache_data
def create_png(tif_path):
    with rio.open(tif_path) as src:
        img = src.read()  # shape: (3, H, W)
        profile = src.profile  # Get metadata
        bounds = src.bounds  # Get bounds
        # Project bounds to WGS84 (EPSG:4326) if needed

    bbox = box(*src.bounds)

    # Create a GeoDataFrame for reprojection
    gdf_bbox = gpd.GeoDataFrame(geometry=[bbox], crs=src.crs)
    bbox_wgs = gdf_bbox.to_crs("EPSG:4326").geometry.iloc[0]

    # Extract bottom left and top right coordinates
    minx, miny, maxx, maxy = bbox_wgs.bounds

    # Calculate centroid of the bounds rectangle as (lat, lon)
    lat_centroid = (miny + maxy) / 2
    lon_centroid = (minx + maxx) / 2
    centroid = [lat_centroid, lon_centroid]

    # Convert to [[lat_min, lon_min], [lat_max, lon_max]]
    image_bounds = [
        [miny, minx],  # [lat_min, lon_min]
        [maxy, maxx],  # [lat_max, lon_max]
    ]

    # Save as temporary PNG if not already
    tmp_path = Path("/tmp/overlay.png")
    if not tmp_path.exists():

        img_rgb = img.transpose(1, 2, 0)
        Image.fromarray(img_rgb).save(tmp_path)
    return profile, image_bounds, centroid, tmp_path


def create_segmentation_geojson(
    tif_path, profile, coords=None, use_model="FastSAM", imgsz=1024, conf=0.2, iou=0.5
):
    """
    Run segmentation on the TIFF image, convert masks to polygon geometries,
    and save them as a GeoJSON file.

    Parameters:
    tif_path (str): Path to the TIFF image.
    output_geojson_path (str): Output path for the GeoJSON file.
    coords (list, optional): Optional points for interactive segmentation.
    """

    # Load model and run inference
    if use_model == "FastSAM":
        model = FastSAM("FastSAM-x.pt")
    elif use_model == "MobileSAM":
        model = SAM("mobile_sam.pt")
    elif use_model == "SAM2-t":
        model = SAM("sam2_t.pt")
    else:
        raise ValueError(
            f"Invalid model name: {use_model}. Expected 'FastSAM', 'MobileSAM', or 'SAM2-t'."
        )

    if coords is None:
        results = model(
            tif_path,
            device="cpu",
            retina_masks=True,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
        )
    else:
        results = model(
            tif_path,
            points=coords,
            device="cpu",
            retina_masks=True,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
        )

    res = results[0]
    masks = res.masks.data.cpu().numpy()

    transform = profile["transform"]
    crs = profile["crs"]

    polygons = []
    for mask in masks:
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            if len(cnt) >= 3:  # Valid polygon
                # Convert pixel (row, col) to geographic (x, y)
                pixel_coords = cnt.squeeze()
                if pixel_coords.ndim != 2:  # Skip degenerate contours
                    continue

                geo_coords = [
                    xy(transform, y, x) for x, y in pixel_coords
                ]  # Note x,y switch
                polygon = Polygon(geo_coords)
                if polygon.is_valid and not polygon.is_empty:
                    polygons.append(polygon)

    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)

    gdf = gdf[gdf.is_valid]

    return gdf


def create_map(center, zoom_val, img_path):
    """
    Create a folium map with initial settings and return it.
    """

    m = folium.Map(location=center, zoom_start=zoom_val)

    folium.raster_layers.ImageOverlay(
        str(img_path), bounds=image_bounds, opacity=1, name="RGB image"
    ).add_to(m)

    return m


def trigger_segmentation():
    st.session_state["show_segmentation"] = True
    st.session_state["segmentation_done"] = False


def clear_segmentation():
    st.session_state["show_segmentation"] = False
    st.session_state["segmentation_done"] = False
    st.session_state["gdf"] = None


def delete_points():
    if "center" in st.session_state["out"]:
        new_center = [
            st.session_state["out"]["center"]["lat"],
            st.session_state["out"]["center"]["lng"],
        ]
        st.session_state["center"] = new_center
        st.session_state["zoom"] = st.session_state["out"]["zoom"]
    st.session_state["points"] = []
    st.session_state["map_key"] += 1


profile, image_bounds, centroid, tmp_path = create_png(TIF_PATH)

if "center" not in st.session_state:
    st.session_state["center"] = centroid
if "zoom" not in st.session_state:
    st.session_state["zoom"] = 13

if "map_key" not in st.session_state:
    st.session_state["map_key"] = 0

m = create_map(st.session_state["center"], st.session_state["zoom"], tmp_path)


draw = Draw(
    draw_options={
        "polyline": False,
        "polygon": False,
        "circle": False,
        "rectangle": False,
        "circlemarker": False,
        "marker": True,  # Enable only point (marker)
    },
    edit_options={"edit": False, "remove": False},
)
draw.add_to(m)

fg = folium.FeatureGroup(name="Markers")

if "out" in st.session_state:
    if "all_drawings" in st.session_state["out"]:
        points = st.session_state["out"]["all_drawings"]

if st.session_state["points"]:
    for point in st.session_state["points"]:
        fg.add_child(
            folium.Marker(
                location=[point[1], point[0]],
            )
        )


def download_polys(gdf):
    return gdf.to_file("preds.geojson", driver="GeoJSON")


with st.sidebar:
    st.header("Controls")
    st.selectbox(
        label="Model to use",
        options=("FastSAM", "MobileSAM", "SAM2-t"),
        placeholder="FastSAM",
        key="model_name"
    )
    st.button(
        "Create Predictions",
        on_click=trigger_segmentation,
        help="Run the specified segmentation model for markers or current extent",
    )
    st.button(
        "Delete Points",
        on_click=delete_points,
        help="Delete all points you have drawn.",
    )
    st.button(
        "Delete Predictions",
        on_click=clear_segmentation,
        help="Clear segmentation prediction polygons.",
    )
    if st.session_state.get("gdf") is not None:
        geojson_str = st.session_state["gdf"].to_json()
        st.sidebar.download_button(
            "Download predictions",
            data=geojson_str,
            file_name="seg_preds.geojson",
            mime="application/geo+json",
        )

if st.session_state["show_segmentation"] and not st.session_state["segmentation_done"]:
    points_use = st.session_state["points"]
    pixel_coords = to_pixel_coordinates(points_use, profile)
    with st.spinner(
        f"Generating {st.session_state["model_name"]} predictions...", show_time=True
    ):
        gdf = create_segmentation_geojson(
            TIF_PATH, profile, pixel_coords, st.session_state["model_name"]
        )
    geosjon_file = download_polys(gdf)
    st.session_state["gdf"] = gdf
    st.session_state["segmentation_done"] = True

    if "center" in st.session_state["out"]:
        new_center = [
            st.session_state["out"]["center"]["lat"],
            st.session_state["out"]["center"]["lng"],
        ]
        st.session_state["center"] = new_center
        st.session_state["zoom"] = st.session_state["out"]["zoom"]

if st.session_state["show_segmentation"]:
    # Add segmentation polygons to the map
    folium.GeoJson(
        st.session_state["gdf"],
        name="Segmentation Polygons",
        color="red",
        fill=False,
    ).add_to(m)


folium.LayerControl().add_to(m)

out = st_folium(
    m,
    center=st.session_state["center"],
    zoom=st.session_state["zoom"],
    feature_group_to_add=fg,
    key=f"out_{st.session_state["map_key"]}",
    height=600,
    width=900,
)

st.session_state["out"] = out

current_points = out["all_drawings"]
stored_points = st.session_state["points"]

if current_points:
    new_points = [p["geometry"]["coordinates"] for p in current_points] + stored_points
    new_points = list(set(tuple(pt) for pt in new_points))
    new_points = [list(pt) for pt in new_points]
    st.session_state["points"] = new_points
