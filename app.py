import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw
import rasterio as rio
from pathlib import Path
from rasterio.windows import Window, transform as window_transform
from rasterio.features import shapes
from shapely.geometry import shape
from rasterio.warp import transform_bounds
from ultralytics import FastSAM, SAM
import numpy as np
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


TIF_PATH = Path("data") / "S2C_20250516_latn563lonw0021_T30VWH_ORB080_20250516122950_8bit_clipped.tif"

IMGSZ_DEFAULT = 512
CONF_DEFAULT = 0.3
IOU_DEFAULT = 0.5

if "initialized" not in st.session_state:
    st.session_state["gdf"] = None
    st.session_state["segmentation_run"] = False
    st.session_state["initialized"] = True
    st.session_state["out"] = {}
    st.session_state["points"] = []
    st.session_state["map"] = None
    st.session_state["new_gdf"] = False
    st.session_state["imgsz"] = IMGSZ_DEFAULT
    st.session_state["conf"] = CONF_DEFAULT
    st.session_state["iou"] = IOU_DEFAULT


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
        lon, lat = coord[1], coord[0]
        x, y = transformer.transform(lon, lat)
        # Convert world coordinates (x, y) to pixel coordinates
        col, row = ~profile["transform"] * (x, y)
        col, row = int(round(col)), int(round(row))
        # Check if the pixel lies within the image bounds
        if 0 <= col < profile["width"] and 0 <= row < profile["height"]:
            
            pixel_coords.append([col, row])

    return pixel_coords


def get_window_array(tif_path):

        with rio.open(tif_path) as f:
            profile = f.profile

        win_size = st.session_state["win_size"]

        center_latlon = [st.session_state["out"]["center"]["lat"], st.session_state["out"]["center"]["lng"]]

        width, height = profile["width"], profile["height"]
        print(center_latlon)
        pixel_coords = to_pixel_coordinates([center_latlon], profile)

        if not pixel_coords:
            return None, None, None  # Outside image bounds

        col, row = pixel_coords[0]
        half = win_size // 2
        col_off = max(0, min(col - half, width - win_size))
        row_off = max(0, min(row - half, height - win_size))

        win = Window(
            col_off=col_off, row_off=row_off, width=win_size, height=win_size
        ).round_offsets()

        with rio.open(tif_path) as f:
            win_arr = f.read(window=win)
            win_arr = win_arr.transpose(1,2,0)

        win_trans = window_transform(win, profile["transform"])

        return win_arr, win_trans, profile["crs"]


def create_segmentation_geojson(
    tif_path, coords=None, use_model="FastSAM", imgsz=1024, conf=0.2, iou=0.5
):
    """
    Run segmentation on the TIFF image, convert masks to polygon geometries,
    and save them as a GeoJSON file.

    Parameters:
    tif_path (str): Path to the TIFF image.
    output_geojson_path (str): Output path for the GeoJSON file.
    coords (list, optional): Optional points for interactive segmentation.
    """
    img_win, win_trans, use_crs = get_window_array(tif_path)
    if img_win is None or not img_win.any():
        return None
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
            img_win,
            device="cpu",
            retina_masks=True,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
        )
    else:
        results = model(
            img_win,
            points=coords,
            device="cpu",
            retina_masks=True,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
        )

    res = results[0]
    if res.masks is None:
        return None
    masks = res.masks.data.cpu().numpy()

    transform = win_trans
    crs = use_crs

    polygons = []
    for mask in masks:
        m = mask.astype("uint8")
        for geom_dict, val in shapes(m, transform=transform):
            if val == 1:  # only foreground
                polygons.append(shape(geom_dict))

    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)

    gdf = gdf[gdf.is_valid]

    return gdf


def create_map(center, zoom_val):
    """
    Create a folium map with initial settings and return it.
    """

    m = folium.Map(location=center, zoom_start=zoom_val, max_zoom=15)

    folium.TileLayer(
    tiles="https://tomwilsonsco.github.io/s2_tiles/tiles//{z}/{x}/{y}.png",
    attr="Sentinel-2 10m",
    name="RGB 10m tiles",
    overlay=True,
    control=True,
    no_wrap=True,
    max_zoom=15,
    ).add_to(m)

    return m


def trigger_segmentation():
    st.session_state["gdf"] = None
    st.session_state["segmentation_run"] = True
    st.session_state["no_masks"] = None


def clear_segmentation():
    if "center" in st.session_state["out"]:
        old_center = [
            st.session_state["out"]["center"]["lat"],
            st.session_state["out"]["center"]["lng"],
        ]
        old_zoom = st.session_state["out"]["zoom"]
        st.session_state["center"] = old_center
        st.session_state["zoom"] = old_zoom
    st.session_state["segmentation_done"] = False
    st.session_state["gdf"] = None


def reset_params():
    st.session_state["imgsz"] = IMGSZ_DEFAULT
    st.session_state["conf"] = CONF_DEFAULT
    st.session_state["iou"] = IOU_DEFAULT


def delete_points():
    if "center" in st.session_state["out"]:
        old_center = [
            st.session_state["out"]["center"]["lat"],
            st.session_state["out"]["center"]["lng"],
        ]
        old_zoom = st.session_state["out"]["zoom"]
        st.session_state["center"] = old_center
        st.session_state["zoom"] = old_zoom
    st.session_state["no_masks"] = False
    st.session_state["points"] = []
    st.session_state["map_key"] += 1


# if "center" not in st.session_state:
#     st.session_state["center"] = centroid
if "zoom" not in st.session_state:
    st.session_state["zoom"] = 13

if "map_key" not in st.session_state:
    st.session_state["map_key"] = 0

st.session_state["center"] = [55.967, -2.5199]
st.session_state["zoom"] = 13

m = create_map(st.session_state["center"], st.session_state["zoom"])


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
        key="model_name",
    )
    with st.expander("Segmentation parameters", expanded=False):
        if st.button("Reset parameters"):
            reset_params()
            st.session_state["no_masks"] = None
        st.selectbox(
            "Input image size",
            (128, 256, 512, 1024, 2048),
            index=1,
            key="win_size",
            help="Size to resize the image for segmentation",
        )
        st.slider(
            "Confidence threshold",
            min_value=0.01,
            max_value=0.99,
            value=st.session_state["conf"],
            step=0.01,
            key="conf",
            help="Minimum confidence for mask predictions",
        )
        st.slider(
            "IOU threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state["iou"],
            step=0.01,
            key="iou",
            help="Non-max suppression IOU threshold",
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

if st.session_state["gdf"] is None and st.session_state["segmentation_run"]:
    points_use = st.session_state["points"]
    pixel_coords = None #to_pixel_coordinates(points_use, profile)
    with st.spinner(
        f"Generating {st.session_state["model_name"]} predictions...", show_time=True
    ):
        gdf = create_segmentation_geojson(
            TIF_PATH,
            pixel_coords,
            st.session_state["model_name"],
            imgsz=1024,
            conf=st.session_state["conf"],
            iou=st.session_state["iou"],
        )
    if gdf is None:
        st.session_state["no_masks"] = True
        st.session_state["segmentation_run"] = False
        st.session_state["gdf"] = None
    else:
        st.session_state["no_masks"] = False
        geosjon_file = download_polys(gdf)
        st.session_state["gdf"] = gdf
        st.session_state["new_gdf"] = True
        st.session_state["segmentation_run"] = False


if st.session_state.get("no_masks"):
    st.error(
        "No segmentation masks were generated with those parameters.  \n"
        "Try lowering the confidence or IOU thresholds, or increasing image size."
    )


if st.session_state["gdf"] is not None:
    if "center" in st.session_state["out"] and st.session_state["new_gdf"]:
        old_center = [
            st.session_state["out"]["center"]["lat"],
            st.session_state["out"]["center"]["lng"],
        ]
        old_zoom = st.session_state["out"]["zoom"]
        st.session_state["center"] = old_center
        st.session_state["zoom"] = old_zoom
        st.session_state["new_gdf"] = False
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
