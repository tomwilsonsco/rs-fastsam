import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw
from pathlib import Path
import geopandas as gpd

# internal import
from src.rssam.predict import create_segmentation_geojson

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
    1. Zoom in on the area you wish to segment. You need to zoom to level 14 or 15 to generate predictions.
    2. (Optional) Use the draw marker / rectangle tools on the top left to place features on the map to target specific areas.
    3. Choose **FastSAM**, **MobileSAM**, **SAM2-t** from the sidebar.  
    4. Click Create Predictions and wait for the segmentation predictions to draw.  
    5. (Optional) Download the predictions as a GeoJSON file.  
    6. Delete drawings or predictions and run again as needed.

    ##### Tips:
    1. FastSAM is quicker, but MobileSAM can produce better quality segmentations for full extent.
    2. Rectangle drawings are for targetting specific features like a field or lake, not for segmenting all features in a smaller extent.
    2. When drawing features, only features in the current display extent are used for the segmentations.
    """
    )


TIF_PATH = (
    Path("data")
    / "S2C_20250516_latn563lonw0021_T30VWH_ORB080_20250516122950_8bit_clipped_sharp_rs.tif"  # "nir_8bit_sharpened.tif"#"S2C_20250516_latn563lonw0021_T30VWH_ORB080_20250516122950_8bit_clipped.tif"
)

IMGSZ_DEFAULT = 512
CONF_DEFAULT = 0.3
IOU_DEFAULT = 0.5

if "initialized" not in st.session_state:
    st.session_state["gdf"] = gpd.GeoDataFrame([])
    st.session_state["segmentation_run"] = False
    st.session_state["initialized"] = True
    st.session_state["points"] = []
    st.session_state["labels"] = []
    st.session_state["rectangles"] = []
    st.session_state["map"] = None
    st.session_state["predict_disabled"] = False
    st.session_state["imgsz"] = IMGSZ_DEFAULT
    st.session_state["conf"] = CONF_DEFAULT
    st.session_state["iou"] = IOU_DEFAULT


def create_map(center, zoom_val):
    """
    Create a folium map with initial settings and return it.
    """

    m = folium.Map(location=center, zoom_start=zoom_val, max_zoom=15)

    folium.TileLayer(
        tiles="https://tomwilsonsco.github.io/s2_tiles/tiles//{z}/{x}/{y}.png",
        attr="Sentinel-2 10m",
        name="Sentinel 2 RGB 10m",
        overlay=True,
        control=True,
        no_wrap=True,
        max_zoom=15,
    ).add_to(m)

    return m


def trigger_segmentation():
    st.session_state["gdf"] = gpd.GeoDataFrame([])
    st.session_state["segmentation_run"] = True
    st.session_state["no_masks"] = None
    old_center = [
        st.session_state["out"]["center"]["lat"] + 0.0001,
        st.session_state["out"]["center"]["lng"] + 0.0001,
    ]
    old_zoom = st.session_state["out"]["zoom"]
    st.session_state["center"] = old_center
    st.session_state["zoom"] = old_zoom
    ## After running this once st.session_state["out"]["all_drawings"]
    ## becomes None. If delete all features with draw edit tools becomes []
    # Therefore current_drawings is Not None current works to achieve this
    current_drawings = st.session_state["out"]["all_drawings"]
    if current_drawings is not None:
        st.session_state["draw_features"] = current_drawings
    #     all_points = [
    #         p["geometry"]["coordinates"]
    #         for p in current_drawings
    #         if p["geometry"]["type"] == "Point"
    #     ]
    #     st.session_state["points"] = all_points
    #     all_rectangles = [
    #         p["geometry"]["coordinates"]
    #         for p in current_drawings
    #         if p["geometry"]["type"] == "Polygon"
    #     ]
    #     st.session_state["rectangles"] = all_rectangles


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
    st.session_state["gdf"] = gpd.GeoDataFrame([])


def reset_params():
    st.session_state["imgsz"] = IMGSZ_DEFAULT
    st.session_state["conf"] = CONF_DEFAULT
    st.session_state["iou"] = IOU_DEFAULT


def delete_features():
    if st.session_state["out"].get("center", False):
        clean_center = [
            st.session_state["out"]["center"]["lat"] + 0.0001,
            st.session_state["out"]["center"]["lng"],
        ]
        clean_zoom = st.session_state["out"]["zoom"]
        st.session_state["center"] = clean_center
        st.session_state["zoom"] = clean_zoom
    st.session_state["no_masks"] = False
    st.session_state["points"] = []
    st.session_state["rectangles"] = []
    st.session_state["labels"] = []


def download_polys(gdf):
    return gdf.to_file("data/preds.geojson", driver="GeoJSON")


if st.session_state.get("out", False):
    current_zoom = st.session_state["out"]["zoom"]
    if current_zoom < 14:
        st.session_state["predict_disabled"] = True
    else:
        st.session_state["predict_disabled"] = False


with st.sidebar:
    st.selectbox(
        label="Model to use",
        options=("FastSAM", "MobileSAM", "SAM2-t"),
        placeholder="FastSAM",
        key="model_name",
    )
    with st.expander("Prediction parameters", expanded=False):
        if st.button("Reset parameters"):
            reset_params()
            st.session_state["no_masks"] = None
        st.selectbox(
            "Input image size",
            (256, 512, 640, 768, 1024),
            index=2,
            key="imgsz",
            help="Only applied for FastSAM. Size to resize the image for segmentation.",
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
        disabled=st.session_state["predict_disabled"],
        help="Run the specified segmentation model. Must be zoomed in to 14 or 15",
    )
    if not st.session_state.get("gdf").empty:
        geojson_str = st.session_state["gdf"].to_json()
        st.sidebar.download_button(
            "Download predictions",
            data=geojson_str,
            file_name="seg_preds.geojson",
            mime="application/geo+json",
        )

    if st.session_state.get("out", False):
        props = st.session_state["out"]
        zoom = props["zoom"]
        lat = props["center"]["lat"]
        lng = props["center"]["lng"]
        st.caption(f"Zoom Level: {zoom}")
        st.caption(f"Centre: {lat:.3f}, {lng:.3f}")


if st.session_state.get("out", False):
    if (
        st.session_state["gdf"].empty
        and st.session_state["segmentation_run"]
        and current_zoom >= 14
    ):
        if st.session_state.get("draw_features", False):
            mapped_features = st.session_state["draw_features"]
        else:
            mapped_features = []
        with st.spinner(
            f"Generating {st.session_state["model_name"]} predictions...",
            show_time=True,
        ):
            gdf = create_segmentation_geojson(
                TIF_PATH,
                mapped_features,
                None,
                st.session_state["model_name"],
                imgsz=st.session_state["imgsz"],
                conf=st.session_state["conf"],
                iou=st.session_state["iou"],
            )
        if gdf is None:
            st.session_state["no_masks"] = True
            st.session_state["segmentation_run"] = False
            st.session_state["gdf"] = gpd.GeoDataFrame([])
        else:
            st.session_state["no_masks"] = False
            geosjon_file = download_polys(gdf)
            st.session_state["gdf"] = gdf
            st.session_state["segmentation_run"] = False


if st.session_state.get("no_masks"):
    st.error(
        "No segmentation masks were generated with those parameters.  \n"
        "Try lowering the confidence or IOU thresholds, or increasing image size."
    )

if "zoom" not in st.session_state:
    st.session_state["zoom"] = 14

if "center" not in st.session_state:
    st.session_state["center"] = [56.020, -2.754]


# PREDICTIONS LAYER
pred_fg = folium.FeatureGroup(name="Prediction Polygons", control=True)

if not st.session_state["gdf"].empty:
    pred_fg.add_child(
        folium.GeoJson(
            st.session_state["gdf"],
            name="Prediction Polygons",
            style_function=lambda feature: {
                "fillColor": "red",
                "color": "red",
                "fillOpacity": 0,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["area_disp"], labels=True, aliases=[""], localize=True
            ),
            overlay=True,
            control=True,
        )
    )


# POINTS LAYER

fg = folium.FeatureGroup(name="Drawing features", control=True)

if st.session_state.get("draw_features", False):
    for feature in st.session_state["draw_features"]:
        # Use folium.GeoJson for all types of features for simplicity and consistency.
        # The Draw plugin will correctly interpret these GeoJSON features.
        folium.GeoJson(feature).add_to(fg)


# for point in st.session_state.get("points", None):
#     fg.add_child(
#         folium.Marker(location=[point[1], point[0]], name="point markers", control=True)
#     )

# # RECTANGLES LAYER
# for rect in st.session_state.get("rectangles", None):
#     rect = rect[0]
#     rect = [[lat, lon] for lon, lat in rect]
#     fg.add_child(
#         folium.Polygon(
#             locations=rect,
#             name="rectangle markers",
#             fill=True,
#             fill_opacity=0.2,
#             control=True,
#         )
#     )

m = create_map(st.session_state["center"], st.session_state["zoom"])

# Add layers to map
m.add_child(fg)
m.add_child(pred_fg)


draw = Draw(
    feature_group=fg,
    draw_options={
        "polyline": False,
        "polygon": False,
        "circle": False,
        "rectangle": True,
        "circlemarker": False,
        "marker": True,
    },
    edit_options={"edit": True, "remove": True},
)
draw.add_to(m)

folium.LayerControl().add_to(m)

if st.session_state.get("out", False):
    print(st.session_state["out"]["all_drawings"])

out = st_folium(
    m,
    center=st.session_state["center"],
    zoom=st.session_state["zoom"],
    feature_group_to_add=[fg, pred_fg],
    key="out",
    height=600,
    width=900,
    pixelated=True,
)
