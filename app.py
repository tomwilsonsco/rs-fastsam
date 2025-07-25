import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw
from pathlib import Path
import geopandas as gpd
from ultralytics import FastSAM, SAM
import pickle
import yaml
from pathlib import Path

# internal class import
from src.rssam import RasterSegmentor, classify_polygons


# Load configuration
@st.cache_data
def load_config():
    """Load configuration from YAML file."""
    config_path = Path("config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Load config at startup
config = load_config()

# Page config using config values
st.set_page_config(
    page_title=config["app"]["title"],
    layout=config["app"]["page_config"]["layout"],
    initial_sidebar_state=config["app"]["page_config"]["initial_sidebar_state"],
)

# Source images from config
SEGMENT_TIF_PATH = Path(config["data"]["segment_tif_path"])
CLASSIFY_TIF_PATH = Path(config["data"]["classify_tif_path"])

# Default values from config
UPSCALE_DEFAULT = config["defaults"]["upscale"]
SHARP_DEFAULT = config["defaults"]["sharp"]
IMGSZ_DEFAULT = config["defaults"]["imgsz"]
CONF_DEFAULT = config["defaults"]["conf"]
IOU_DEFAULT = config["defaults"]["iou"]

# Page config
st.set_page_config(
    page_title="🛰️ Sentinel-2 Image Segmentation",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.html("static/style.css")

st.title(" 🛰️Segment Sentinel-2 Imagery")

# Help guide
with st.expander("❓ How to use"):
    st.markdown(
        """
    See [the project's repository](https://github.com/tomwilsonsco/rs-fastsam) for full details. 
    Image date 16 May 2025 ([Sentinel-2](https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-2) 
    © European Space Agency (ESA) / Copernicus Programme).
    
    ##### Instructions for use:
    1. Zoom in on the area you wish to segment. You need to zoom to level 14 or 15 to generate segment predictions.
    2. (Optional) Use the draw marker / rectangle tools on the top left to place features on the map to target specific areas.
    3. Choose **FastSAM**, **MobileSAM**, **SAM2-t** from the sidebar.  
    4. Click Run Segmentation and wait for the segmentation predictions to draw. 
    5. Click Classify Segments to predict segment land use. 
    6. (Optional) Download the predictions as a GeoJSON file.  
    7. Delete or adjust drawings and pan to a new extent and create more segmentations as needed.

    ##### Tips:
    1. For extent predictions FastSAM is quicker, but MobileSAM or SAM2-t can produce better quality segmentations.
    2. For individual feature predictions using markers, all models are typically fast.
    3. Rectangle drawings are for segmenting all features in a smaller extent.
    4. When drawing features, only features in the current display extent are used for the segmentations.
    5. The classification of segment land use is very experimental. No ground truth data has been used.

    ##### Credits:
    Imagery: ESA Sentinel-2 | Models: 
    [SAM](https://segment-anything.com/), 
    [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM), 
    [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), 
    [SAM2](https://ai.meta.com/sam2/) | 
    Implementation: [Ultralytics](https://www.ultralytics.com/).  
    This app is for demonstration and research purposes only.
    """
    )


# Initial session states
if "initialized" not in st.session_state:
    st.session_state["gdf"] = gpd.GeoDataFrame([])
    st.session_state["segmentation_run"] = False
    st.session_state["initialized"] = True
    st.session_state["predict_disabled"] = False
    st.session_state["upscale"] = UPSCALE_DEFAULT
    st.session_state["sharp"] = SHARP_DEFAULT
    st.session_state["imgsz"] = IMGSZ_DEFAULT
    st.session_state["conf"] = CONF_DEFAULT
    st.session_state["iou"] = IOU_DEFAULT


# Cached model loading function
@st.cache_resource
def load_model(model_name: str):
    """Load the specified model (cached)."""
    model_paths = {
        "FastSAM": config["models"]["fastsam_path"],
        "MobileSAM": config["models"]["mobilesam_path"],
        "SAM2-t": config["models"]["sam2_path"],
    }

    if model_name == "FastSAM":
        return FastSAM(model_paths["FastSAM"])
    elif model_name == "MobileSAM":
        return SAM(model_paths["MobileSAM"])
    elif model_name == "SAM2-t":
        return SAM(model_paths["SAM2-t"])
    else:
        raise ValueError(f"Invalid model name: {model_name}")


# Load rf classifier for segment landuse classification
@st.cache_resource
def load_classifier():
    with open(Path(config["models"]["classifier_path"]), "rb") as f:
        return pickle.load(f)


def create_map(center, zoom_val):
    """
    Create a folium map with initial settings and return it.
    """

    m = folium.Map(
        location=center, zoom_start=zoom_val, max_zoom=config["map"]["max_zoom"]
    )

    folium.TileLayer(
        tiles=config["map"]["tiles"]["url"],
        attr=config["map"]["tiles"]["attribution"],
        name=config["map"]["tiles"]["name"],
        overlay=True,
        control=True,
        no_wrap=True,
        max_zoom=config["map"]["max_zoom"],
        detect_retina=False,
    ).add_to(m)

    return m


def trigger_segmentation():
    """When click run segmentation"""
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
    ## After running predictions once st.session_state["out"]["all_drawings"]
    ## becomes None. If delete all features with draw edit tools becomes []
    # Therefore current_drawings is Not None current works to achieve this
    # `if current_drawings` would not suffice so do not fix
    current_drawings = st.session_state.get("out", {}).get("all_drawings")
    if current_drawings is not None:
        st.session_state["draw_features"] = current_drawings


def trigger_classification():
    """When click classify segments"""
    current_gdf = st.session_state["gdf"]
    if not current_gdf.empty:
        st.session_state["classify"] = True


def reset_params():
    """When click reset parameters"""
    st.session_state["imgsz"] = IMGSZ_DEFAULT
    st.session_state["conf"] = CONF_DEFAULT
    st.session_state["iou"] = IOU_DEFAULT
    st.session_state["upscale"] = UPSCALE_DEFAULT
    st.session_state["sharp"] = SHARP_DEFAULT


def download_polys(gdf):
    """Download predictions as geojson"""
    return gdf.to_file("data/preds.geojson", driver="GeoJSON")


# Disable predictions when zoomed out
if st.session_state.get("out", False):
    current_zoom = st.session_state["out"]["zoom"]
    if current_zoom < 14:
        st.session_state["predict_disabled"] = True
    else:
        st.session_state["predict_disabled"] = False

# SIDEBAR WIDGETS
with st.sidebar:
    st.selectbox(
        label="📊 Model to use",
        options=config["ui"]["model_options"],
        placeholder="FastSAM",
        key="model_name",
    )
    with st.expander("⚙️ Segment Settings", expanded=False):
        if st.button("Reset parameters"):
            reset_params()
            st.session_state["no_masks"] = None
        st.selectbox(
            "Image upscaling",
            config["ui"]["upscale_options"],
            index=1,
            key="upscale",
            help="The upscaling factor applied to the image before it is segmented. \
            Upscaling uses bilinear rescaling. Higher values are more upscaling, but will \
            slow down prediction time.",
        )
        st.slider(
            "Sharpening amount",
            min_value=0.1,
            max_value=2.5,
            value=st.session_state["sharp"],
            step=0.1,
            key="sharp",
            help="The higher the amount the more the sharpening is applied to the \
            image before it is segmented",
        )
        st.selectbox(
            "Input image size",
            config["ui"]["imgsz_options"],
            index=2,
            key="imgsz",
            help="The resolution to which the input image will be resized before processing.\
            Higher values may improve segmentation accuracy by preserving finer details, \
            however higher values will also increase processing time.",
        )
        st.slider(
            "Confidence threshold",
            min_value=0.01,
            max_value=0.99,
            value=st.session_state["conf"],
            step=0.01,
            key="conf",
            help="Minimum confidence for mask predictions. \
            Higher thresholds may result in fewer predictions.",
        )
        st.slider(
            "IOU threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state["iou"],
            step=0.01,
            key="iou",
            help="Intersect over union non-maximum supression threshold. \
            Higher values mean fewer overlapping predictions are removed",
        )
    st.button(
        "⚡Run Segmentation",
        on_click=trigger_segmentation,
        disabled=st.session_state["predict_disabled"],
        help="Run the specified segmentation model. Must be zoomed in to 14 or 15",
        use_container_width=True,
    )
    if not st.session_state.get("gdf").empty:
        st.button(
            "🏷️ Classify Segments",
            on_click=trigger_classification,
            disabled=st.session_state["classify_disabled"],
            help="Add land use type classifications to segment polygons",
            use_container_width=True,
        )
        geojson_str = st.session_state["gdf"].to_json()
        st.sidebar.download_button(
            "🗺️ Download Result",
            data=geojson_str,
            file_name="seg_preds.geojson",
            mime="application/geo+json",
            use_container_width=True,
        )
    if st.session_state.get("out", False):
        props = st.session_state["out"]
        zoom = props["zoom"]
        lat = props["center"]["lat"]
        lng = props["center"]["lng"]
        st.caption(f"Zoom Level: {zoom}")
        st.caption(f"Centre: {lat:.3f}, {lng:.3f}")

        if not st.session_state["gdf"].empty:
            gdf = st.session_state["gdf"]
            total_area = float(gdf["geometry"].area.sum()) / 10000
            st.caption(f"Prediction area total: {total_area:.2f} ha")
            # Group and display areas by predicted class
            filtered_gdf = gdf[gdf["predicted_class_desc"] != "N/A"]
            if not filtered_gdf.empty:
                grouped_areas = (
                    filtered_gdf.groupby("predicted_class_desc")["geometry"]
                    .apply(lambda geom: geom.area.sum() / 10000)
                    .reset_index()
                )
                for _, row in grouped_areas.iterrows():
                    st.caption(
                        f"- {row['predicted_class_desc']}: {row['geometry']:.2f} ha"
                    )


# Run segmentations but check clean slate and zoomed in
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
            f"Generating {st.session_state['model_name']} predictions...",
            show_time=True,
        ):
            try:
                model = load_model(st.session_state["model_name"])
                gdf = RasterSegmentor(
                    SEGMENT_TIF_PATH,
                    mapped_features,
                    model,
                    upscale=st.session_state["upscale"],
                    sharp=st.session_state["sharp"],
                    imgsz=st.session_state["imgsz"],
                    conf=st.session_state["conf"],
                    iou=st.session_state["iou"],
                ).process_window()
            # Failed to run
            except Exception as e:
                st.session_state["no_masks"] = True
                st.session_state["segmentation_run"] = False
                st.session_state["gdf"] = gpd.GeoDataFrame([])
                st.error(f"Segmentation failed: {e}")
        # Created no results
        if gdf.empty:
            st.session_state["no_masks"] = True
            st.session_state["segmentation_run"] = False
            st.session_state["gdf"] = gpd.GeoDataFrame([])
        # Success
        else:
            gdf["predicted_class_desc"] = "N/A"
            st.session_state["no_masks"] = False
            st.session_state["segmentation_run"] = False
            st.session_state["gdf"] = gdf
            st.session_state["classify_disabled"] = False

# Classify segments using rf if triggered
if st.session_state.get("classify", False):
    current_gdf = st.session_state["gdf"]
    with st.spinner(f"Classifying land use of {len(current_gdf)} segments..."):
        classifier = load_classifier()
        classified_gdf = classify_polygons(current_gdf, CLASSIFY_TIF_PATH, classifier)
        st.session_state["gdf"] = classified_gdf
        st.session_state["classify"] = False
        st.session_state["classify_disabled"] = True

# Warning when no results / segmentation failed
if st.session_state.get("no_masks"):
    st.error(
        "No segmentation masks were generated with those parameters.  \n"
        "Try lowering the confidence, or increasing IoU, or image size."
    )

# Initial map positions
if "zoom" not in st.session_state:
    st.session_state["zoom"] = config["map"]["default_zoom"]

if "center" not in st.session_state:
    st.session_state["center"] = config["map"]["default_center"]


# PREDICTIONS LAYER
pred_fg = folium.FeatureGroup(name="Prediction Polygons", control=True)
# Predictions to feature group
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
                fields=["predicted_class_desc", "area_disp"],
                labels=True,
                aliases=["Predicted Use", "Area"],
                localize=True,
                sticky=False,
            ),
            overlay=True,
            control=True,
        )
    )


# DRAWING LAYERS
# Drawing feature group
fg = folium.FeatureGroup(name="Drawing features", control=True)

if st.session_state.get("draw_features", False):
    for feature in st.session_state["draw_features"]:
        folium.GeoJson(feature).add_to(fg)

# Create folium map
m = create_map(st.session_state["center"], st.session_state["zoom"])

# Add layers to map
m.add_child(fg)
m.add_child(pred_fg)

# Drawing tools
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

# Can switch layers on and off in top right of map
folium.LayerControl().add_to(m)

# Create the streamlit folium st_folium object

out = st_folium(
    m,
    center=st.session_state["center"],
    zoom=st.session_state["zoom"],
    feature_group_to_add=[fg, pred_fg],
    key="out",
    use_container_width=True,
    pixelated=False,
    height=550,
)
