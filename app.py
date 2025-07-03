import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw
import rasterio as rio
from pathlib import Path
from rasterio.windows import from_bounds, transform as window_transform
from rasterio.features import shapes
from shapely.geometry import shape
from ultralytics import FastSAM, SAM
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import box, Point, Polygon

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


def get_crs(tif_path):
    with rio.open(tif_path) as f:
        prof = f.profile
        return prof["crs"]


def to_pixel_coordinates(coordinates, crs, win_trans):
    """Convert list of [lon, lat] coordinates to pixel coordinates.

    For each [lon, lat] coordinate:
    - Convert from EPSG:4326 (WGS84) to the image CRS (profile['crs']).
    - Convert world coordinates to pixel coordinates rasterio.transform.rowcol
    - Exclude any points that fall outside the image bounds defined by profile['width'] and profile['height'].

    Returns a list of [col, row] pixel coordinates.
    """

    if not coordinates:
        return None

    gdf = gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lon, lat in coordinates],
        crs="EPSG:4326",
    )

    gdf = gdf.to_crs(crs)

    bbox = extent_to_gdf().to_crs(crs)

    minx, miny, maxx, maxy = bbox.bounds.iloc[0].to_list()

    pixel_coords = []
    for i, gdf_row in gdf.iterrows():
        x, y = gdf_row.geometry.x, gdf_row.geometry.y
        if minx <= x <= maxx and miny <= y <= maxy:
            row, col = rio.transform.rowcol(win_trans, x, y, op=round)
            pixel_coords.append([int(col), int(row)])
    return pixel_coords


def get_current_extent():
    minx = st.session_state["out"]["bounds"]["_southWest"]["lng"]
    miny = st.session_state["out"]["bounds"]["_southWest"]["lat"]
    maxx = st.session_state["out"]["bounds"]["_northEast"]["lng"]
    maxy = st.session_state["out"]["bounds"]["_northEast"]["lat"]
    return minx, miny, maxx, maxy


def extent_to_gdf():
    minx, miny, maxx, maxy = get_current_extent()
    bbox = box(minx, miny, maxx, maxy)
    return gpd.GeoDataFrame(geometry=[bbox], crs="epsg:4326")


def get_window_array(tif_path, bbox_gdf=None):

    with rio.open(tif_path) as f:
        profile = f.profile

    if not isinstance(bbox_gdf, gpd.GeoDataFrame):
        use_gdf = extent_to_gdf()  
    else:
        use_gdf = bbox_gdf      
    
    use_gdf = use_gdf.to_crs(profile["crs"])
    minx, miny, maxx, maxy = use_gdf.bounds.iloc[0].to_list()
    minx, miny, maxx, maxy = minx, miny, maxx, maxy
    win = from_bounds(minx, miny, maxx, maxy, profile["transform"])

    with rio.open(tif_path) as f:
        win_arr = f.read(window=win)
        win_arr = win_arr.transpose(1, 2, 0)

    win_trans = window_transform(win, profile["transform"])

    return win_arr, win_trans


def convert_boxes(boxes, crs):
    minx, miny, maxx, maxy = get_current_extent()
    buffer_threshold = 0.0001
    extent_bbox = box(
        minx + buffer_threshold,
        miny + buffer_threshold,
        maxx - buffer_threshold,
        maxy - buffer_threshold,
    )
    drawn_bboxes = [tuple(Polygon(poly[0]).bounds) for poly in boxes]
    results = []
    for rect in drawn_bboxes:
        rect_box = box(*rect)
        if rect_box.intersects(extent_bbox):
            clipped = rect_box.intersection(extent_bbox)
            if not clipped.is_empty:
                results.append(clipped)
    results_gdf = gpd.GeoDataFrame(geometry=results, crs="epsg:4326").to_crs(crs)
    return results_gdf


def predict_extent(
    img_win, win_trans, crs, points, labels, use_model, imgsz, conf, iou
):

    points = to_pixel_coordinates(points, crs, win_trans)

    if img_win is None or not img_win.any():
        return None
    # Load model and run inference
    if use_model == "FastSAM":
        model = FastSAM("FastSAM-x.pt")
        imgsz = imgsz
    elif use_model == "MobileSAM":
        model = SAM("mobile_sam.pt")
        imgsz = imgsz
    elif use_model == "SAM2-t":
        model = SAM("sam2_s.pt")
        imgsz = imgsz
    else:
        raise ValueError(
            f"Invalid model name: {use_model}. Expected 'FastSAM', 'MobileSAM', or 'SAM2-t'."
        )
    try:
        results = model.predict(
            img_win,
            points=points if points else None,
            labels=labels,
            device="cpu",
            retina_masks=True,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            verbose=True,
            max_det=1200,
        )
    except Exception as e:
        st.error(f"An error occurred during segmentation: {e}")
        return None

    res = results[0]
    if res.masks is None:
        return None
    else:
        return res.masks.data.cpu().numpy()


def masks_to_geodataframe(masks, transform, crs, min_area=1000):
    """
    Convert segmentation masks to a GeoDataFrame of polygons.

    Parameters:
    masks (numpy.ndarray): Array of segmentation masks.
    transform (Affine): Affine transformation for the raster.
    crs (str): Coordinate reference system for the GeoDataFrame.
    min_area (float): Minimum area threshold for polygons to be included.

    Returns:
    geopandas.GeoDataFrame: GeoDataFrame containing valid polygons.
    """
    polygons = []
    for mask in masks:
        m = mask.astype("uint8")
        for geom_dict, val in shapes(m, transform=transform):
            if val == 1:
                polygons.append(shape(geom_dict))

    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    gdf = gdf[gdf.is_valid]
    gdf["area"] = gdf.geometry.area
    gdf = gdf.explode(index_parts=False)
    gdf["area"] = gdf.geometry.area
    gdf = gdf[gdf["area"] >= min_area]

    return gdf


def create_segmentation_geojson(
    tif_path,
    points=None,
    boxes=None,
    labels=None,
    use_model="FastSAM",
    imgsz=1024,
    conf=0.2,
    iou=0.5,
):
    """
    Run segmentation on the TIFF image, convert masks to polygon geometries,
    and save them as a GeoJSON file.

    Parameters:
    tif_path (str): Path to the TIFF image.
    output_geojson_path (str): Output path for the GeoJSON file.
    coords (list, optional): Optional points for interactive segmentation.
    """
    crs = get_crs(tif_path)

    box_process = False

    if boxes:
        box_process = True
        boxes_gdf = convert_boxes(boxes, crs)
        if not boxes_gdf.empty:
            gdfs = [gpd.GeoDataFrame([])]
            for i in range(len(boxes_gdf)):
                b = boxes_gdf.iloc[[i]]
                img_win, win_trans = get_window_array(tif_path, b)
                m = predict_extent(
                    img_win, win_trans, crs, points, labels, use_model, imgsz, conf, iou
                )
                if m is not None:
                    m_gdf = masks_to_geodataframe(m, win_trans, crs)
                    if not m_gdf.empty:
                        gdfs.append(m_gdf)
            gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
        else:
            box_process = False

    if not box_process:
        img_win, win_trans = get_window_array(tif_path)
        masks = predict_extent(
            img_win, win_trans, crs, points, labels, use_model, imgsz, conf, iou
        )
        gdf = masks_to_geodataframe(masks, win_trans, crs)

    return gdf


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
        st.session_state["out"]["center"]["lat"],
        st.session_state["out"]["center"]["lng"],
    ]
    old_zoom = st.session_state["out"]["zoom"]
    st.session_state["center"] = old_center
    st.session_state["zoom"] = old_zoom
    current_drawings = st.session_state["out"]["all_drawings"]
    if current_drawings:
        all_points = [
            p["geometry"]["coordinates"]
            for p in current_drawings
            if p["geometry"]["type"] == "Point"
        ]
        st.session_state["points"] = all_points
        all_rectangles = [
            p["geometry"]["coordinates"]
            for p in current_drawings
            if p["geometry"]["type"] == "Polygon"
        ]
        st.session_state["rectangles"] = all_rectangles


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
    st.session_state["no_masks"] = False
    st.session_state["points"] = []
    st.session_state["rectangles"] = []
    st.session_state["labels"] = []
    if st.session_state["out"].get("center", False):
        clean_center = [
            st.session_state["out"]["center"]["lat"] + 0.0001,
            st.session_state["out"]["center"]["lng"],
        ]
        clean_zoom = st.session_state["out"]["zoom"]
        st.session_state["center"] = clean_center
        st.session_state["zoom"] = clean_zoom


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
    with st.expander("Segmentation parameters", expanded=False):
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
    st.button(
        "Delete Drawn Features",
        on_click=delete_features,
        help="Delete all points and rectangles you have drawn.",
    )
    st.button(
        "Delete Predictions",
        on_click=clear_segmentation,
        help="Clear segmentation prediction polygons.",
    )
    if not st.session_state.get("gdf").empty:
        geojson_str = st.session_state["gdf"].to_json()
        st.sidebar.download_button(
            "Download predictions",
            data=geojson_str,
            file_name="seg_preds.geojson",
            mime="application/geo+json",
        )
    st.radio("Marker type", ["Foreground", "Background"], key="fore_or_back")
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
        mapped_points = st.session_state["points"]
        mapped_rectangles = st.session_state["rectangles"]
        with st.spinner(
            f"Generating {st.session_state["model_name"]} predictions...",
            show_time=True,
        ):
            gdf = create_segmentation_geojson(
                TIF_PATH,
                mapped_points,
                mapped_rectangles,
                st.session_state["labels"],
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
            color="red",
            fill=False,
            overlay=True,
            control=True,
        )
    )


# POINTS LAYER

fg = folium.FeatureGroup(name="Drawing features", control=True)


for point in st.session_state.get("points", None):
    fg.add_child(
        folium.Marker(location=[point[1], point[0]], name="point markers", control=True)
    )

# RECTANGLES LAYER
for rect in st.session_state.get("rectangles", None):
    rect = rect[0]
    rect = [[lat, lon] for lon, lat in rect]
    fg.add_child(
        folium.Polygon(
            locations=rect,
            name="rectangle markers",
            fill=True,
            fill_opacity=0.2,
            control=True,
        )
    )

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
    edit_options={"edit": False, "remove": False},
)
draw.add_to(m)

folium.LayerControl().add_to(m)


def point_labels():
    if st.session_state.get("out", False):
        f_or_g = {"Foreground": 1, "Background": 0}
        current_drawings = st.session_state["out"]["all_drawings"]
        if current_drawings:
            all_points = [
                p["geometry"]["coordinates"]
                for p in current_drawings
                if p["geometry"]["type"] == "Point"
            ]
            all_labels = st.session_state["labels"]
            if len(all_labels) < len(all_points):
                current_type = st.session_state["fore_or_back"]
                all_labels.append(f_or_g[current_type])
                st.session_state["labels"] = all_labels


print(st.session_state["labels"])

out = st_folium(
    m,
    center=st.session_state["center"],
    zoom=st.session_state["zoom"],
    feature_group_to_add=[fg, pred_fg],
    key="out",
    height=600,
    width=900,
    pixelated=True,
    on_change=point_labels,
)
