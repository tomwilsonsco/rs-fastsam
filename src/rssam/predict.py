import rasterio as rio
from rasterio.windows import from_bounds, transform as window_transform
from rasterio.features import shapes
from rasterio.crs import CRS
from rasterio.enums import Resampling
from shapely.geometry import shape
from ultralytics import FastSAM, SAM
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from typing import List, Dict, Tuple, Union, Optional
from rasterio.transform import Affine
import streamlit as st
import cv2


def get_crs(tif_path: str) -> Union[str, CRS]:
    """
    Retrieve the coordinate reference system (CRS) from a raster TIFF file.

    Args:
        tif_path (str): Path to the TIFF image file.

    Returns:
        Union[str, CRS]: The CRS of the raster, as a string or rasterio CRS object.
    """
    with rio.open(tif_path) as f:
        prof = f.profile
        return prof["crs"]


def to_pixel_coordinates(
    points_gdf: gpd.GeoDataFrame, extent_gdf: gpd.GeoDataFrame, win_trans: Affine
) -> List[List[int]]:
    """
    Convert point geometries to pixel coordinates relative to a raster window.

    Args:
        points_gdf (gpd.GeoDataFrame): GeoDataFrame of point geometries.
        extent_gdf (gpd.GeoDataFrame): GeoDataFrame defining the spatial extent (bounding box).
        win_trans (Affine): Affine transform for the raster window.

    Returns:
        List[List[int]]: List of [col, row] pixel coordinates for points within the extent.
    """
    minx, miny, maxx, maxy = extent_gdf.bounds.iloc[0].to_list()

    pixel_coords = []
    for i, gdf_row in points_gdf.iterrows():
        x, y = gdf_row.geometry.x, gdf_row.geometry.y
        if minx <= x <= maxx and miny <= y <= maxy:
            row, col = rio.transform.rowcol(win_trans, x, y, op=round)
            pixel_coords.append([int(col), int(row)])
    return pixel_coords


def get_current_extent() -> Tuple[float, float, float, float]:
    """
    Retrieve the current map view bounds from the Streamlit session state.

    Returns:
        Tuple[float, float, float, float]: Bounding box as (minx, miny, maxx, maxy) in WGS84 coordinates.
    """
    minx = st.session_state["out"]["bounds"]["_southWest"]["lng"]
    miny = st.session_state["out"]["bounds"]["_southWest"]["lat"]
    maxx = st.session_state["out"]["bounds"]["_northEast"]["lng"]
    maxy = st.session_state["out"]["bounds"]["_northEast"]["lat"]
    return minx, miny, maxx, maxy


def extent_to_gdf():
    """
    Convert the current map view extent to a GeoDataFrame with a bounding box geometry.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing the current extent as a single polygon in EPSG:4326.
    """
    minx, miny, maxx, maxy = get_current_extent()
    bbox = box(minx, miny, maxx, maxy)
    return gpd.GeoDataFrame(geometry=[bbox], crs="epsg:4326")


def get_window_array(
    tif_path: str, extent_gdf: gpd.GeoDataFrame, upscale_factor: int = 4
) -> Tuple[np.ndarray, Affine]:
    """
    Read an upscaled windowed array and its affine transform from a raster using a given extent.

    Args:
        tif_path (str): Path to the TIFF image file.
        extent_gdf (gpd.GeoDataFrame): GeoDataFrame representing the bounding extent.
        upscale_factor (int, optional): Factor by which to upscale the windowed image. Defaults to 2.
        resampling (Resampling, optional): Resampling method for upscaling. Defaults to Resampling.bilinear.

    Returns:
        Tuple[np.ndarray, Affine]: Upscaled windowed image array (H, W, C) and its affine transform.
    """

    with rio.open(tif_path) as f:
        prof = f.profile
        trans = prof["transform"]

        # Calculate window from bounds
        minx, miny, maxx, maxy = extent_gdf.bounds.iloc[0].to_list()
        win = from_bounds(minx, miny, maxx, maxy, trans)

        win_arr = f.read(
            window=win,
        )

        win_arr = np.moveaxis(win_arr, 0, 2)

        # Read windowed data with upscaling
        height = int(win_arr.shape[0] * upscale_factor)
        width = int(win_arr.shape[1] * upscale_factor)

        win_arr_up = cv2.resize(
            win_arr, (width, height), interpolation=cv2.INTER_LINEAR
        )

        # Adjust transform for window and upscaling
        win_trans = window_transform(win, trans)
        upscale_transform = win_trans * Affine.scale(
            1 / upscale_factor, 1 / upscale_factor
        )

    return win_arr_up, upscale_transform


def validate_boxes(
    boxes_gdf: gpd.GeoDataFrame, extent_gdf: gpd.GeoDataFrame, edge_removal: float = -15
) -> gpd.GeoDataFrame:
    """
    Clip bounding boxes to a buffered extent and return only valid, non-empty geometries.

    Args:
        boxes_gdf (gpd.GeoDataFrame): GeoDataFrame of input bounding box geometries.
        extent_gdf (gpd.GeoDataFrame): GeoDataFrame representing the extent to buffer and clip to.
        edge_removal (float, optional): Distance (in CRS units) to buffer inward from extent. Defaults to -15.

    Returns:
        gpd.GeoDataFrame: Filtered GeoDataFrame containing only valid, non-empty geometries.
    """
    extent_selection = extent_gdf.buffer(edge_removal)
    boxes_gdf["geometry"] = boxes_gdf["geometry"].intersection(extent_selection)

    boxes_gdf = boxes_gdf[
        boxes_gdf["geometry"].is_valid & ~boxes_gdf["geometry"].is_empty
    ]

    return boxes_gdf


def predict_extent(
    img_win: np.ndarray,
    win_trans: Affine,
    extent_gdf: gpd.GeoDataFrame,
    points: gpd.GeoDataFrame,
    labels: Optional[List[int]],
    use_model: str,
    imgsz: int,
    conf: float,
    iou: float,
) -> Optional[np.ndarray]:
    """
    Run segmentation model inference on a raster window and return binary masks.

    Args:
        img_win (np.ndarray): Image window array (H, W, C) extracted from the raster.
        win_trans (Affine): Affine transform for the image window.
        extent_gdf (gpd.GeoDataFrame): GeoDataFrame defining the spatial extent of the window.
        points (gpd.GeoDataFrame): GeoDataFrame of input points for segmentation.
        labels (Optional[List[int]]): List of binary labels associated with the points.
        use_model (str): Model name to use for prediction ("FastSAM", "MobileSAM", or "SAM2-t").
        imgsz (int): Input image size for the model.
        conf (float): Confidence threshold for mask generation.
        iou (float): IoU threshold for mask filtering.

    Returns:
        Optional[np.ndarray]: Numpy array of predicted binary masks, or None if prediction fails.
    """
    points = to_pixel_coordinates(points, extent_gdf, win_trans)

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
            verbose=False,
        )
    except Exception as e:
        st.error(f"An error occurred during segmentation: {e}")
        return None

    res = results[0]
    if res.masks is None:
        return None
    else:
        return res.masks.data.cpu().numpy()


def masks_to_geodataframe(
    masks: np.ndarray,
    transform: Affine,
    crs: Union[str, dict],
    box_area: float,
    min_area: float = 1000,
) -> gpd.GeoDataFrame:
    """
    Convert a set of binary masks into a GeoDataFrame of filtered polygon geometries.

    Args:
        masks (np.ndarray): Array of binary segmentation masks with shape (N, H, W).
        transform (Affine): Affine transform associated with the raster window.
        crs (Union[str, dict]): Coordinate reference system for the output geometries.
        box_area (float): Area of the bounding box used to limit overly large polygons.
        min_area (float, optional): Minimum polygon area to retain (default is 1000).

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing valid, filtered polygon geometries with area labels.
    """
    polygons = []
    for mask in masks:
        m = mask.astype("uint8")
        for geom_dict, val in shapes(m, transform=transform):
            if val == 1:
                polygons.append(shape(geom_dict))

    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)

    gdf = gdf.explode(index_parts=False)
    gdf = gdf[gdf.is_valid]
    gdf["area"] = gdf.area
    gdf = gdf[(gdf.area >= min_area) & (gdf.area / box_area < 0.9)]
    gdf["area_disp"] = (gdf.area / 10000).round(2).astype(str) + " ha"

    return gdf


def extract_features(features):
    """
    Split a list of GeoJSON features into two GeoDataFrames:
    one containing only polygons and one containing only points.

    Parameters:
    features (list): List of GeoJSON features (dicts).

    Returns:
    tuple: (polygon_gdf, point_gdf)
    """
    polygon_geoms = [
        shape(f["geometry"])
        for f in features
        if f and f["geometry"]["type"] == "Polygon"
    ]
    point_geoms = [
        shape(f["geometry"]) for f in features if f and f["geometry"]["type"] == "Point"
    ]

    poly_gdf = gpd.GeoDataFrame(geometry=polygon_geoms, crs="EPSG:4326")
    point_gdf = gpd.GeoDataFrame(geometry=point_geoms, crs="EPSG:4326")

    return point_gdf, poly_gdf


def prepare_inputs(
    tif_path: str, features: List[Dict]
) -> Tuple[str, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Prepare CRS, extent, points, and box geometries from a TIFF path and GeoJSON features.

    Args:
        tif_path (str): Path to the TIFF image file.
        features (List[Dict]): List of GeoJSON-style features (points and polygons).

    Returns:
        Tuple[str, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]: Tuple containing the CRS string, extent GeoDataFrame, point GeoDataFrame, and box GeoDataFrame.
    """
    crs = get_crs(tif_path)
    extent_gdf = extent_to_gdf().to_crs(crs)
    points_gdf, boxes_gdf = extract_features(features)
    points_gdf = points_gdf.to_crs(crs)
    boxes_gdf = boxes_gdf.to_crs(crs)
    return crs, extent_gdf, points_gdf, boxes_gdf


def unsharp(img_arr, radius=1.2, amount=1.0):
    """
    Apply an unsharp mask to an image array to enhance its sharpness.

    Parameters:
        img_arr (numpy.ndarray): The input image array. Expected to be in a format
                                 compatible with OpenCV (e.g., uint8).
        radius (float, optional): The standard deviation of the Gaussian blur
                                  used to create the unsharp mask. Default is 1.2.
        amount (float, optional): The scaling factor for the unsharp mask.
                                  Higher values increase the sharpness effect. Default is 1.0.

    Returns:
        numpy.ndarray: The sharpened image array with values clipped to the range [1, 255].
    """
    arr_f = img_arr.astype(np.float32)
    blur = cv2.GaussianBlur(arr_f, (0, 0), radius)
    sharp = arr_f + amount * (arr_f - blur)
    sharp = np.clip(sharp, 1, 255).astype(np.uint8)

    return sharp


def process_box_segmentations(
    tif_path: str,
    boxes_gdf: gpd.GeoDataFrame,
    points_gdf: gpd.GeoDataFrame,
    labels: Optional[List[int]],
    use_model: str,
    upscale: int,
    sharp: float,
    imgsz: int,
    conf: float,
    iou: float,
    crs: str,
) -> gpd.GeoDataFrame:
    """
    Run segmentation for each bounding box and return a combined GeoDataFrame of polygons.

    Args:
        tif_path (str): Path to the TIFF image file.
        boxes_gdf (gpd.GeoDataFrame): GeoDataFrame containing input bounding boxes.
        points_gdf (gpd.GeoDataFrame): GeoDataFrame of labeled point features.
        labels (Optional[List[int]]): List of binary labels associated with the points.
        use_model (str): Name of the segmentation model to use.
        imgsz (int): Image size to be passed to the model.
        conf (float): Confidence threshold for segmentation.
        iou (float): IoU threshold for segmentation.
        crs (str): Coordinate reference system to assign to the output geometries.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing all valid polygons from the segmented boxes.
    """
    gdfs = []
    for _, b in boxes_gdf.iterrows():
        b = gpd.GeoDataFrame([b], crs=boxes_gdf.crs)
        box_area = b.geometry.area.sum()
        img_win, win_trans = get_window_array(tif_path, b)
        img_win = unsharp(img_win)
        masks = predict_extent(
            img_win, win_trans, b, points_gdf, labels, use_model, imgsz, conf, iou
        )
        if masks is not None:
            m_gdf = masks_to_geodataframe(masks, win_trans, crs, box_area)
            if not m_gdf.empty:
                gdfs.append(m_gdf)
    return (
        gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
        if gdfs
        else gpd.GeoDataFrame()
    )


def process_full_extent_segmentation(
    tif_path: str,
    extent_gdf: gpd.GeoDataFrame,
    points_gdf: gpd.GeoDataFrame,
    labels: Optional[List[int]],
    use_model: str,
    upscale: int,
    sharp: float,
    imgsz: int,
    conf: float,
    iou: float,
    crs: str,
) -> gpd.GeoDataFrame:
    """
    Run segmentation across the full map extent and return the resulting polygons.

    Args:
        tif_path (str): Path to the TIFF image file.
        extent_gdf (gpd.GeoDataFrame): GeoDataFrame representing the full bounding box extent.
        points_gdf (gpd.GeoDataFrame): GeoDataFrame of labeled point features.
        labels (Optional[List[int]]): List of binary labels associated with the points.
        use_model (str): Name of the segmentation model to use.
        imgsz (int): Image size to be passed to the model.
        conf (float): Confidence threshold for segmentation.
        iou (float): IoU threshold for segmentation.
        crs (str): Coordinate reference system to assign to the output geometries.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing the segmented polygons within the full extent.
    """
    img_win, win_trans = get_window_array(tif_path, extent_gdf)
    img_win = unsharp(img_win)
    box_area = extent_gdf.geometry.area.sum()
    masks = predict_extent(
        img_win, win_trans, extent_gdf, points_gdf, labels, use_model, imgsz, conf, iou
    )
    return masks_to_geodataframe(masks, win_trans, crs, box_area)


def create_segmentation_geojson(
    tif_path: str,
    features: List[Dict],
    labels: Optional[List[int]] = None,
    use_model: str = "FastSAM",
    upscale: int = 2,
    sharp: float = 1.2,
    imgsz: int = 1024,
    conf: float = 0.2,
    iou: float = 0.5,
) -> gpd.GeoDataFrame:
    """
    Run image segmentation on a TIFF file using user-supplied features and return polygon geometries.

    Args:
        tif_path (str): Path to the TIFF image file.
        features (List[Dict]): List of GeoJSON-style features containing points and polygons.
        labels (Optional[List[int]], optional): List of binary labels associated with point features. Defaults to None.
        use_model (str, optional): Name of the segmentation model to use ("FastSAM", "MobileSAM", "SAM2-t"). Defaults to "FastSAM".
        imgsz (int, optional): Image size to be passed to the model. Defaults to 1024.
        conf (float, optional): Confidence threshold for segmentation. Defaults to 0.2.
        iou (float, optional): IoU threshold for segmentation. Defaults to 0.5.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing segmented polygons with area metadata.
    """
    crs, extent_gdf, points_gdf, boxes_gdf = prepare_inputs(tif_path, features)

    if not boxes_gdf.empty:
        boxes_gdf = validate_boxes(boxes_gdf, extent_gdf)
        if not boxes_gdf.empty:
            gdf = process_box_segmentations(
                tif_path,
                boxes_gdf,
                points_gdf,
                labels,
                use_model,
                upscale,
                sharp,
                imgsz,
                conf,
                iou,
                crs,
            )
            if not gdf.empty:
                return gdf

    return process_full_extent_segmentation(
        tif_path,
        extent_gdf,
        points_gdf,
        labels,
        use_model,
        upscale,
        sharp,
        imgsz,
        conf,
        iou,
        crs,
    )
