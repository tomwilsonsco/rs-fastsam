import rasterio as rio
from rasterio.windows import from_bounds, transform as window_transform
from rasterio.features import shapes
from rasterio.crs import CRS
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from typing import List,Tuple, Union
from rasterio.transform import Affine
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


def unsharp(img_arr, amount=1.0, radius=1.2):
    """
    Apply an unsharp mask to an image array to enhance its sharpness.

    Parameters:
        img_arr (numpy.ndarray): The input image array. Expected to be in a format
                                 compatible with OpenCV (e.g., uint8).
        amount (float, optional): The scaling factor for the unsharp mask.
                                  Higher values increase the sharpness effect. Default is 1.
        radius (float, optional): The standard deviation of the Gaussian blur
                                  used to create the unsharp mask. Default is 1.2.

    Returns:
        numpy.ndarray: The sharpened image array with values clipped to the range [1, 255].
    """
    arr_f = img_arr.astype(np.float32)
    blur = cv2.GaussianBlur(arr_f, (0, 0), radius)
    sharp = arr_f + amount * (arr_f - blur)
    sharp = np.clip(sharp, 1, 255).astype(np.uint8)

    return sharp
