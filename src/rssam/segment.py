import rasterio as rio
from rasterio.features import shapes
from rasterio.crs import CRS
from shapely.geometry import shape
from ultralytics import FastSAM, SAM
import numpy as np
import geopandas as gpd
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional
from rasterio.transform import Affine
import streamlit as st
import cv2

from src.rssam.stfolium_utils import extract_features, extent_to_gdf
from src.rssam.raster_utils import (
    get_crs,
    get_window_array,
    to_pixel_coordinates,
    validate_boxes,
    unsharp,
)


class RasterSegmentor:
    """
    Segment windows of raster image based on input features and extent coordinates
    """

    def __init__(
        self,
        tif_path: str,
        features: List[Dict],
        model: Union[FastSAM, SAM],
        upscale: int = 2,
        sharp: float = 1.2,
        imgsz: int = 1024,
        conf: float = 0.2,
        iou: float = 0.5,
        min_area: float = 1000.0,
    ):
        self.tif_path = tif_path
        self.features = features
        self.model = model
        self.upscale = upscale
        self.sharp = sharp
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.min_area = min_area

        with rio.open(self.tif_path) as f:
            self.crs = f.crs
            self.transform = f.transform

    def _prepare_inputs(
        self,
    ) -> Tuple[str, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Prepare CRS, extent, points, and box geometries from a TIFF path and GeoJSON features.

        Args:
            tif_path (str): Path to the TIFF image file.
            features (List[Dict]): List of GeoJSON-style features (points and polygons).

        Returns:
            Tuple[str, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]: Tuple containing the CRS string, extent GeoDataFrame, point GeoDataFrame, and box GeoDataFrame.
        """
        extent_gdf = extent_to_gdf().to_crs(self.crs)
        points_gdf, boxes_gdf = extract_features(self.features)
        points_gdf = points_gdf.to_crs(self.crs)
        boxes_gdf = boxes_gdf.to_crs(self.crs)
        return extent_gdf, points_gdf, boxes_gdf

    def _predict_extent(
        self,
        img_win: np.ndarray,
        win_trans: Affine,
        extent_gdf: gpd.GeoDataFrame,
        points: gpd.GeoDataFrame,
    ) -> Optional[np.ndarray]:
        """
        Run segmentation model inference on a raster window and return binary masks.

        Args:
            img_win (np.ndarray): Image window array (H, W, C) extracted from the raster.
            win_trans (Affine): Affine transform for the image window.
            extent_gdf (gpd.GeoDataFrame): GeoDataFrame defining the spatial extent of the window.
            points (gpd.GeoDataFrame): GeoDataFrame of input points for segmentation.
            model (Union[FastSAM, SAM]): Model to use for prediction ("FastSAM", "MobileSAM", or "SAM2-t").
            imgsz (int): Input image size for the model.
            conf (float): Confidence threshold for mask generation.
            iou (float): IoU threshold for mask filtering.

        Returns:
            Optional[np.ndarray]: Numpy array of predicted binary masks, or None if prediction fails.
        """
        points = to_pixel_coordinates(points, extent_gdf, win_trans)

        if img_win is None or not img_win.any():
            return None

        try:
            results = self.model.predict(
                img_win,
                points=points if points else None,
                device="cpu",
                retina_masks=True,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
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

    def _masks_to_geodataframe(
        self,
        masks: np.ndarray,
        transform: Affine,
        box_area: float,
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

        gdf = gpd.GeoDataFrame(geometry=polygons, crs=self.crs)
        #gdf = gpd.overlay(gdf, gdf, how="union")
        gdf = gdf.explode(index_parts=False)
        gdf = gdf[gdf.is_valid]
        gdf["area"] = gdf.area
        gdf = gdf[(gdf.area >= self.min_area) & (gdf.area / box_area < 0.9)]
        gdf["area_disp"] = (gdf.area / 10000).round(2).astype(str) + " ha"

        return gdf

    def _process_box_segmentations(
        self,
        boxes_gdf: gpd.GeoDataFrame,
        points_gdf: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """
        Run segmentation for each bounding box and return a combined GeoDataFrame of polygons.

        Args:
            tif_path (str): Path to the TIFF image file.
            boxes_gdf (gpd.GeoDataFrame): GeoDataFrame containing input bounding boxes.
            points_gdf (gpd.GeoDataFrame): GeoDataFrame of labeled point features.
            model (Union[FastSAM, SAM]): Name of the segmentation model to use.
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
            img_win, win_trans = get_window_array(self.tif_path, b, self.upscale)
            img_win = unsharp(img_win, self.sharp)
            masks = self._predict_extent(img_win, win_trans, b, points_gdf)
            if masks is not None:
                m_gdf = self._masks_to_geodataframe(masks, win_trans, box_area)
                if not m_gdf.empty:
                    gdfs.append(m_gdf)
        return (
            gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
            if gdfs
            else gpd.GeoDataFrame()
        )

    def _process_full_extent_segmentation(
        self,
        extent_gdf: gpd.GeoDataFrame,
        points_gdf: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """
        Run segmentation across the full map extent and return the resulting polygons.

        Args:
            tif_path (str): Path to the TIFF image file.
            extent_gdf (gpd.GeoDataFrame): GeoDataFrame representing the full bounding box extent.
            points_gdf (gpd.GeoDataFrame): GeoDataFrame of labeled point features.
            model (Union[FastSAM, SAM]): FastSAM or SAM model instance to use.
            imgsz (int): Image size to be passed to the model.
            conf (float): Confidence threshold for segmentation.
            iou (float): IoU threshold for segmentation.
            crs (str): Coordinate reference system to assign to the output geometries.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing the segmented polygons within the full extent.
        """
        img_win, win_trans = get_window_array(self.tif_path, extent_gdf, self.upscale)
        img_win = unsharp(img_win, self.sharp)
        box_area = extent_gdf.geometry.area.sum()
        masks = self._predict_extent(
            img_win,
            win_trans,
            extent_gdf,
            points_gdf,
        )
        return self._masks_to_geodataframe(masks, win_trans, box_area)

    def process_window(self) -> gpd.GeoDataFrame:
        """
        Main public method to run image segmentation on a TIFF file using user-supplied features and return polygon geometries.

        Args:
            tif_path (str): Path to the TIFF image file.
            features (List[Dict]): List of GeoJSON-style features containing points and polygons.
            model (Union[FastSAM, SAM]): Name of the segmentation model to use ("FastSAM", "MobileSAM", "SAM2-t"). Defaults to "FastSAM".
            imgsz (int, optional): Image size to be passed to the model. Defaults to 1024.
            conf (float, optional): Confidence threshold for segmentation. Defaults to 0.2.
            iou (float, optional): IoU threshold for segmentation. Defaults to 0.5.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing segmented polygons with area metadata.
        """
        extent_gdf, points_gdf, boxes_gdf = self._prepare_inputs()

        if not boxes_gdf.empty:
            boxes_gdf = validate_boxes(boxes_gdf, extent_gdf)
            if not boxes_gdf.empty:
                gdf = self._process_box_segmentations(
                    boxes_gdf,
                    points_gdf,
                )
                if not gdf.empty:
                    return gdf

        return self._process_full_extent_segmentation(
            extent_gdf,
            points_gdf,
        )
