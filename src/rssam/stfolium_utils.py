import geopandas as gpd
from shapely.geometry import box
from typing import Tuple
from shapely.geometry import shape
import streamlit as st

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