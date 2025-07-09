import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
from shapely.ops import polygonize
from typing import List


def planar_union(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Create a planar union of polygons that preserves all internal boundaries.

    Args:
        gdf: GeoDataFrame containing polygon geometries to union.

    Returns:
        GeoDataFrame with non-overlapping polygons that partition the union
        of all input polygons, preserving internal boundaries.
    """
    # Reset index to ensure clean processing
    gdf = gdf.reset_index(drop=True)

    # Create all possible intersections between polygons
    intersections: List = []

    for i in range(len(gdf)):
        for j in range(i + 1, len(gdf)):
            geom_i = gdf.geometry.iloc[i]
            geom_j = gdf.geometry.iloc[j]

            # Find intersection
            intersection = geom_i.intersection(geom_j)

            if not intersection.is_empty:
                intersections.append(intersection)

    # Start with original geometries
    result_geoms: List = list(gdf.geometry)

    # Add all intersections
    result_geoms.extend(intersections)

    # Create union of all geometries to get non-overlapping pieces
    if result_geoms:

        # Extract all boundaries
        boundaries: List = []
        for geom in result_geoms:
            if hasattr(geom, "boundary"):
                boundaries.append(geom.boundary)

        # Polygonize the boundaries to get non-overlapping pieces
        polygons: List = list(polygonize(unary_union(boundaries)))

        # Filter out any empty or invalid polygons
        valid_polygons: List = [p for p in polygons if p.is_valid and not p.is_empty]

        # Create GeoDataFrame with the result
        result_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
            geometry=valid_polygons, crs=gdf.crs
        )

        # Final clean up
        result_gdf = result_gdf.explode(index_parts=False)
        result_gdf = result_gdf[result_gdf.is_valid]
        result_gdf["area"] = result_gdf.area
        return result_gdf

    return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)


def merge_small_polygons(
    gdf: gpd.GeoDataFrame, area_threshold: float = 1000
) -> gpd.GeoDataFrame:
    """Merge small polygons into large polygons based on longest shared boundary.

    Args:
        gdf: GeoDataFrame containing polygon geometries to merge.
        area_threshold: Area threshold to distinguish large vs small polygons.
            Polygons with area < threshold are considered small.

    Returns:
        GeoDataFrame with small polygons merged into adjacent large polygons.
    """

    # Calculate areas
    gdf = gdf.copy()
    gdf["area"] = gdf.geometry.area

    # Separate large and small polygons
    large_polygons: gpd.GeoDataFrame = gdf[gdf["area"] >= area_threshold].copy()
    small_polygons: gpd.GeoDataFrame = gdf[gdf["area"] < area_threshold].copy()

    # If no small polygons, return original
    if small_polygons.empty:
        return gdf

    # For each small polygon, find the large polygon with longest shared boundary
    unmerged_small: List = []

    for _, small_row in small_polygons.iterrows():
        small_geom = small_row.geometry
        max_shared_length: float = 0
        best_large_idx = None

        # Check shared boundary with each large polygon
        for large_idx, large_row in large_polygons.iterrows():
            large_geom = large_row.geometry

            # Calculate shared boundary length
            shared_boundary = small_geom.boundary.intersection(large_geom.boundary)
            if hasattr(shared_boundary, "length"):
                shared_length: float = shared_boundary.length
            else:
                shared_length: float = 0

            if shared_length > max_shared_length:
                max_shared_length = shared_length
                best_large_idx = large_idx

        # If shared boundary found, merge with best large polygon
        if best_large_idx is not None and max_shared_length > 0:
            # Merge small polygon with the best large polygon
            large_geom = large_polygons.loc[best_large_idx, "geometry"]
            merged_geom = large_geom.union(small_geom)
            large_polygons.loc[best_large_idx, "geometry"] = merged_geom
            large_polygons.loc[best_large_idx, "area"] = merged_geom.area
        else:
            # No shared boundary, preserve as is
            unmerged_small.append(small_row)

    # Combine results
    result_gdf: gpd.GeoDataFrame = large_polygons.copy()
    if unmerged_small:
        unmerged_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(unmerged_small, crs=gdf.crs)
        result_gdf = pd.concat([result_gdf, unmerged_gdf], ignore_index=True)
        result_gdf["area"] = result_gdf.area
    return result_gdf.reset_index(drop=True)


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
