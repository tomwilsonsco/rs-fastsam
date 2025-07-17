import rasterio as rio
import numpy as np
from rasterio.mask import mask
from typing import Union
from pathlib import Path
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier

def clip_and_normalize(X: np.ndarray) -> np.ndarray:
    """
    Clip and normalize spectral bands to their expected ranges.
    
    Args:
        X (numpy.ndarray): A 2D array where rows represent pixels and columns represent
            the 7 bands in the order ('Blue', 'Green', 'Red', 'RE_B6', 'NIR_B8', 'SWIR1', 'SWIR2').
    
    Returns:
        numpy.ndarray: A 2D array with clipped and normalized bands.
    """
    X = X.astype("float32")
    
    # Define clipping ranges for each band
    clip_ranges = [
        (0, 200),  # Blue
        (0, 200),  # Green  
        (0, 200),  # Red
        (0, 600),  # RE_B6
        (0, 600),  # NIR_B8
        (0, 400),  # SWIR1
        (0, 400),  # SWIR2
    ]
    
    # Apply clipping and normalization to each band
    X_normalized = np.zeros_like(X)
    
    for i, (min_val, max_val) in enumerate(clip_ranges):
        # Clip values to specified range
        X_clipped = np.clip(X[:, i], min_val, max_val)
        
        # Min-max normalize to [0, 1]
        X_normalized[:, i] = (X_clipped - min_val) / (max_val - min_val)
    
    return X_normalized


def calculate_indices(X: np.ndarray) -> np.ndarray:
    """
    Calculate spectral indices from input bands.

    Args:
        X (numpy.ndarray): A 2D array where rows represent pixels and columns represent
            the 7 bands in the order ('Blue', 'Green', 'Red', 'RE_B6', 'NIR_B8', 'SWIR1', 'SWIR2').

    Returns:
        numpy.ndarray: A 2D array where the original bands are combined with the calculated indices.
    """
    X = X.astype("float32")

    X_normalised = clip_and_normalize(X)

    # epsilon to avoid division by zero
    epsilon = 1e-8
    X_safe = X_normalised + epsilon

    ndvi = (X_safe[:, 4] - X_safe[:, 2]) / (X_safe[:, 4] + X_safe[:, 2])
    ndwi = (X_safe[:, 4] - X_safe[:, 5]) / (X_safe[:, 4] + X_safe[:, 5])
    nbr = (X_safe[:, 4] - X_safe[:, 6]) / (X_safe[:, 4] + X_safe[:, 6])
    evi2 = 2.5 * (X_safe[:, 4] - X_safe[:, 2]) / (X_safe[:, 4] + X_safe[:, 2] + 1)
    ndvi_re = (X_safe[:, 4] - X_safe[:, 3]) / (X_safe[:, 4] + X_safe[:, 3])

    indices = np.column_stack([ndvi, ndwi, nbr, evi2, ndvi_re])

    X_with_indices = np.hstack([X, indices])
    return X_with_indices


def classes_desc(class_code: int) -> str:
    """
    Convert classification codes to descriptions.

    Args:
        class_code (int): Classification code.

    Returns:
        str: Description corresponding to the classification code.
    """
    class_desc = {
        0: "Water",
        1: "Grassland",
        2: "Emerging crop",
        3: "Bare ground",
        4: "Mature crop",
        5: "Forest",
    }
    return class_desc.get(class_code, "not classified")


def classify_polygons(
    gdf: gpd.GeoDataFrame,
    image_path: Union[str, Path],
    rf_classifier: RandomForestClassifier,
) -> gpd.GeoDataFrame:
    """
    Classify polygons using median pixel values from image bands and derived indices.

    Args:
        gdf (geopandas.GeoDataFrame): Input GeoDataFrame containing polygons to classify.
        image_path (str or Path): Path to the raster image file.
        rf_classifier (sklearn.ensemble.RandomForestClassifier): Trained RandomForestClassifier.

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame with polygons to classify that intersect the image.
    """
    gdf_result = gdf.copy()
    predicted_classes = []

    with rio.open(image_path) as src:
        for idx, row in gdf_result.iterrows():
            try:
                geom = [row.geometry]
                masked_img, masked_transform = mask(src, geom, crop=True, filled=False)

                # Get valid pixels (not masked)
                valid_mask = ~masked_img.mask[0]

                if np.any(valid_mask):
                    # Extract pixel values for all bands
                    pixels = masked_img[:, valid_mask].T

                    pixels = np.asarray(pixels)

                    # Calculate indices
                    pixels_with_indices = calculate_indices(pixels)

                    median_values = np.median(pixels_with_indices, axis=0)

                    # Reshape for prediction
                    median_values = median_values.reshape(1, -1)

                    # Predict class
                    predicted_class = rf_classifier.predict(median_values)[0]
                    predicted_classes.append(predicted_class)

                else:
                    predicted_classes.append(-1)

            except Exception as e:
                predicted_classes.append(-1)

    # Add predicted class codes and descriptions to the result geodataframe
    gdf_result["predicted_class"] = predicted_classes
    gdf_result["predicted_class_desc"] = [classes_desc(p) for p in predicted_classes]

    return gdf_result
