import pandas as pd 
import matplotlib.pyplot as plt
from skimage import io
import numpy as np

NLCD_CLASS = {
     0: "No Data",
    11: "Open Water",
    12: "Perennial Ice/Snow",
    21: "Developed, Open Space",
    22: "Developed, Low Intensity",
    23: "Developed, Medium Intensity",
    24: "Developed, High Intensity",
    31: "Barren Land (Rock/Sand/City)",
    41: "Deciduous Forest",
    42: "Evergreen Forest",
    43: "Mixed Forest",
    52: "Shrub/Scrub",
    71: "Grassland/Herbaceous",
    81: "Pasture/Hay",
    82: "Cultivated Crops",
    90: "Woody Wetlands",
    95: "Emergent Herbaceous Wetlands"
}

MAPPING = {
    "Water": [11],
    "Tree Canopy": [41, 42, 43, 52, 90, 95],
    "Low Vegetation": [21, 71, 81, 82],
    "Impervious": [22, 23, 24, 31],
    "No Data": [0, 12],
}


NLCD_IDX_TO_REDUCED_LC_MAP = np.array([
    4,#  0 No data 0
    0,#  1 Open Water
    4,#  2 Ice/Snow
    2,#  3 Developed Open Space
    3,#  4 Developed Low Intensity
    3,#  5 Developed Medium Intensity
    3,#  6 Developed High Intensity
    3,#  7 Barren Land
    1,#  8 Deciduous Forest
    1,#  9 Evergreen Forest
    1,# 10 Mixed Forest
    1,# 11 Shrub/Scrub
    2,# 12 Grassland/Herbaceous
    2,# 13 Pasture/Hay
    2,# 14 Cultivated Crops
    1,# 15 Woody Wetlands
    1,# 16 Emergent Herbaceious Wetlands
])


def get_image_ndarray(csv_path, col, row):
    df = pd.read_csv(csv_path)
    a = io.imread(df.loc[row, col])
    return a


def display_image(csv_path, col, row):
    """Display one tif image from csv file
    
    Args:
        csv_path: the path to the csv data file
        col: str, the name of the column
        row: int, row number

    Example:
        display_image("data/splits/val_inference_both.csv", "label_fn", 3)
    """
    a = get_image_ndarray(csv_path, col, row)
    plt.imshow(a)
    plt.show()


def get_tif_dimension(csv_path, col, row):
    """Get the dimension of a TIF image
    
    Args:
        csv_path: the path to the csv data file
        col: str, the column name 
        row: int, the row number

    Returns:
        the dimension of the TIF image
    """
    a = get_image_ndarray(csv_path, col, row)
    return a.shape

