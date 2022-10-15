import pandas as pd 
import matplotlib.pyplot as plt
from skimage import io

NLCD_CLASS = {
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
    51: "Dwarf Scrub",
    52: "Shrub/Scrub",
    71: "Grassland/Herbaceous",
    72: "Sedge/Herbaceous",
    73: "Lichens",
    74: "Moss",
    81: "Pasture/Hay",
    82: "Cultivated Crops",
    90: "Woody Wetlands",
    95: "Emergent Herbaceous Wetlands"
}


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

