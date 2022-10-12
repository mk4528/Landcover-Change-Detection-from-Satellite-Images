import pandas as pd 
import matplotlib.pyplot as plt
from skimage import io


def display_image(csv_path, col, row):
    """Display one tif image from csv file
    
    Args:
        csv_path: the path to the csv data file
        col: str, the name of the column
        row: int, row number

    Example:
        display_image("data/splits/val_inference_both.csv", "label_fn", 3)
    """
    df = pd.read_csv(csv_path)
    a = io.imread(df.loc[row, col])
    plt.imshow(a)
    plt.show()
