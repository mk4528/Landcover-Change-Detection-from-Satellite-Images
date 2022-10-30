"""Download TIF images. """
import pandas as pd
import os 


training_both = "data/splits/training_set_naip_nlcd_both.csv"
training_both_df = pd.read_csv(training_both)


def download_tif_images(df, img_year: int, img_number: int, output_dir: str):
    """Download TIF image from the URL and save it to output_dir
    
    Args:
        df: a DataFrame that has the "label_fn" column
        img_year: int, e.g. 2013
        img_number: int e.g. 3716
        output_dir: name of the directory to save the image
    """
    for i in range(len(df)):
        if f"{img_number}_nlcd-{img_year}" in df.loc[i, "label_fn"]:
            os.system(f"wget -P {output_dir} {df.loc[i, 'label_fn']}")
    

def download_tif_images_batch(df, img_year: int, num: int, output_dir: str):
    """Download num TIF images from img_year """
    for i in range(len(df)):
        if num > 0:
            if f"_nlcd-{img_year}" in df.loc[i, "label_fn"]:
                os.system(f"wget -P {output_dir} {df.loc[i, 'label_fn']}")
                num -= 1
        else:
            break


if __name__ == "__main__":
    df = training_both_df
    img_year = 2016
    img_number = 3716
    output_dir = "TIFimages"
    # download_tif_images(df, img_year, img_number, output_dir)
    download_tif_images_batch(df, 2013, 100, "TIFimages")