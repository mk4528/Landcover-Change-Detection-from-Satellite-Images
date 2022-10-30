"""Download TIF images. """
import pandas as pd
import os
from tqdm import trange


training_both = "data/splits/training_set_naip_nlcd_both.csv"
training_both_df = pd.read_csv(training_both)


def download_nlcd_images(df, img_year: int, img_number: int, output_dir: str):
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
    

def download_nlcd_images_batch(df, img_year: int, num: int, naip_output_dir, nlcd_output_dir):
    """Download num TIF images from img_year """
    # 2013 - group 0
    # 2016 - group 1
    group = 0 if img_year == 2013 else 1
    df = df[df['group'] == group]
    for i in trange(num):    
        os.system(f"wget -P {naip_output_dir} {df.iloc[i]['image_fn']}")
        os.system(f"wget -P {nlcd_output_dir} {df.iloc[i]['label_fn']}")

if __name__ == "__main__":
    df = training_both_df
    img_year = 2013
    num = 20
    # img_number = 3716
    naip_output_dir = "TIFimages/NAIP_2013"
    nlcd_output_dir = "TIFimages/NLCD_2013"
    # download_tif_images(df, img_year, img_number, output_dir)
    download_nlcd_images_batch(df, img_year, num, naip_output_dir, nlcd_output_dir)