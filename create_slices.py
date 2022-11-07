import os
import utils
import shutil
import re
from skimage import io
import pandas as pd


def rerun(naip_train, nlcd_train) -> bool:
    """Remove all directories and subdirectories if want to rerun"""
    if os.path.exists(naip_train):
        shutil.rmtree(naip_train)
    if os.path.exists(nlcd_train):
        shutil.rmtree(nlcd_train)


def create_dirs(naip_train, nlcd_train):
    if not os.path.exists(naip_train):
        os.makedirs(naip_train)    
    if not os.path.exists(nlcd_train):
        os.makedirs(nlcd_train)


def get_image_num(img_path) -> int:
    """Extract image number from img_path
    E.g. 496_naip-2013.tif -> 496
    """
    m = re.findall("([\d]+)_naip-[\d]+", img_path)
    if m:
        return m[0]


def create_naip_nlcd_slices(train_csv_path, stride, sample_size,
                            input_size, naip_outdir, nlcd_outdir):
    """Read training_set_naip_nlcd_both.csv, create slices from 
    NAIP 4D images and NLCD 2D images, save to directories

    Args:
        train_csv_path (str): path to the training csv
        year (int): 2013/2016
        stride (int): _description_
        sample_size (int): how many slices
        input_size (int): _description_
        naip_outdir (str): _description_
        nlcd_outdir (str): _description_
    """
    df = pd.read_csv(train_csv_path)
    total_count = 1
    for i in range(len(df)):
        naip_im = io.imread(df.iloc[i]['image_fn']) # RGBI image, 4 channels
        nlcd_im = io.imread(df.iloc[i]['label_fn']) # 2D image
        img_num = get_image_num(df.iloc[i]['image_fn'])
        print(f"Begin Slicing NAIP/NLCD {img_num}...")
        assert naip_im.shape[0] == nlcd_im.shape[0]
        assert naip_im.shape[1] == nlcd_im.shape[1]
        y, counter = 0, 1 
        h, w = naip_im.shape[0], naip_im.shape[1]
        while y * stride + input_size < h:
            x = 0
            while x * stride + input_size < w:
                if total_count > sample_size:
                    print("Finished!")
                    return
                naip_im_slice = naip_im[y*stride: y*stride+input_size, x*stride: x*stride+input_size]
                nlcd_im_slice = nlcd_im[y*stride: y*stride+input_size, x*stride: x*stride+input_size]
                # have to save NAIP as png because jpg doesn't support alpha channel
                io.imsave(f"{naip_outdir}/{img_num}_{counter}.png", naip_im_slice)
                # turn off check_contrast warning because NLCD is saved as 2D and easily 
                # the contrast is low
                io.imsave(f"{nlcd_outdir}/{img_num}_{counter}.png", nlcd_im_slice, check_contrast=False)
                x += 1
                counter += 1
                total_count += 1
            y += 1
        print(f"NAIP/NLCD {img_num} Finished!")
        print(f"Finished {total_count}, {sample_size - total_count} left")
    print("Finished!")
        

if __name__ == "__main__":
    rerun(utils.naip_train2013, utils.nlcd_train2013)
    create_dirs(utils.naip_train2013, utils.nlcd_train2013)
    
    create_naip_nlcd_slices(utils.train_naip_nlcd_2013, 
                            utils.train_stride, 
                            utils.train_sample, 
                            utils.train_input_size, 
                            naip_outdir=utils.naip_train2013, 
                            nlcd_outdir=utils.nlcd_train2013)
