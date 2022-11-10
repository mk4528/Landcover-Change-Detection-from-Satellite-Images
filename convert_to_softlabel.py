import os
import utils
import glob
import numpy as np
from skimage import io
from tqdm import tqdm


if __name__ == "__main__":
    NLCD_images = glob.glob(f"{utils.nlcd_train2013}/*.png")
    for nlcd in tqdm(NLCD_images):
        nlcd_number = os.path.splitext(nlcd)[0].split("/")[-1]
        nlcd_im = io.imread(nlcd) # 256 x 256
        h, w = nlcd_im.shape
        temp = np.vectorize(lambda x: utils.nlcd_softmap[x], signature='()->(n)')(nlcd_im).reshape(h, w, 4) # 256 x 256 x 4
        temp = np.rint(temp * 100) # times 100 so that we can save it as a png image
        temp = np.uint8(temp) # convert to uint8 so that skimage can save
        io.imsave(f"{utils.nlcd_train2013_soft}/{nlcd_number}.png", temp, check_contrast=False)
