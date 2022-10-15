import utils

TEST_FILE = "data/splits/val_inference_both.csv"
IMAGE_COL = "image_fn"
LABEL_COL = "label_fn"
GROUP_COL = "gropu"
ROW = 3

if __name__ == "__main__":
    # utils.display_image(TEST_FILE, COL, ROW)
    # both training NAIP images and NLCD labels are 3880 x 3880
    # let's say I want to view a NLCD label at a specific location
    img = utils.get_image_ndarray(TEST_FILE, LABEL_COL, 3)
