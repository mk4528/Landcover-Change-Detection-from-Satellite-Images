import utils

TEST_FILE = "data/splits/val_inference_both.csv"
COL = "label_fn"
ROW = 3

if __name__ == "__main__":
    utils.display_image(TEST_FILE, COL, ROW)