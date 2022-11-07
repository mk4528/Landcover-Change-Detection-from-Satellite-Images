import utils
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import numpy as np


def nlcd_preprocess(img):
    # map from orignal ints to ints within 0 to 5
    remapper = lambda x: utils.nlcd_2_int_mapping[x]
    return np.vectorize(remapper)(img)


def get_customized_gen(naip_gen, nlcd_gen):
    while True:
        naip_im = next(naip_gen)
        nlcd_im = next(nlcd_gen)
        # keras requires that we one-hot-encode the labels
        # see https://stackoverflow.com/questions/45178513/how-to-load-image-masks-labels-for-image-segmentation-in-keras
        nlcd_im = keras.utils.to_categorical(nlcd_im, num_classes=len(utils.MAPPING))
        yield naip_im, nlcd_im


def create_train_val_dataset():
    naip_gen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1/255, # rescale
        validation_split=0.2,
    )
    
    nlcd_gen = ImageDataGenerator(
        preprocessing_function=nlcd_preprocess,
        validation_split=0.2,
    )
    
    naip_2013_gen_train = naip_gen.flow_from_directory(
        directory = utils.naip_datagen_train2013,
        target_size = (utils.train_input_size, utils.train_input_size),
        class_mode = None,
        color_mode = "rgba", # NAIP has 4 channels
        batch_size = utils.train_batch_size,
        seed = utils.train_seed,
        shuffle = True,
        subset="training",        
    )
    
    naip_2013_gen_val = naip_gen.flow_from_directory(
        directory = utils.naip_datagen_train2013,
        target_size = (utils.train_input_size, utils.train_input_size),
        class_mode = None,
        color_mode = "rgba", # NAIP has 4 channels
        batch_size = utils.train_batch_size,
        seed = utils.train_seed,
        shuffle = True,
        subset="validation",        
    )
    
    nlcd_2013_gen_train = nlcd_gen.flow_from_directory(
        directory = utils.nlcd_datagen_train2013,
        target_size = (utils.train_input_size, utils.train_input_size),
        class_mode = None,
        color_mode = "grayscale",
        batch_size = utils.train_batch_size,
        seed = utils.train_seed,
        shuffle = True,
        subset="training",
    )
    
    nlcd_2013_gen_val = nlcd_gen.flow_from_directory(
        directory = utils.nlcd_datagen_train2013,
        target_size = (utils.train_input_size, utils.train_input_size),
        class_mode = None,
        color_mode = "grayscale",
        batch_size = utils.train_batch_size,
        seed = utils.train_seed,
        shuffle = True,
        subset="validation",
    )
    
    return {
        "train_gen": get_customized_gen(naip_2013_gen_train, nlcd_2013_gen_train),
        "train_size": len(naip_2013_gen_train),
        "val_gen": get_customized_gen(naip_2013_gen_val, nlcd_2013_gen_val),
        "val_size": len(naip_2013_gen_val)
    }