import utils
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import numpy as np


def nlcd_preprocess(img):
    # map from orignal ints to ints within 0 to 4 (5 classes in total)
    remapper = lambda x: utils.nlcd_2_int_mapping[x]
    return np.vectorize(remapper)(img)


# def nlcd_preprocess_soft_label(img):
#     img = img.reshape((img.shape[0], img.shape[1]))
#     res = []
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             res.append(utils.nlcd_softmap[img[i][j]]) # because it's 3D (256, 256, 1)
#     return np.array(res).reshape(img.shape[0], img.shape[1], 4)
    

def get_customized_gen(naip_gen, nlcd_gen):
    while True:
        naip_im = next(naip_gen)
        nlcd_im = next(nlcd_gen)
        # keras requires that we one-hot-encode the labels
        # see https://stackoverflow.com/questions/45178513/how-to-load-image-masks-labels-for-image-segmentation-in-keras
        nlcd_im = keras.utils.to_categorical(nlcd_im, num_classes=len(utils.MAPPING))
        yield naip_im, nlcd_im


def get_softlabeled_gen(naip_gen, nlcd_gen):
    while True:
        naip_batch = next(naip_gen)
        nlcd_batch = next(nlcd_gen) # one batch
        yield naip_batch, nlcd_batch
    

def create_train_val_dataset():
    naip_gen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1.0/255, # rescale
        validation_split=0.2,
    )
    
    nlcd_gen = ImageDataGenerator(
        # rescale=1.0/100, # for soft labels
        validation_split=0.2,
        preprocessing_function=nlcd_preprocess,
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
        # directory = utils.nlcd_datagen_train2013_softlabel, # for soft labels
        target_size = (utils.train_input_size, utils.train_input_size),
        class_mode = None,
        color_mode = "grayscale",
        # color_mode = "rgba", # for soft labels
        batch_size = utils.train_batch_size,
        seed = utils.train_seed,
        shuffle = True,
        subset="training",
    )
    
    nlcd_2013_gen_val = nlcd_gen.flow_from_directory(
        directory = utils.nlcd_datagen_train2013,
        # directory = utils.nlcd_datagen_train2013_softlabel, # for soft labels
        target_size = (utils.train_input_size, utils.train_input_size),
        class_mode = None,
        color_mode = "grayscale",
        # color_mode = "rgba", # for soft labels
        batch_size = utils.train_batch_size,
        seed = utils.train_seed,
        shuffle = True,
        subset="validation",
    )
    
    return {
        # "train_gen": get_softlabeled_gen(naip_2013_gen_train, nlcd_2013_gen_train), # for soft labels
        "train_gen": get_customized_gen(naip_2013_gen_train, nlcd_2013_gen_train),
        "train_size": len(naip_2013_gen_train),
        # "val_gen": get_softlabeled_gen(naip_2013_gen_val, nlcd_2013_gen_val), # for soft labels
        "val_gen": get_customized_gen(naip_2013_gen_val, nlcd_2013_gen_val),
        "val_size": len(naip_2013_gen_val)
    }