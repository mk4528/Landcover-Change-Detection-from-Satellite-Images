from tensorflow import keras
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D


def FCN(input_size, classes):
    """ref: https://colab.research.google.com/drive/1VgwnN7jf-eZ12BaoyiuIKrqeWWprU7Nq?usp=sharing"""
    inputs = keras.Input(shape=(input_size, input_size, 4))
    x = Conv2D(filters=16, kernel_size=3, strides=1, padding="same", activation="gelu")(inputs)
    x = BatchNormalization()(x)    
    
    x = Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="gelu")(x)
    x = BatchNormalization()(x)
    
    outputs = Conv2D(filters=classes, kernel_size=1, strides=1, padding="same", activation="softmax")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="FCN")
    return model


def FCN_soft_label(input_size, classes):
    """ref: https://colab.research.google.com/drive/1VgwnN7jf-eZ12BaoyiuIKrqeWWprU7Nq?usp=sharing"""
    inputs = keras.Input(shape=(input_size, input_size, 4))
    x = Conv2D(filters=16, kernel_size=3, strides=1, padding="same", activation="gelu")(inputs)
    x = BatchNormalization()(x)    
    
    x = Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="gelu")(x)
    x = BatchNormalization()(x)
    
    outputs = Conv2D(filters=classes, kernel_size=1, strides=1, padding="same", activation="softmax")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="FCN")
    return model


# credit: https://www.youtube.com/watch?v=oBIkr7CAE6g&t=498s
def conv_block(input, num_filters):
    x = keras.layers.Conv2D(num_filters, 4, padding="same")(input)
    x = keras.layers.BatchNormalization()(x) # not in the original network
    x = keras.layers.Activation('relu')(x)
    
    x = keras.layers.Conv2D(num_filters, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x) # not in the original network
    x = keras.layers.Activation('relu')(x)

    return x

# Encoder block: Conv block followed by maxpooling
def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = keras.layers.MaxPool2D((2, 2))(x)
    return x, p


# Decoder block
# skip_features gets input from encoder for concatenation
def decoder_block(input, skip_features, num_filters):
    x = keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = keras.layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

# Build Unet
def UNet(input_shape, n_classes):
    inputs = keras.Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1025) # bridge

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    if n_classes == 1:
        activation = "sigmoid"
    else:
        activation = "softmax"
    
    outputs = Conv2D(n_classes, 1, padding="same", activation=activation)(d4) 

    model = keras.Model(inputs, outputs, name="U-Net")
    return model