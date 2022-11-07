import model
import tensorflow as tf
import create_training_dataset
import utils


# credit: https://github.com/tensorflow/tensorflow/issues/32875
# because the default implementation can't work, have to 
# implement ourselves
class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self, y_true=None, y_pred=None, num_classes=None, name=None, dtype=None):
        super(UpdatedMeanIoU, self).__init__(num_classes=num_classes, name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)


if __name__ == "__main__":
    res = create_training_dataset.create_train_val_dataset()

    for naip_im, nlcd_im in res["train_gen"]:
        print(naip_im.shape)
        print(nlcd_im.shape)
        break
    
    # input shape (batch, 256, 256, 4)
    # output shape (batch, 256, 256, 5)
    fcnModel = model.FCN(utils.train_input_size, 4)
    fcnModel.compile(
        optimizer='adam', 
        loss=tf.keras.losses.CategoricalCrossentropy(), # need to use categorical entropy
        metrics=["accuracy", 
                 tf.keras.metrics.AUC(),
                 tf.keras.metrics.MeanIoU(num_classes=4, sparse_y_true=False, sparse_y_pred=False),]
    )

    print(fcnModel.summary())
    
    history = fcnModel.fit(
        res["train_gen"],
        epochs=utils.train_epochs,
        validation_data=res["val_gen"],
        steps_per_epoch=res["train_size"] // utils.train_batch_size,
        validation_steps=res["val_size"] // utils.train_batch_size
    )
    
    fcnModel.save("./checkpoints/FCN_softlabeled_model.h5")
    
    # unetModel = model.UNet((utils.train_input_size, utils.train_input_size, 4), len(utils.MAPPING))
    
    # unetModel.compile(
    #     optimizer='adam', 
    #     loss=tf.keras.losses.CategoricalCrossentropy(), # need to use categorical entropy
    #     metrics=["accuracy", 
    #              tf.keras.metrics.AUC(),
    #              tf.keras.metrics.MeanIoU(num_classes=5, sparse_y_true=False, sparse_y_pred=False),]
    # )

    # print(unetModel.summary())
    
    # history = unetModel.fit(
    #     res["train_gen"],
    #     epochs=utils.train_epochs,
    #     validation_data=res["val_gen"],
    #     steps_per_epoch=res["train_size"] // utils.train_batch_size,
    #     validation_steps=res["val_size"] // utils.train_batch_size
    # )
    
    # unetModel.save("./checkpoints/unet_model.h5")