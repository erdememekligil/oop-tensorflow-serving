import tensorflow as tf

from model.image_classifier_model import ImageClassifierModel


class LenetModel(ImageClassifierModel):
    """
    Lenet: A very basic CNN architecture.
    No need to rescale the data, pre-processing is done by the model.
    """

    def create_model_io(self, input_shape: tuple = (28, 28, 3), num_classes: int = 10):
        """
        Creates a lenet convolutional image classifier model.
        :return: the input and the output.
        """
        inp = tf.keras.Input(input_shape)

        out = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)(inp)
        out = tf.keras.layers.Conv2D(6, 5, activation='tanh')(out)
        out = tf.keras.layers.AveragePooling2D(2)(out)
        out = tf.keras.layers.Conv2D(16, 3, activation='tanh')(out)
        out = tf.keras.layers.AveragePooling2D(2)(out)
        out = tf.keras.layers.Conv2D(120, 3, activation='tanh')(out)
        out = tf.keras.layers.Flatten()(out)
        out = tf.keras.layers.Dense(84, activation='tanh')(out)
        out = tf.keras.layers.Dense(num_classes, activation=None)(out)

        return inp, out
