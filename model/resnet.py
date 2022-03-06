import tensorflow as tf

from model.image_classifier_model import ImageClassifierModel


class ResnetModel(ImageClassifierModel):
    """
    This model fine-tunes a pretrained ResNet50V2.
    No need to rescale the data, pre-processing is done by the model.
    """

    def create_model_io(self, input_shape: tuple = (224, 224, 3), num_classes: int = 10):
        """
        Loads a pre-trained ResNet50V2.
        :return: the input and the output.
        """
        inp = tf.keras.Input(input_shape)
        resnet = tf.keras.applications.ResNet50V2(include_top=False, input_shape=input_shape, input_tensor=inp,
                                                  pooling="avg")

        # Add ResNet50V2 specific preprocessing method into the model.
        preprocessed = tf.keras.layers.Lambda(lambda x: tf.keras.applications.resnet_v2.preprocess_input(x))(inp)
        out = resnet(preprocessed)
        out = tf.keras.layers.Dense(num_classes, activation=None)(out)

        return inp, out
