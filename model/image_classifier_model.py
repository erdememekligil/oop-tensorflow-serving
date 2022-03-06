import abc
import tensorflow as tf


class ImageClassifierModel(tf.keras.Model):
    """
    A base class for image classification models. Subclasses must implement create_model_io function that returns the
    input and the output of the model.
    SavedModel of this class will have two serving signatures. The default one (serving_default) calculates predictions
    using images as 4d arrays. The other signature, serving_bytes, operates on base64 encoded image bytes.
    """

    def __init__(self, input_shape: tuple = (224, 224, 3), num_classes: int = 10, *args, **kwargs):
        inp, out = self.create_model_io(input_shape, num_classes)
        kwargs = {**kwargs, **{"inputs": inp, "outputs": out}}
        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def create_model_io(self, input_shape: tuple = (224, 224, 3), num_classes: int = 10):
        pass

    def call(self, inputs, training=None, mask=None):
        return super(ImageClassifierModel, self).call(inputs, training=training, mask=mask)

    def get_config(self):
        return super(ImageClassifierModel, self).get_config()

    def save(self, filepath, overwrite=True, include_optimizer=True, save_format=None, signatures=None, options=None,
             save_traces=True):
        """
        Saves model with custom signatures.
        serving_default: predict using 4 array image (numpy/tensor).
        serving_bytes: predict using base64 encoded image bytes.
        """
        if signatures is None:
            signatures = dict()
        signatures["serving_default"] = self.predict_numpy_image
        signatures["serving_bytes"] = self.predict_bytes_image
        super().save(filepath, overwrite, include_optimizer, save_format, signatures, options, save_traces)

    @tf.function(input_signature=[tf.TensorSpec(name="image_bytes_string", shape=None, dtype=tf.string)])
    def predict_bytes_image(self, image):
        """
        Predict using encoded image bytes.
        :param image: png, jpeg, bmp, gif encoded image bytes.
        :return: prediction result.
        """
        image = tf.reshape(image, [])
        image = tf.io.decode_image(image, channels=3, dtype=tf.uint8, expand_animations=False)
        image = tf.expand_dims(image, 0)

        return self.call(image)

    @tf.function(input_signature=[tf.TensorSpec(name="input_tensor", shape=(None, None, None, 3), dtype=tf.float32)])
    def predict_numpy_image(self, inputs):
        """
        Predict using 4d array image (numpy/tensor).
        :param inputs: 4d array image (batch, height, width, channel).
        :return: prediction result.
        """
        return self.call(inputs)
