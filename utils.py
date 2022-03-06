import tensorflow as tf
from PIL import Image
import numpy as np
import io


def set_gpu_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def numpy_to_image(img: np.array) -> Image:
    return Image.fromarray(img.astype('uint8'), 'RGB')


def image_to_byte_array(image: Image) -> bytes:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")

    return img_byte_arr.getvalue()


def numpy_to_bytes(img: np.array) -> bytes:
    img = numpy_to_image(img)
    return image_to_byte_array(img)
