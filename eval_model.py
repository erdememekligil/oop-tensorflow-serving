import argparse
import sys

import tensorflow as tf
import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.python.data.ops.dataset_ops import DatasetV2
from datasets import create_test_dataset
from utils import set_gpu_memory_growth, numpy_to_bytes


def calculate_accuracy(test_dataset: DatasetV2, model) -> float:
    y_list = []
    pred_list = []

    for batch_x, batch_y in tqdm.tqdm(test_dataset):
        prob = model(batch_x)
        pred = np.argmax(prob.numpy(), axis=1)

        y_list.extend(batch_y.numpy())
        pred_list.extend(pred)

    acc = accuracy_score(y_list, pred_list)
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dataset_path', type=str, required=True,
                        help="Training dataset path. Example: data_directory/test")
    parser.add_argument('--model_path', type=str, required=True,
                        help="Model path should include version directory (if exists). Example: .../ResnetModel/1")

    try:
        args = parser.parse_args()
    except Exception:
        parser.print_help()
        sys.exit(0)

    set_gpu_memory_growth()

    print("Loading keras model", args.model_path)
    # We can load a Keras based SavedModel via both keras and tf.saved_model utilities.
    # Keras load_model function cannot interpret the "tf" in Lambda layer. So, we must add tf as custom_objects.
    keras_model = tf.keras.models.load_model(args.model_path, custom_objects={"tf": tf})

    image_shape = keras_model.input_shape[1: 3]  # (w, h) from (batch, w, h, n_dim)

    test_dataset = create_test_dataset(args.test_dataset_path, image_shape=image_shape)
    print("Test dataset is created.")

    # Use Keras evaluate function to calculate test accuracy and loss.
    keras_model.evaluate(test_dataset, verbose=1)

    # or we can work with tf.saved_model and manually calculate accuracy.
    model = tf.saved_model.load(args.model_path)

    acc = calculate_accuracy(test_dataset, model)
    print("accuracy:", acc)

    # Load sample
    sample = next(test_dataset.__iter__())[0][:1].numpy()  # (1, h, w, ch)
    png_bytes = numpy_to_bytes(sample[0])  # convert numpy array to png bytes.

    # @tf.functions can be called using both of the loaded models.
    print("keras model predict_numpy_image", keras_model.predict_numpy_image(sample))
    print("keras model predict_bytes_image", keras_model.predict_bytes_image(png_bytes))

    print("saved model predict_numpy_image", model.predict_numpy_image(sample))
    print("saved model predict_bytes_image", model.predict_bytes_image(png_bytes))


if __name__ == '__main__':
    main()
