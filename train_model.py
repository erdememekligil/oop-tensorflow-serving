import argparse
import os
import sys
from typing import Type
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import DatasetV2
from model.image_classifier_model import ImageClassifierModel
from datasets import create_train_val_datasets, create_test_dataset
from utils import set_gpu_memory_growth
from model.lenet import LenetModel
from model.resnet import ResnetModel


def train(model_fn: Type[ImageClassifierModel], train_dataset: DatasetV2, val_dataset: DatasetV2,
          model_path: str, image_shape: tuple = (224, 224), num_epochs: int = 200, patience: int = 5) -> tf.keras.Model:
    input_shape = image_shape + tuple([3])  # 3rd dimension. RGB

    model = model_fn(input_shape=input_shape)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    model.summary()

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True),
                 tf.keras.callbacks.ModelCheckpoint(model_path, 'val_accuracy', save_best_only=True)]

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=val_dataset,
        callbacks=callbacks
    )

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_path', type=str, required=True,
                        help="Training dataset path. Example: data_directory/train")
    parser.add_argument('--test_dataset_path', type=str, required=True,
                        help="Test dataset path. Example: data_directory/test")
    parser.add_argument('--model_folder_path', type=str, required=True,
                        help="Trained model will be saved in this folder with given model name and version. "
                             "Example: model_folder/ResnetModel/1/")
    parser.add_argument('--model_name', type=str, default="ResnetModel",
                        help="model_name must be a subclass of ImageClassifierModel. "
                             "This class must be imported in this script.")
    parser.add_argument('--model_version', type=int, default=1,
                        help="Tensorflow Serving requires model files to be in directories named as version numbers."
                             "Trained models will be saved in model_name/version/ directories. Example: ResnetModel/1/")
    parser.add_argument('--input_width', type=int, default=224,
                        help="Width of the model input and the data. "
                             "The data will be resized if its size is different.")
    parser.add_argument('--input_height', type=int, default=224,
                        help="Height of the model input and the data. "
                             "The data will be resized if its size is different.")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for training.")
    parser.add_argument('--num_epochs', type=int, default=200,
                        help="Number of maximum epochs. If the model converges before this number of epochs, the "
                             "training will be stopped early.")
    parser.add_argument('--patience', type=int, default=5,
                        help="How many epochs to count to conclude that the model is converged before the "
                             "training stops early.")

    try:
        args = parser.parse_args()
    except Exception:
        parser.print_help()
        sys.exit(0)

    model_fn = globals()[args.model_name]  # Model class must be imported before this line.
    model_path = os.path.join(args.model_folder_path, model_fn.__name__, str(args.model_version))
    image_shape = (args.input_height, args.input_width)

    if not issubclass(model_fn, ImageClassifierModel):
        raise ValueError("Model not recognized. It must be a subclass of ImageClassifierModel.")

    set_gpu_memory_growth()  # limit gpu memory consumption

    train_dataset, val_dataset = create_train_val_datasets(args.train_dataset_path, image_shape=image_shape,
                                                           batch_size=args.batch_size)
    print("Train and validation datasets are created.")

    model = train(model_fn, train_dataset, val_dataset, model_path, image_shape, args.num_epochs, args.patience)

    test_dataset = create_test_dataset(args.test_dataset_path, image_shape=image_shape, batch_size=args.batch_size)
    print("Test dataset is created.")

    model.evaluate(test_dataset, verbose=1)


if __name__ == '__main__':
    main()
