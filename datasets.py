import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import DatasetV2


def create_train_val_datasets(folder_path: str, image_shape: tuple = (224, 224), batch_size: int = 32,
                              split_ratio: float = 0.2) -> (DatasetV2, DatasetV2):
    """
    Creates training and validation datasets by randomly splitting training data into two parts.
    The data will be shuffled before it gets split into two. Each batch will also be shuffled at each epoch.
    The split of the training and validation sets will stay same between runs since a constant seed (42) is used.
    :param folder_path: Should contain subdirectories named with class names.
    These subdirectories should contain images.
    :param image_shape: Images will be resized in to this shape. (224, 224) by default.
    :param batch_size: Mini-batch size. Default: 32.
    :param split_ratio: Training data will be split into train and validation by this ratio. Default: 0.2.
    :return: train and validation datasets.
    """
    image_dataset_from_directory = tf.keras.preprocessing.image_dataset_from_directory
    train_dataset = image_dataset_from_directory(folder_path, image_size=image_shape, shuffle=True, seed=42,
                                                 batch_size=batch_size, validation_split=split_ratio, subset='training')
    val_dataset = image_dataset_from_directory(folder_path, image_size=image_shape, shuffle=True, seed=42,
                                               batch_size=batch_size, validation_split=split_ratio, subset='validation')

    return train_dataset, val_dataset


def create_test_dataset(folder_path: str, image_shape: tuple = (224, 224), batch_size: int = 32) -> DatasetV2:
    """
    Create test dataset. This dataset will not be shuffled since it is the test set.
    :param folder_path: Should contain subdirectories named with class names.
    These subdirectories should contain images.
    :param image_shape: Images will be resized in to this shape. (224, 224) by default.
    :param batch_size: Mini-batch size. Default: 32.
    :return: test dataset.
    """
    return tf.keras.preprocessing.image_dataset_from_directory(folder_path, image_size=image_shape, shuffle=False,
                                                               batch_size=batch_size)
