from tensorflow import keras
import os, shutil, pathlib
from tensorflow.keras import layers

conv_base = keras.applications.vgg16.VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(180, 180, 3)
)
# print(conv_base.summary())


new_base_dir = pathlib.Path("data/cats_vs_dogs_small")
from tensorflow.keras.utils import image_dataset_from_directory

train_dataset = image_dataset_from_directory(
    new_base_dir / "train",
    image_size=(180, 180),
    batch_size=32
)
validation_dataset = image_dataset_from_directory(
    new_base_dir / "validation",
    image_size=(180, 180),
    batch_size=32
)
test_dataset = image_dataset_from_directory(
    new_base_dir /  "test",
    image_size=(180, 180),
    batch_size=32
)

import numpy as np


def get_features_and_labels(dataset):
    all_features = []
    all_labels = []
    for images, labels in dataset:
        preprocessed_images = keras.applications.vgg16.preprocess_input(images)
        features = conv_base.predict(preprocessed_images)
        all_features.append(features)
        all_labels.append(labels)
    return np.concatenate(all_features), np.concatenate(all_labels)

def no_extension():
    train_features, train_labels = get_features_and_labels(train_dataset)
    test_features, test_labels = get_features_and_labels(test_dataset)
    val_features, val_labels = get_features_and_labels(validation_dataset)
    print(train_features.shape)

    inputs = keras.Input(shape=(5, 5, 512))


