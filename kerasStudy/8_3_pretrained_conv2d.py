from tensorflow import keras
import os, shutil, pathlib
from tensorflow.keras import layers

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

conv_base = keras.applications.vgg16.VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(180, 180, 3)
)
# print(conv_base.summary())

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
    x = layers.Flatten()(inputs)
    x = layers.Dense(256)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="data/feature_extraction.keras",
            save_best_only=True,
            monitor="val_loss"
        )
    ]
    history = model.fit(
        train_features, train_labels,
        epochs=20,
        validation_data=(val_features, val_labels),
        callbacks=callbacks
    )

# no_extension()

conv_base = keras.applications.vgg16.VGG16(
    weights="imagenet",
    include_top=False
)
conv_base.trainable = False
def with_extension():
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.2),
        ]
    )
    inputs = keras.Input(shape=(180,180,3))
    x = data_augmentation(inputs)
    x = keras.applications.vgg16.preprocess_input(x)
    x = conv_base(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="data/feature_extraction_with_data_augmentation.keras",
            save_best_only=True,
            monitor="val_loss"
        )
    ]
    history = model.fit(
        train_dataset,
        epochs=50,
        validation_data=validation_dataset,
        callbacks=callbacks
    )
# with_extension()

def fine_tuning():
    model = keras.models.load_model("data/feature_extraction_with_data_augmentation.keras")
    conv_base.trainable = True
    for layer in conv_base.layers[:-4]:
        layer.trainable = False

    model.compile(loss="binary_crossentropy",
                  optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),
                  metrics=["accuracy"])

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="data/fine_tuning.keras",
            save_best_only=True,
            monitor="val_loss"
        )
    ]
    history = model.fit(
        train_dataset,
        epochs=30,
        validation_data=validation_dataset,
        callbacks=callbacks
    )

# fine_tuning()

def fine_tuning_test():
    model = keras.models.load_model("data/fine_tuning.keras")
    test_loss, test_acc = model.evaluate(test_dataset)
    print(test_acc)

fine_tuning_test()