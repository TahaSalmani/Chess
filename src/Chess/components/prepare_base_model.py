import os

import tensorflow as tf
from tensorflow.keras import layers
from Chess.config.configuration import PrepareBaseModelConfig
from pathlib import Path



class PrepareBaseModel :
    def __init__(self, config : PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.Sequential([

            layers.Input(shape=tuple(self.config.params_image_size), name="board_input"),

            layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
            layers.BatchNormalization(),

            layers.Conv2D(128, kernel_size=3, padding="same", activation="relu"),
            layers.BatchNormalization(),

            layers.Conv2D(128, kernel_size=3, padding="same", activation="relu"),
            layers.BatchNormalization(),

            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(self.config.params_classes, activation="linear", name="position_eval")
        ])

        self.save_model (path=self.config.model_path, model=self.model)
        return self.model

    def update_model(self):
        self.full_model = self.model

        self.full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=["mae"]

        )
        self.full_model.summary()
        self.save_model(path=self.config.updated_base_model_path , model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)


