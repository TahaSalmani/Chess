import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from Chess import logger
from Chess.components.prepare_base_model import PrepareBaseModel
from Chess.config.configuration import PrepareTrainingConfig
from tensorflow.keras.callbacks import TensorBoard

class TrainBaseModel:

    def __init__(self, config: PrepareTrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.params_learning_rate
            ),
            loss="sparse_categorical_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.SparseTopKCategoricalAccuracy(
                    k=5, name="top_5_acc"
                ),
            ],
        )
        return self.model

    def get_Data(self):
        x_numpy = os.path.join(
            "artifacts", "data_transformation", "processed", "X.npy"
        )
        y_numpy = os.path.join(
            "artifacts", "data_transformation", "processed", "Y.npy"
        )
        x = np.load(x_numpy)
        y = np.load(y_numpy)
        return x, y

    def Train_Base_Model(self):
        x, y = self.get_Data()
        self.get_base_model()
        log_dir = os.path.join("artifacts", "tensorboard_logs", "fit")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                verbose=1,
                restore_best_weights=True,
                patience=10,
            ),
            ReduceLROnPlateau(

                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-5,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=self.config.trained_model_path,
                monitor="val_loss" ,
                save_best_only=True,
                verbose=1,
            ),
            tensorboard_callback
        ]

        logger.info("Starting base model fine-tuning...")

        history = self.model.fit(
            x,
            y,
            batch_size=self.config.params_batch_size,
            epochs=self.config.params_epochs,
            validation_split=self.config.params_validation_split,
            shuffle=self.config.params_shuffle,
            callbacks=callbacks,
        )

        os.makedirs(
            os.path.dirname(self.config.trained_model_path), exist_ok=True
        )
        self.model.save(self.config.trained_model_path)
        logger.info(f"trained model saved at: {self.config.trained_model_path}")