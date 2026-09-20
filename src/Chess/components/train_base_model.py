import tensorflow as tf
import numpy as np
from Chess.config.configuration import PrepareTrainingConfig
from Chess.components.prepare_base_model import PrepareBaseModel

import os
from  Chess import logger

class TrainBaseModel:
    def __init__(self , config: PrepareTrainingConfig):
        self.config = config


    def get_base_model(self):
       self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
       self.model.compile(
           optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
           loss="categorical_crossentropy",
           metrics=["accuracy"]
       )
       return self.model

    def get_Data(self):
        x_numpy = os.path.join("artifacts" , "data_transformation" , "processed" , "X.npy")
        y_numpy = os.path.join("artifacts" , "data_transformation" , "processed" , "Y.npy")
        x = np.load(x_numpy)
        y = np.load(y_numpy)
        return x , y


    def Train_Base_Model(self):
        x, y = self.get_Data()
        self.get_base_model()
        self.model.fit(
            x,
            y,
            batch_size=self.config.params_batch_size ,
            epochs=self.config.params_epochs,
            validation_split=self.config.params_validation_split,
            shuffle=self.config.params_shuffle
        )
        os.makedirs(os.path.dirname(self.config.trained_model_path), exist_ok=True)
        self.model.save(self.config.trained_model_path)
        logger.info(f"Trained model saved at: {self.config.trained_model_path}")



