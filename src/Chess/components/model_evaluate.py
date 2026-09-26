import json

from Chess import logger
import tensorflow as tf
from pathlib import Path
from Chess.config.configuration import PrepareEvaluationConfig

from Chess.utils.common import save_json

class ModelEvaluate:
    def __init__(self , config : PrepareEvaluationConfig):
        self.config = config

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self, X_test, y_test):
        model = self.load_model(self.config.trained_model_path)

        logger.info("Evaluating model on test dataset...")
        self.score = model.evaluate(X_test, y_test)



    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}

        save_path = self.config.evaluate_scores

        with open(save_path, "w") as f:
            json.dump(scores, f, indent=4)

        save_json(path=Path("scores.json"), data=scores)

        logger.info(f"Scores saved successfully at {save_path}: {scores}")








