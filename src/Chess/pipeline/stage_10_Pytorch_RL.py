import os
import torch
from Chess import logger
from Chess.config.configuration import ConfigurationManager
from Chess.components.policy_value_torch_model import ValuePolicyNet
from Chess.components.train_rl import train_rl_step

STAGE_NAME = "Train Reinforcement Learning Stage"


class TrainRLPipeline:
    def __init__(self):
        pass

    def main(self):
        logger.info("Fetching configuration for RL training...")
        config_manager = ConfigurationManager()

        base_model_config = config_manager.get_torch_base_model()
        training_config = config_manager.get_torch_training_config()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        model = ValuePolicyNet(base_model_config)

        weights_path = training_config.trained_model_path
        if os.path.exists(weights_path):
            logger.info(f"Loading existing weights from: {weights_path}")
            model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
        else:
            logger.info("No pre-trained weights found. Starting from base model weights.")

        self_play_data_dir = os.path.join("artifacts", "self_play_data")

        logger.info(f"Starting RL training step using data from {self_play_data_dir}...")
        train_rl_step(
            model=model,
            data_dir=self_play_data_dir,
            epochs=training_config.params_epochs,
            batch_size=training_config.params_batch_size,
            lr=training_config.params_learning_rate,
            device=device
        )


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        obj = TrainRLPipeline()
        obj.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed successfully! <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e